"""
shadow_detector.py
==================
Direction-Agnostic Shadow Detection for Backward-Pull-Harvest Inpainting.

DESIGN GOALS:
  1. No light direction assumption — works for overhead sun, low angle, indoor.
  2. Per-view: each source camera sees shadow at a different position/angle.
  3. Conservative: prefer missing a faint shadow over excluding real background.
  4. Robust when shadow is tiny (overhead sun): shadow_mask is just empty → no harm.

METHOD 1 — Source-side shadow exclusion  (compute_shadow_masks):
  For each source view:
    a. Sample "safe" background V (brightness) from far-outside-mask pixels.
    b. Create a search ring around the mask: mask_edge → mask_edge + search_px.
    c. Shadow candidates = ring pixels where V < bg_median - thresh_k * bg_MAD.
    d. Connectivity filter: keep only blobs that TOUCH the object mask.
    e. Sanity cap: if >40% of ring is "shadow", it's just a dark scene → skip.
    f. Merge into view["mask_dilated"] → pull harvest skips shadow pixels automatically.

METHOD 2 — Target-side brightness untrust  (brightness_untrust_filter):
  Run after pull harvest + Strategy B, before Fix A:
    a. Get target outer ring V stats (safe background brightness reference).
    b. For each trusted pulled pixel, check its V in canvas.
    c. If V < bg_median - bright_untrust_k * bg_MAD → force untrust.
    d. Flows into dead_mask → LaMa fills instead of shadow contaminating output.

WHY DIRECTION-AGNOSTIC HANDLES OVERHEAD SUN:
  Overhead sun → shadow is tiny, directly under object → often INSIDE the SAM mask
  already.  If not, it's within a few pixels of the mask edge → the search ring
  still catches it (the ring starts at mask edge, not far away).
  If shadow is literally 0 px → shadow_mask is empty → no change. ✓

TUNING:
  search_px    70 covers most indoor/outdoor shadows. Use 100-150 for very long shadows.
  thresh_k     1.5 moderate. 1.0 aggressive. 2.0 conservative.
  bright_k     2.5 conservative target-side filter. 2.0 moderate.
"""

import cv2
import numpy as np


# ============================================================================
# Internal helpers
# ============================================================================
def _detect_shadow_mask_for_view(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    search_px: int = 70,
    safe_margin_px: int = 20,
    thresh_k: float = 1.5,
    min_shadow_blob_px: int = 150,
    connectivity_touch_px: int = 20,
    max_ring_shadow_ratio: float = 0.40,
) -> np.ndarray:
    """
    Detect cast shadow for one source view. Returns shadow_mask (uint8 255=shadow).

    Algorithm
    ---------
    1. Sample "safe background" V stats from pixels far (search_px + safe_margin_px)
       outside the mask.
    2. Search ring = band between mask edge and mask + search_px.
    3. Shadow candidates = ring pixels with V < bg_med - thresh_k * bg_MAD.
    4. Connectivity filter: keep only blobs touching the mask (cast shadow is adjacent).
    5. Sanity check: if > max_ring_shadow_ratio of ring is "shadow" → dark scene, skip.
    6. Morphological cleanup.
    """
    H, W = img_bgr.shape[:2]
    mask_bool = mask_u8 > 0

    # ── 1. Safe background V stats ────────────────────────────────
    total_safe_px = search_px + safe_margin_px
    safe_k     = np.ones((total_safe_px * 2 + 1, total_safe_px * 2 + 1), np.uint8)
    far_zone   = cv2.dilate(mask_u8, safe_k, iterations=1) > 0
    safe_zone  = ~far_zone

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = img_hsv[:, :, 2].astype(np.float32)

    if safe_zone.any():
        bg_V_vals = V[safe_zone]
    else:
        bg_V_vals = V[~mask_bool]          # fallback if mask covers most of image

    if len(bg_V_vals) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    bg_V_med = float(np.median(bg_V_vals))
    bg_V_mad = max(float(np.median(np.abs(bg_V_vals - bg_V_med))), 3.0)
    threshold = bg_V_med - thresh_k * bg_V_mad

    # ── 2. Search ring ────────────────────────────────────────────
    search_k    = np.ones((search_px * 2 + 1, search_px * 2 + 1), np.uint8)
    outer_zone  = cv2.dilate(mask_u8, search_k, iterations=1) > 0
    search_ring = outer_zone & ~mask_bool

    ring_area = int(search_ring.sum())
    if ring_area == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # ── 3. Shadow candidates ──────────────────────────────────────
    shadow_cand = search_ring & (V < threshold)
    n_cand = int(shadow_cand.sum())
    if n_cand == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # ── 4. Sanity cap ─────────────────────────────────────────────
    if (n_cand / ring_area) > max_ring_shadow_ratio:
        # Too much of the ring is "dark" → probably just a dark-textured scene
        return np.zeros((H, W), dtype=np.uint8)

    # ── 5. Connectivity filter ────────────────────────────────────
    touch_k      = np.ones((connectivity_touch_px * 2 + 1,
                             connectivity_touch_px * 2 + 1), np.uint8)
    mask_exp     = cv2.dilate(mask_u8, touch_k, iterations=1) > 0

    shadow_u8    = shadow_cand.astype(np.uint8) * 255
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(shadow_u8, connectivity=8)

    shadow_mask = np.zeros((H, W), dtype=np.uint8)
    for lbl in range(1, n_lbl):
        if stats[lbl, cv2.CC_STAT_AREA] < min_shadow_blob_px:
            continue
        blob = (lbls == lbl)
        if (blob & mask_exp).any():           # touches mask → it's a cast shadow
            shadow_mask[blob] = 255

    if not shadow_mask.any():
        return np.zeros((H, W), dtype=np.uint8)

    # ── 6. Morphological cleanup ──────────────────────────────────
    close_k     = np.ones((9, 9), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, close_k)
    grow_k      = np.ones((3, 3), np.uint8)
    shadow_mask = cv2.dilate(shadow_mask, grow_k, iterations=1)

    return shadow_mask


# ============================================================================
# Public API
# ============================================================================
def compute_shadow_masks(
    views: list,
    search_px: int = 70,
    safe_margin_px: int = 20,
    thresh_k: float = 1.5,
    min_shadow_blob_px: int = 150,
    verbose: bool = True,
) -> None:
    """
    Method 1: Detect cast shadows per source view, merge into views[i]["mask_dilated"].

    After this call, pull harvest's source-side check
        is_bg_in_src = ~src_view["mask_dilated"][v, u]
    will automatically exclude shadow pixels from being pulled into target view.

    Modifies views in-place (mask_dilated field only).

    Parameters
    ----------
    views             : list of view dicts from _load_all_views
    search_px         : Shadow search range beyond mask edge (px).
                        Default 70. Use 100-150 for very long shadows.
    safe_margin_px    : Extra margin beyond search_px for clean background sampling.
    thresh_k          : Shadow brightness threshold factor.
                        1.5 = moderate. 1.0 = aggressive. 2.0 = conservative.
    min_shadow_blob_px: Minimum blob area to keep (noise rejection).
    verbose           : Print summary statistics.
    """
    total_shadow  = 0
    n_shadow_views = 0

    for v in views:
        sm = _detect_shadow_mask_for_view(
            img_bgr=v["img"],
            mask_u8=v["mask"],
            search_px=search_px,
            safe_margin_px=safe_margin_px,
            thresh_k=thresh_k,
            min_shadow_blob_px=min_shadow_blob_px,
        )
        n_sm = int((sm > 0).sum())
        if n_sm > 0:
            v["mask_dilated"] = np.maximum(v["mask_dilated"], sm)
            total_shadow  += n_sm
            n_shadow_views += 1

    if verbose:
        avg = total_shadow // max(n_shadow_views, 1)
        print(f"    🌑 [ShadowDet M1] {n_shadow_views}/{len(views)} views with shadow | "
              f"avg {avg:,} px/view | search={search_px}px thresh_k={thresh_k}")


def brightness_untrust_filter(
    canvas: np.ndarray,
    v_t: np.ndarray,
    u_t: np.ndarray,
    filled: np.ndarray,
    target_img: np.ndarray,
    target_mask: np.ndarray,
    ring_px: int = 20,
    bright_untrust_k: float = 2.5,
) -> tuple:
    """
    Method 2: Untrust pulled pixels that are significantly darker than background.

    Called on the TARGET side after pull harvest + Strategy B.
    Catches shadow pixels that slipped through Method 1 (e.g., unusual depth
    consistency, different SAM mask boundary in source vs target).

    Shadow pixels have lower HSV Value (V) than the clean background.
    This is true regardless of color, texture, or shadow direction.

    Parameters
    ----------
    canvas          : (H, W, 3) BGR uint8, canvas after pull harvest
    v_t, u_t        : (n_holes,) mask pixel row/col coordinates
    filled          : (n_holes,) bool, True = trusted (pulled)
    target_img      : (H, W, 3) BGR uint8, original target image
    target_mask     : (H, W) bool, True = mask area
    ring_px         : Outer ring distance for background V sampling
    bright_untrust_k: Shadow detection threshold factor.
                      2.5 = conservative. 2.0 = moderate. 1.5 = aggressive.

    Returns
    -------
    filled (updated bool), n_newly_untrusted (int)
    """
    filled_idx = np.where(filled)[0]
    if len(filled_idx) == 0:
        return filled, 0

    # ── Background V from target image outer ring ─────────────────
    mask_u8  = target_mask.astype(np.uint8) * 255
    # 取距離 mask 夠遠的「安全區」當 bg 參考（避免 shadow 污染 ring）
    # safe_px 必須大於陰影可能延伸的最大距離
    safe_px  = max(ring_px * 4, 80)   # 預設 80px，足夠遠離 shadow
    safe_k   = np.ones((safe_px * 2 + 1, safe_px * 2 + 1), np.uint8)
    # 取 > safe_px 之外的像素（NOT 在 dilated 區內 → 確實是遠端）
    outer    = ~(cv2.dilate(mask_u8, safe_k) > 0)
    if not outer.any():
        outer = ~target_mask   # fallback
    if not outer.any():
        return filled, 0

    tgt_hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)
    V_bg    = tgt_hsv[:, :, 2].astype(np.float32)[outer]
    bg_V_med = float(np.median(V_bg))
    bg_V_mad = max(float(np.median(np.abs(V_bg - bg_V_med))), 3.0)
    threshold = bg_V_med - bright_untrust_k * bg_V_mad

    # ── V of pulled pixels in canvas ─────────────────────────────
    cv_hsv   = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    V_canvas = cv_hsv[:, :, 2].astype(np.float32)
    pulled_V = V_canvas[v_t[filled_idx], u_t[filled_idx]]

    too_dark   = pulled_V < threshold
    n_too_dark = int(too_dark.sum())

    if n_too_dark > 0:
        dark_idx   = filled_idx[too_dark]
        new_filled = filled.copy()
        new_filled[dark_idx] = False
        pct = 100 * n_too_dark / max(len(filled_idx), 1)
        print(f"      🌑 [ShadowDet M2] {n_too_dark:,} dark px untrusted "
              f"({pct:.1f}%) | bg_V={bg_V_med:.0f}±{bg_V_mad:.0f} "
              f"thresh={threshold:.0f}")
        return new_filled, n_too_dark

    print(f"      🌑 [ShadowDet M2] 0 shadow (bg_V={bg_V_med:.0f} "
          f"thresh={threshold:.0f})")
    return filled, 0