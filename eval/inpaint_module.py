"""
generative_inpaint_module_360.py  ── Phase 1 + Strat B v4 + Shadow Detection v5
================================================================================
v5 新增（解決陰影造成黑色髒塊）:

  [Shadow M1] Source-side shadow exclusion
      陰影根因：陰影落在 SAM mask 外，屬於「可見的背景像素」
                pull harvest 的三重 filter（FOV + dilated mask + depth）全部通過
                → 陰影被當成真實背景 pull 進 canvas → LaMa 看到暗色 context → 生黑色
      M1 修法：對每個 source view 偵測 cast shadow 區域（不假設光源方向）
              把 shadow_mask 合進 mask_dilated → pull harvest 自動排除

  [Shadow M2] Target-side brightness untrust
      即使 M1 成功，仍可能有陰影 slip through（depth test 剛好過、連通性判斷失效）
      M2 修法：pull 完後，對每個 trusted pixel 檢查亮度（HSV V channel）
              明顯偏暗 → 強制 untrust → 進 dead_mask → LaMa 填補

  模組：eval/shadow_detector.py（需先放到對應路徑）
        若找不到該 module，graceful fallback：只 print 警告，其他功能不受影響

可調參數 (ref_cache):
  ── 原有 ──
  _src_dilation_px       = 11
  _tgt_dilation_px       = 5
  _use_poisson           = False
  _phot_z_thresh         = 3.0
  _phot_ring_px          = 20
  _local_bg_radius       = 20
  _local_bg_extra_dil    = 30
  _min_trusted_blob      = 1000
  _bilateral_d           = 15     (0 = 關閉)
  _bilateral_sigma       = 30.0
  _inpainter             = build_inpainter("lama")

  ── v5 新增（Shadow Detection）──
  _shadow_search_px      = 70     # M1: mask 外搜尋陰影的距離 (px)
  _shadow_thresh_k       = 1.5    # M1: 陰影亮度門檻 (越低越激進)
  _min_shadow_blob       = 150    # M1: 最小陰影 blob 存活大小 (px)
  _bright_untrust_k      = 2.5    # M2: target-side 亮度判斷門檻

  # Debug:
  _debug_dump_dir        = "debug_dump"
  _debug_target_indices  = [274]
"""
import os
import cv2
import numpy as np
from pathlib import Path

try:
    from eval.dead_zone_inpainter import CV2Inpainter
    _INPAINTER_MODULE_AVAILABLE = True
except ImportError:
    _INPAINTER_MODULE_AVAILABLE = False

# v5: optional shadow detector (graceful fallback if not installed)
try:
    from eval.shadow_detector import compute_shadow_masks, brightness_untrust_filter
    _SHADOW_DETECTOR_AVAILABLE = True
except ImportError:
    _SHADOW_DETECTOR_AVAILABLE = False
    print("[Inpaint-360] ⚠️  eval/shadow_detector.py not found — shadow detection disabled")


# ============================================================================
# Helpers (unchanged from v4)
# ============================================================================
def _resolve_mask_path(idx, image_paths, mask_paths, mask_dir):
    if mask_paths is not None:
        return str(mask_paths[idx])
    return os.path.join(str(mask_dir), os.path.basename(image_paths[idx]))


def _load_all_views(image_paths, mask_paths, mask_dir, raw_depth_maps, intrinsics,
                    src_mask_dilation_px: int = 11):
    src_kernel = (
        np.ones((src_mask_dilation_px, src_mask_dilation_px), np.uint8)
        if src_mask_dilation_px > 0 else None
    )
    views = []
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"❌ 讀不到 image: {img_path}")
        H, W = img.shape[:2]
        mask_path = _resolve_mask_path(idx, image_paths, mask_paths, mask_dir)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"❌ 讀不到 mask: {mask_path}")
        mask = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        mask_dilated = (
            cv2.dilate(mask, src_kernel, iterations=1)
            if src_kernel is not None else mask.copy()
        )
        depth_low = raw_depth_maps[idx]
        depth = cv2.resize(depth_low.astype(np.float32), (W, H),
                           interpolation=cv2.INTER_NEAREST)
        scale_x = W / depth_low.shape[1]
        scale_y = H / depth_low.shape[0]
        K = intrinsics[idx].copy().astype(np.float64)
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        views.append({
            "img": img, "mask": mask, "mask_dilated": mask_dilated,
            "depth": depth, "K": K, "H": H, "W": W, "idx": idx,
        })
    return views


def _compute_view_directions(views, all_cam_to_world_mat):
    N = len(views)
    cam_centers = np.zeros((N, 3), dtype=np.float64)
    for v in views:
        c2w = np.linalg.inv(all_cam_to_world_mat[v["idx"]])
        cam_centers[v["idx"]] = c2w[:3, 3]
    scene_center = cam_centers.mean(axis=0)
    view_dirs = scene_center - cam_centers
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9)
    return view_dirs


def _scene_depth_range(views):
    all_d = np.concatenate([v["depth"][v["depth"] > 0].ravel() for v in views])
    return float(all_d.max() - all_d.min())


def _get_default_inpainter():
    if _INPAINTER_MODULE_AVAILABLE:
        return CV2Inpainter()
    class _BareCV2:
        name = "cv2-inline"
        def inpaint(self, canvas, dead_mask, context=None):
            if not dead_mask.any():
                return canvas
            dead_u8 = dead_mask.astype(np.uint8) * 255
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(dead_u8, kernel, iterations=1)
            return cv2.inpaint(canvas, dilated, 10, cv2.INPAINT_TELEA)
    return _BareCV2()


def _poisson_blend(canvas, target_img, blend_mask):
    if not blend_mask.any():
        return canvas
    H, W = canvas.shape[:2]
    mask_u8 = blend_mask.astype(np.uint8) * 255
    safe_mask = np.zeros_like(mask_u8)
    safe_mask[1:-1, 1:-1] = mask_u8[1:-1, 1:-1]
    if not safe_mask.any():
        return canvas
    yc, xc = np.where(safe_mask > 0)
    cx, cy = int(np.mean(xc)), int(np.mean(yc))
    margin = 3
    if not (margin <= cx < W - margin and margin <= cy < H - margin):
        return canvas
    try:
        return cv2.seamlessClone(canvas, target_img, safe_mask,
                                 (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        return canvas


# ============================================================================
# Strategy B helpers (unchanged from v4)
# ============================================================================
def _build_local_bg_estimate(target_img, target_mask_bool,
                              extra_dilation_px: int = 30,
                              radius: int = 20):
    mask_u8 = target_mask_bool.astype(np.uint8) * 255
    if extra_dilation_px > 0:
        kernel = np.ones((extra_dilation_px, extra_dilation_px), np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    local_bg = cv2.inpaint(target_img, mask_u8, radius, cv2.INPAINT_TELEA)
    return local_bg.astype(np.float32)


def _build_bg_mad(target_img, target_mask_bool, ring_px: int = 20):
    mask_u8  = target_mask_bool.astype(np.uint8) * 255
    ring_k   = np.ones((ring_px, ring_px), np.uint8)
    outer_u8 = cv2.dilate(mask_u8, ring_k)
    outer_ring = (outer_u8 > 0) & (~target_mask_bool)
    if not outer_ring.any():
        outer_ring = ~target_mask_bool
    if not outer_ring.any():
        return np.array([20., 20., 20.], dtype=np.float32)
    bg_pixels = target_img[outer_ring].astype(np.float32)
    bg_median = np.median(bg_pixels, axis=0)
    bg_mad    = np.median(np.abs(bg_pixels - bg_median), axis=0)
    return np.maximum(bg_mad, 2.0)


def _photometric_validate(canvas, v_t, u_t, filled,
                           local_bg_est, bg_mad,
                           z_thresh: float = 3.0,
                           debug_print: bool = False):
    """Strategy B: detect contaminated pulled pixels (no canvas change)."""
    filled_idx = np.where(filled)[0]
    if len(filled_idx) == 0:
        return filled, 0
    pulled_v = v_t[filled_idx]
    pulled_u = u_t[filled_idx]
    pulled_colors = canvas[pulled_v, pulled_u].astype(np.float32)
    local_ref = local_bg_est[pulled_v, pulled_u]
    z_scores = np.abs(pulled_colors - local_ref) / (1.4826 * bg_mad)
    worst_z  = z_scores.max(axis=1)

    if debug_print:
        pct = np.percentile(worst_z, [50, 75, 90, 95, 99])
        print(f"      📊 z-score | 50%={pct[0]:.2f} 75%={pct[1]:.2f} "
              f"90%={pct[2]:.2f} 95%={pct[3]:.2f} 99%={pct[4]:.2f} | "
              f"z_thresh={z_thresh}")

    contaminated_local = worst_z > z_thresh
    n_contaminated = int(contaminated_local.sum())
    if n_contaminated > 0:
        contaminated_idx = filled_idx[contaminated_local]
        new_filled = filled.copy()
        new_filled[contaminated_idx] = False
        pct = 100 * n_contaminated / max(len(filled_idx), 1)
        print(f"      🔍 [Strat-B] mark {n_contaminated:,} untrusted "
              f"({pct:.1f}% of pulled) | bg_mad={bg_mad.astype(int)} z={z_thresh}")
        return new_filled, n_contaminated
    print(f"      🔍 [Strat-B] 0 contaminated (all pass z<{z_thresh:.1f})")
    return filled, 0


def _spatial_coherence_filter(filled, filled_2d, v_t, u_t, target_mask,
                               min_blob: int = 1000):
    """Fix A: Remove isolated trusted pixel clusters (parallax artifacts)."""
    trusted_in_mask = (filled_2d & target_mask).astype(np.uint8) * 255
    if not trusted_in_mask.any():
        return filled, filled_2d, 0
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        trusted_in_mask, connectivity=8
    )
    trusted_clean = np.zeros((filled_2d.shape[0], filled_2d.shape[1]), dtype=bool)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_blob:
            trusted_clean[labels == lbl] = True
    isolated_2d = (filled_2d & target_mask) & ~trusted_clean
    isolated_1d = isolated_2d[v_t, u_t]
    n_removed = int(isolated_1d.sum())
    if n_removed > 0:
        new_filled = filled & ~isolated_1d
        new_filled_2d = filled_2d.copy()
        new_filled_2d[v_t, u_t] = new_filled
        print(f"      🏝️  [Fix A] removed {n_removed:,} isolated trusted patches "
              f"(CC < {min_blob}px)")
        return new_filled, new_filled_2d, n_removed
    print(f"      🏝️  [Fix A] no isolated patches")
    return filled, filled_2d, 0


def _bilateral_smooth(canvas, target_mask, d: int = 15, sigma_color: float = 30.0):
    """Fix D: Edge-preserving bilateral smoothing within target_mask."""
    if d <= 0:
        return canvas
    smoothed = cv2.bilateralFilter(canvas, d=d, sigmaColor=sigma_color, sigmaSpace=d)
    result = canvas.copy()
    result[target_mask] = smoothed[target_mask]
    return result


# ============================================================================
# Debug helper (unchanged)
# ============================================================================
def _debug_save(dump_dir, target_idx, stage_name, image_or_mask, is_mask=False):
    if dump_dir is None:
        return
    out_dir = Path(dump_dir) / f"V_{target_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stage_name}.png"
    if is_mask:
        img = (image_or_mask.astype(np.uint8) * 255) if image_or_mask.dtype == bool else image_or_mask
    else:
        img = np.clip(image_or_mask, 0, 255).astype(np.uint8) if image_or_mask.dtype != np.uint8 else image_or_mask
    cv2.imwrite(str(out_path), img)


# ============================================================================
# 核心
# ============================================================================
def _harvest_target_view(target_idx, views, all_cam_to_world_mat,
                         view_dirs, scene_d_range,
                         depth_eps_factor=0.05,
                         inpainter=None,
                         tgt_dilation_px: int = 5,
                         use_poisson: bool = False,
                         phot_z_thresh: float = 3.0,
                         phot_ring_px: int = 20,
                         local_bg_radius: int = 20,
                         local_bg_extra_dil: int = 30,
                         min_trusted_blob: int = 1000,
                         bilateral_d: int = 15,
                         bilateral_sigma: float = 30.0,
                         bright_untrust_k: float = 2.5,   # v5 新增
                         debug_dump_dir=None,
                         debug_this_target: bool = False):

    target       = views[target_idx]
    H, W         = target["H"], target["W"]
    K_t          = target["K"]
    w2c_t        = all_cam_to_world_mat[target_idx]
    c2w_t        = np.linalg.inv(w2c_t)
    target_img   = target["img"]
    target_depth = target["depth"]
    epsilon      = scene_d_range * depth_eps_factor

    if debug_this_target:
        _debug_save(debug_dump_dir, target_idx, "01_target_img", target_img)

    # ── Fix 2: target mask dilation ──────────────────────────────
    target_mask_orig = target["mask"] > 0
    if tgt_dilation_px > 0:
        tgt_k  = np.ones((tgt_dilation_px, tgt_dilation_px), np.uint8)
        tgt_u8 = target_mask_orig.astype(np.uint8) * 255
        tgt_u8 = cv2.dilate(tgt_u8, tgt_k, iterations=1)
        target_mask = tgt_u8 > 0
    else:
        target_mask = target_mask_orig

    if debug_this_target:
        _debug_save(debug_dump_dir, target_idx, "02_target_mask_used",
                    target_mask, is_mask=True)

    canvas   = target_img.copy()
    v_t, u_t = np.where(target_mask)
    n_holes  = len(v_t)
    if n_holes == 0:
        return canvas, np.zeros((H, W), dtype=bool)

    # ── Step 1: unproject ─────────────────────────────────────────
    z_t = target_depth[v_t, u_t]
    valid_z = (z_t > 0) & np.isfinite(z_t)
    if not valid_z.all():
        median_z = np.median(z_t[valid_z]) if valid_z.any() else 1.0
        z_t = np.where(valid_z, z_t, median_z)
    x = (u_t - K_t[0, 2]) * z_t / K_t[0, 0]
    y = (v_t - K_t[1, 2]) * z_t / K_t[1, 1]
    pts_cam_t = np.stack([x, y, z_t], axis=1)
    pts_h     = np.hstack([pts_cam_t, np.ones((n_holes, 1))])
    pts_world = (c2w_t @ pts_h.T).T[:, :3]

    # ── Step 2: sort source views ─────────────────────────────────
    target_dir = view_dirs[target_idx]
    sims = view_dirs @ target_dir
    sims[target_idx] = -2.0
    source_order = np.argsort(-sims)

    # ── Step 3: pull harvest ──────────────────────────────────────
    filled      = np.zeros(n_holes, dtype=bool)
    source_used = np.full(n_holes, -1, dtype=np.int32)
    pts_world_h = np.hstack([pts_world, np.ones((n_holes, 1))])
    n_sources_tried = 0

    for src_idx in source_order:
        if filled.all():
            break
        unfilled_idx = np.where(~filled)[0]
        if len(unfilled_idx) == 0:
            break
        src_view  = views[src_idx]
        src_K     = src_view["K"]
        src_w2c   = all_cam_to_world_mat[src_idx]
        src_img   = src_view["img"]
        src_depth = src_view["depth"]
        H_s, W_s  = src_view["H"], src_view["W"]
        # v5: mask_dilated now includes shadow regions (from compute_shadow_masks)
        src_mask_check = src_view["mask_dilated"] > 0

        pts_check   = pts_world_h[unfilled_idx]
        pts_src_cam = (src_w2c @ pts_check.T).T[:, :3]
        z_src       = pts_src_cam[:, 2]
        valid_z_src = z_src > 0.1
        z_safe      = np.where(valid_z_src, z_src, 1.0)
        u_src = src_K[0, 0] * pts_src_cam[:, 0] / z_safe + src_K[0, 2]
        v_src = src_K[1, 1] * pts_src_cam[:, 1] / z_safe + src_K[1, 2]
        u_si  = np.where(np.isfinite(u_src), np.round(u_src), -1).astype(np.int32)
        v_si  = np.where(np.isfinite(v_src), np.round(v_src), -1).astype(np.int32)
        in_bounds = (valid_z_src & (u_si >= 0) & (u_si < W_s) & (v_si >= 0) & (v_si < H_s))
        if not in_bounds.any():
            n_sources_tried += 1
            continue
        u_safe_arr = np.clip(u_si, 0, W_s - 1)
        v_safe_arr = np.clip(v_si, 0, H_s - 1)
        is_bg_in_src = ~src_mask_check[v_safe_arr, u_safe_arr]
        src_d_at = src_depth[v_safe_arr, u_safe_arr]
        valid_d  = (src_d_at > 0) & np.isfinite(src_d_at)
        z_diff   = np.abs(z_src - src_d_at)
        depth_ok = valid_d & (z_diff < epsilon)
        valid_pull = in_bounds & is_bg_in_src & depth_ok
        if not valid_pull.any():
            n_sources_tried += 1
            continue
        local_pull_idx = unfilled_idx[valid_pull]
        canvas[v_t[local_pull_idx], u_t[local_pull_idx]] = (
            src_img[v_safe_arr[valid_pull], u_safe_arr[valid_pull]]
        )
        filled[local_pull_idx]      = True
        source_used[local_pull_idx] = src_idx
        n_sources_tried += 1

    n_pulled_raw = int(filled.sum())

    if debug_this_target:
        _debug_save(debug_dump_dir, target_idx, "04_canvas_after_pull", canvas)

    # ── Step 3.5: local_bg_est + Strategy B ──────────────────────
    local_bg_est = _build_local_bg_estimate(target_img, target_mask,
                                             extra_dilation_px=local_bg_extra_dil,
                                             radius=local_bg_radius)
    bg_mad = _build_bg_mad(target_img, target_mask, ring_px=phot_ring_px)

    if debug_this_target:
        _debug_save(debug_dump_dir, target_idx, "03_local_bg_est", local_bg_est)

    filled, n_contaminated = _photometric_validate(
        canvas, v_t, u_t, filled, local_bg_est, bg_mad,
        z_thresh=phot_z_thresh, debug_print=debug_this_target,
    )

    # ── Step 3.6: Shadow Method 2 — brightness untrust ───────────
    # Catches shadow pixels that slipped through Method 1 (source-side).
    # Runs BEFORE Fix A so that shadow-untrusted pixels are included in
    # dead_mask and handled by LaMa (not mistakenly kept as trusted).
    if _SHADOW_DETECTOR_AVAILABLE:
        filled, n_shadow_untrust = brightness_untrust_filter(
            canvas, v_t, u_t, filled,
            target_img, target_mask,
            bright_untrust_k=bright_untrust_k,
        )
    else:
        n_shadow_untrust = 0

    # ── Step 3.7: Fix A — Spatial Coherence Filter ────────────────
    filled_2d = np.zeros((H, W), dtype=bool)
    filled_2d[v_t, u_t] = filled

    filled, filled_2d, n_isolated = _spatial_coherence_filter(
        filled, filled_2d, v_t, u_t, target_mask, min_blob=min_trusted_blob
    )

    if debug_this_target:
        _debug_save(debug_dump_dir, target_idx, "05_filled_2d_after_fixA",
                    filled_2d & target_mask, is_mask=True)

    # ── Step 4: stats ─────────────────────────────────────────────
    n_filled = int(filled.sum())
    n_dead   = n_holes - n_filled
    pct_pull = 100 * n_filled / max(n_holes, 1)
    pct_dead = 100 * n_dead   / max(n_holes, 1)
    n_unique = len(np.unique(source_used[filled])) if n_filled > 0 else 0
    print(f"      📤 V_{target_idx:3d}: "
          f"raw={n_pulled_raw:,} → stratB={n_pulled_raw-n_contaminated:,} "
          f"→ shadow={n_pulled_raw-n_contaminated-n_shadow_untrust:,} "
          f"→ fixA={n_filled:,}/{n_holes:,} ({pct_pull:.1f}%) "
          f"from {n_unique} src | dead={n_dead:,} ({pct_dead:.1f}%)")

    # ── Step 5: dead_mask + morphology + CC ──────────────────────
    dead_mask = np.zeros((H, W), dtype=bool)
    dead_idx  = np.where(~filled)[0]
    if len(dead_idx) > 0:
        dead_mask[v_t[dead_idx], u_t[dead_idx]] = True

    if debug_this_target:
        _debug_save(debug_dump_dir, target_idx, "06_dead_mask_raw",
                    dead_mask, is_mask=True)

    if dead_mask.any():
        dead_u8 = dead_mask.astype(np.uint8) * 255
        close_kernel  = np.ones((11, 11), np.uint8)
        dead_u8       = cv2.morphologyEx(dead_u8, cv2.MORPH_CLOSE, close_kernel)
        dilate_kernel = np.ones((5, 5), np.uint8)
        dead_u8       = cv2.dilate(dead_u8, dilate_kernel, iterations=1)
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(dead_u8, connectivity=8)
        min_blob_px = 500
        clean_u8 = np.zeros_like(dead_u8)
        for lbl in range(1, n_lbl):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_px:
                clean_u8[lbls == lbl] = 255
        dead_mask_final = clean_u8 > 0

        if debug_this_target:
            _debug_save(debug_dump_dir, target_idx, "07_dead_mask_final",
                        dead_mask_final, is_mask=True)

        # Fix B: clean all untrusted pixels with local_bg_est
        untrusted_in_mask = target_mask & ~filled_2d
        n_untrusted = int(untrusted_in_mask.sum())
        if n_untrusted > 0:
            canvas[untrusted_in_mask] = np.clip(
                np.round(local_bg_est[untrusted_in_mask]), 0, 255
            ).astype(np.uint8)
            print(f"      🧹 [Fix B] cleaned {n_untrusted:,} untrusted pixels")

        # Fix D: bilateral smoothing
        canvas = _bilateral_smooth(canvas, target_mask,
                                   d=bilateral_d, sigma_color=bilateral_sigma)
        if bilateral_d > 0:
            print(f"      🌀 [Fix D] bilateral d={bilateral_d} σ={bilateral_sigma:.0f}")

        if debug_this_target:
            _debug_save(debug_dump_dir, target_idx, "08_canvas_before_lama", canvas)

        _ip = inpainter if inpainter is not None else _get_default_inpainter()
        context = {
            "target_idx":    target_idx,
            "depth":         target_depth,
            "scene_d_range": scene_d_range,
        }
        canvas = _ip.inpaint(canvas, dead_mask_final, context=context)
        if canvas.shape[:2] != (H, W):
            print(f"      ⚠️  Size mismatch {canvas.shape[:2]} → ({H},{W}), resizing")
            canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)

        if debug_this_target:
            _debug_save(debug_dump_dir, target_idx, "09_canvas_after_lama", canvas)
    else:
        dead_mask_final = dead_mask

    if use_poisson:
        canvas = _poisson_blend(canvas, target_img, target_mask)

    return canvas, dead_mask_final


# ============================================================================
# 主入口
# ============================================================================
def generative_multi_ref_propagation(
    ref_indices, target_idx, image_paths, mask_dir,
    raw_depth_maps, all_cam_to_world_mat, intrinsics,
    output_dir, ref_cache,
    mask_paths=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 讀取所有參數 ─────────────────────────────────────────────
    src_dilation_px    = ref_cache.get("_src_dilation_px", 11)
    tgt_dilation_px    = ref_cache.get("_tgt_dilation_px", 5)
    use_poisson        = ref_cache.get("_use_poisson", False)
    phot_z_thresh      = ref_cache.get("_phot_z_thresh", 3.0)
    phot_ring_px       = ref_cache.get("_phot_ring_px", 20)
    local_bg_radius    = ref_cache.get("_local_bg_radius", 20)
    local_bg_extra_dil = ref_cache.get("_local_bg_extra_dil", 30)
    min_trusted_blob   = ref_cache.get("_min_trusted_blob", 1000)
    bilateral_d        = ref_cache.get("_bilateral_d", 15)
    bilateral_sigma    = ref_cache.get("_bilateral_sigma", 30.0)
    # v5 shadow params
    shadow_search_px   = ref_cache.get("_shadow_search_px", 70)
    shadow_thresh_k    = ref_cache.get("_shadow_thresh_k", 1.5)
    min_shadow_blob    = ref_cache.get("_min_shadow_blob", 150)
    bright_untrust_k   = ref_cache.get("_bright_untrust_k", 2.5)

    debug_dump_dir       = ref_cache.get("_debug_dump_dir", None)
    debug_target_indices = ref_cache.get("_debug_target_indices", [])
    debug_this_target    = (debug_dump_dir is not None and
                            target_idx in debug_target_indices)

    # ── 第一次呼叫：載入 views + shadow detection ─────────────────
    if "_pullharvest_views" not in ref_cache:
        print("\n[Inpaint-360 v5] 首次呼叫: 載入 view 資料 ...")
        print(f"    src_dil={src_dilation_px}px | tgt_dil={tgt_dilation_px}px | "
              f"z_thresh={phot_z_thresh} | bg_extra_dil={local_bg_extra_dil}px")
        print(f"    min_trusted_blob={min_trusted_blob}px | "
              f"bilateral d={bilateral_d} σ={bilateral_sigma}")
        print(f"    shadow: search={shadow_search_px}px thresh_k={shadow_thresh_k} "
              f"min_blob={min_shadow_blob}px bright_k={bright_untrust_k}")

        views = _load_all_views(
            image_paths, mask_paths, mask_dir,
            raw_depth_maps, intrinsics,
            src_mask_dilation_px=src_dilation_px,
        )

        # ── v5: Shadow Method 1 (source-side) ────────────────────
        if _SHADOW_DETECTOR_AVAILABLE:
            compute_shadow_masks(
                views,
                search_px=shadow_search_px,
                thresh_k=shadow_thresh_k,
                min_shadow_blob_px=min_shadow_blob,
                verbose=True,
            )
        else:
            print("    ⚠️  shadow_detector unavailable — M1 skipped")

        view_dirs     = _compute_view_directions(views, all_cam_to_world_mat)
        scene_d_range = _scene_depth_range(views)
        ref_cache["_pullharvest_views"]     = views
        ref_cache["_pullharvest_view_dirs"] = view_dirs
        ref_cache["_pullharvest_d_range"]   = scene_d_range

        inpainter_obj = ref_cache.get("_inpainter")
        strategy = (getattr(inpainter_obj, "name", "cv2-fallback")
                    if inpainter_obj else "cv2-fallback")
        print(f"    ✅ {len(views)} views | dead-zone=[{strategy}]")

    views         = ref_cache["_pullharvest_views"]
    view_dirs     = ref_cache["_pullharvest_view_dirs"]
    scene_d_range = ref_cache["_pullharvest_d_range"]
    inpainter     = ref_cache.get("_inpainter", None)

    if debug_this_target:
        print(f"\n   ⭐ DEBUG V_{target_idx} ⭐")

    canvas, dead_mask = _harvest_target_view(
        target_idx, views, all_cam_to_world_mat,
        view_dirs, scene_d_range,
        inpainter=inpainter,
        tgt_dilation_px=tgt_dilation_px,
        use_poisson=use_poisson,
        phot_z_thresh=phot_z_thresh,
        phot_ring_px=phot_ring_px,
        local_bg_radius=local_bg_radius,
        local_bg_extra_dil=local_bg_extra_dil,
        min_trusted_blob=min_trusted_blob,
        bilateral_d=bilateral_d,
        bilateral_sigma=bilateral_sigma,
        bright_untrust_k=bright_untrust_k,
        debug_dump_dir=debug_dump_dir,
        debug_this_target=debug_this_target,
    )

    out_path = output_dir / f"inpainted_{target_idx}.png"
    cv2.imwrite(str(out_path), canvas)

    deadmask_dir = output_dir.parent / "deadmasks"
    deadmask_dir.mkdir(exist_ok=True)
    dead_path = deadmask_dir / f"deadmask_{target_idx}.png"
    cv2.imwrite(str(dead_path), dead_mask.astype(np.uint8) * 255)

    return int(dead_mask.sum()), 0.0