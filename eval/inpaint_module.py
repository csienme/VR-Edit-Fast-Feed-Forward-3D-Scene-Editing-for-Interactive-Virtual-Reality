"""
generative_inpaint_module_360.py  ── Phase 1 + Strategy B (2025-05-12)
=======================================================================
Backward Pull Harvesting Pipeline

Phase 1 修正 (前版):
  [Fix 1] Source-side mask dilation (src_dilation_px)
  [Fix 2] Target mask dilation before harvesting (tgt_dilation_px)
  [Fix 4] Poisson seamless blending (use_poisson)

Strategy B 新增 (本版):
  [Strat B] Photometric validation of pulled pixels
      ─ 三重幾何 filter (FOV + mask + depth) 是幾何檢查，無法擋住
        「在幾何上合法但顏色污染」的 source pixel：
          物體投在背景上的陰影、SAM mask 切太緊的 anti-aliased 邊緣、
          物體顏色 bleeding
      ─ 解法：pull 完之後，對每個成功 pull 進來的 pixel
        用 MAD-based robust z-score 跟 local background model 比較
        z-score > threshold → 視為污染 → 退回 dead_mask 交給 LaMa 重填
      ─ Background model 取自 target mask 外圈 (outer ring) 的真實背景像素
        用 median + MAD 而不是 mean + std → 對個別 outlier 不敏感
      ─ 退回的 contaminated pixels 在 canvas 上還原成 bg_median
        確保 LaMa 看到的 context border 是乾淨的背景色
      ─ Strategy A 副作用：dead_mask 自然變大 + solid
        close_kernel 同步升級 7→11 配合 B 產生的較大 dead zone

可調參數 (在 eval_custom_360.py for-loop 之前注入 ref_cache):
  ref_cache["_src_dilation_px"] = 11    # Fix 1: source mask 膨脹半徑
  ref_cache["_tgt_dilation_px"] = 5     # Fix 2: target mask 膨脹半徑
  ref_cache["_use_poisson"]     = False # Fix 4: Poisson blend（forward-facing 建議關）
  ref_cache["_phot_z_thresh"]   = 3.0  # Strat B: 污染判斷 robust z-score 門檻
  ref_cache["_phot_ring_px"]    = 20   # Strat B: background model 取樣圈寬度 (px)
  ref_cache["_inpainter"]       = build_inpainter("lama")
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


# ============================================================================
# Helpers
# ============================================================================
def _resolve_mask_path(idx, image_paths, mask_paths, mask_dir):
    if mask_paths is not None:
        return str(mask_paths[idx])
    return os.path.join(str(mask_dir), os.path.basename(image_paths[idx]))


def _load_all_views(image_paths, mask_paths, mask_dir, raw_depth_maps, intrinsics,
                    src_mask_dilation_px: int = 11):
    """
    載入所有 view 的 img/mask/depth/K，並預計算 mask_dilated (Fix 1 用)。
    """
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
            "img":          img,
            "mask":         mask,
            "mask_dilated": mask_dilated,
            "depth":        depth,
            "K":            K,
            "H":            H,
            "W":            W,
            "idx":          idx,
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
    """Fix 4: Poisson seamless clone."""
    if not blend_mask.any():
        return canvas

    H, W = canvas.shape[:2]
    mask_u8 = blend_mask.astype(np.uint8) * 255
    safe_mask = np.zeros_like(mask_u8)
    safe_mask[1:-1, 1:-1] = mask_u8[1:-1, 1:-1]
    if not safe_mask.any():
        print("      ⚠️  Poisson blend: mask 貼到圖邊，跳過")
        return canvas

    yc, xc = np.where(safe_mask > 0)
    cx, cy = int(np.mean(xc)), int(np.mean(yc))
    margin = 3
    if not (margin <= cx < W - margin and margin <= cy < H - margin):
        print("      ⚠️  Poisson blend: center 太靠近圖邊，跳過")
        return canvas

    try:
        result = cv2.seamlessClone(canvas, target_img, safe_mask,
                                   (cx, cy), cv2.NORMAL_CLONE)
        return result
    except cv2.error as e:
        print(f"      ⚠️  Poisson blend failed: {e}，跳過")
        return canvas


# ============================================================================
# Strategy B helpers
# ============================================================================
def _build_bg_model(target_img, target_mask_bool, ring_px: int = 20):
    """
    建立 local background color model (BGR float32).

    取樣範圍：target_mask 往外膨脹 ring_px 後的外圈（mask 本身不包含）
    統計量：median + MAD（robust，對個別 outlier 不敏感）

    Returns
    -------
    bg_median : (3,) float32  BGR channel median
    bg_mad    : (3,) float32  Median Absolute Deviation（floor = 2.0）
    """
    mask_u8  = target_mask_bool.astype(np.uint8) * 255
    ring_k   = np.ones((ring_px, ring_px), np.uint8)
    outer_u8 = cv2.dilate(mask_u8, ring_k)
    # outer ring = dilated area MINUS the mask itself
    outer_ring = (outer_u8 > 0) & (~target_mask_bool)

    if not outer_ring.any():
        outer_ring = ~target_mask_bool          # fallback: all non-mask pixels
    if not outer_ring.any():
        return (np.array([127., 127., 127.], dtype=np.float32),
                np.array([20.,  20.,  20.],  dtype=np.float32))

    bg_pixels = target_img[outer_ring].astype(np.float32)   # (N, 3)
    bg_median = np.median(bg_pixels, axis=0)                 # (3,)
    bg_mad    = np.median(np.abs(bg_pixels - bg_median), axis=0)
    bg_mad    = np.maximum(bg_mad, 2.0)   # floor: prevent over-sensitivity
    return bg_median, bg_mad


def _photometric_validate(canvas, v_t, u_t, filled,
                           bg_median, bg_mad,
                           z_thresh: float = 3.0):
    """
    Strategy B: 對 pull harvest 成功的 pixel 做 photometric 驗證。

    Per-pixel robust z-score (MAD-based):
      z = max_channel( |pulled_BGR - bg_median| / (1.4826 * bg_mad) )
    z > z_thresh → 視為污染（陰影 / object bleeding / anti-aliased edge）
                 → un-fill → 加入 dead_mask → 由 LaMa/CV2 重填

    污染 pixel 在 canvas 上還原為 bg_median（uint8），
    確保 inpainter 的 context border 是乾淨的背景色。

    Parameters
    ----------
    canvas       : (H, W, 3) BGR uint8，已含 pulled pixels（in-place 修改）
    v_t, u_t     : (n_holes,) int，mask 內所有 pixel 座標
    filled       : (n_holes,) bool，pull harvest 後的填充狀態
    bg_median    : (3,) float32，來自 _build_bg_model()
    bg_mad       : (3,) float32，來自 _build_bg_model()
    z_thresh     : robust z-score 門檻

    Returns
    -------
    filled        : (n_holes,) bool（更新後，contaminated → False）
    n_contaminated: int
    """
    filled_idx = np.where(filled)[0]
    if len(filled_idx) == 0:
        return filled, 0

    pulled_v      = v_t[filled_idx]
    pulled_u      = u_t[filled_idx]
    pulled_colors = canvas[pulled_v, pulled_u].astype(np.float32)   # (N, 3)

    # 1.4826 = consistency factor (Normal dist.) for MAD-based robust z-score
    z_scores = np.abs(pulled_colors - bg_median) / (1.4826 * bg_mad)  # (N, 3)
    worst_z  = z_scores.max(axis=1)                                    # (N,) worst channel

    contaminated_local = worst_z > z_thresh
    n_contaminated     = int(contaminated_local.sum())

    if n_contaminated > 0:
        contaminated_idx = filled_idx[contaminated_local]

        # Un-fill: 退回 dead_mask
        new_filled = filled.copy()
        new_filled[contaminated_idx] = False

        # Canvas 還原到 bg_median → LaMa context border 乾淨
        bg_u8 = np.clip(np.round(bg_median), 0, 255).astype(np.uint8)
        canvas[v_t[contaminated_idx], u_t[contaminated_idx]] = bg_u8

        pct = 100 * n_contaminated / max(len(filled_idx), 1)
        print(f"      🔍 [Strat-B] reject {n_contaminated:,} contaminated "
              f"({pct:.1f}% of pulled) | "
              f"bg_median(BGR)={bg_median.astype(int)} "
              f"bg_mad={bg_mad.astype(int)}")
        return new_filled, n_contaminated

    print(f"      🔍 [Strat-B] 0 contaminated (all pulled pixels pass z<{z_thresh:.1f})")
    return filled, 0


# ============================================================================
# 核心: 對單一 target view 做 backward pull harvesting
# ============================================================================
def _harvest_target_view(target_idx, views, all_cam_to_world_mat,
                         view_dirs, scene_d_range,
                         depth_eps_factor=0.05,
                         inpainter=None,
                         tgt_dilation_px: int = 5,
                         use_poisson: bool = False,
                         phot_z_thresh: float = 3.0,
                         phot_ring_px: int = 20):
    """
    [Fix 1] pull 時用 mask_dilated 做 source 背景檢查
    [Fix 2] 先膨脹 target mask，把 shadow halo 列入修補名單
    [Strat B] pull 完後 photometric 驗證，退回污染 pixel 到 dead_mask
    [Strat A] close_kernel 升級 7→11，solidify 污染退回後的 dead_mask
    [Fix 4] 填補完後 Poisson blend（forward-facing 建議關）
    """
    target       = views[target_idx]
    H, W         = target["H"], target["W"]
    K_t          = target["K"]
    w2c_t        = all_cam_to_world_mat[target_idx]
    c2w_t        = np.linalg.inv(w2c_t)
    target_img   = target["img"]
    target_depth = target["depth"]
    epsilon      = scene_d_range * depth_eps_factor

    # ── Fix 2: 膨脹 target mask ──────────────────────────────────
    target_mask_orig = target["mask"] > 0
    if tgt_dilation_px > 0:
        tgt_k  = np.ones((tgt_dilation_px, tgt_dilation_px), np.uint8)
        tgt_u8 = target_mask_orig.astype(np.uint8) * 255
        tgt_u8 = cv2.dilate(tgt_u8, tgt_k, iterations=1)
        target_mask = tgt_u8 > 0
    else:
        target_mask = target_mask_orig

    canvas   = target_img.copy()
    v_t, u_t = np.where(target_mask)
    n_holes  = len(v_t)
    if n_holes == 0:
        return canvas, np.zeros((H, W), dtype=bool)

    # ── Step 1: 反投影 mask interior 到 3D ───────────────────────
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

    # ── Step 2: 排序 source views by viewing angle proximity ─────
    target_dir   = view_dirs[target_idx]
    sims         = view_dirs @ target_dir
    sims[target_idx] = -2.0
    source_order = np.argsort(-sims)

    # ── Step 3: Pull harvest (vectorized) ────────────────────────
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

        # Fix 1: 用 mask_dilated 做 source 背景判斷
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

        # (a) FOV
        in_bounds = (valid_z_src &
                     (u_si >= 0) & (u_si < W_s) &
                     (v_si >= 0) & (v_si < H_s))
        if not in_bounds.any():
            n_sources_tried += 1
            continue
        u_safe_arr = np.clip(u_si, 0, W_s - 1)
        v_safe_arr = np.clip(v_si, 0, H_s - 1)

        # (b) Fix 1: dilated source mask 判斷
        is_bg_in_src = ~src_mask_check[v_safe_arr, u_safe_arr]

        # (c) Depth consistency
        src_d_at = src_depth[v_safe_arr, u_safe_arr]
        valid_d  = (src_d_at > 0) & np.isfinite(src_d_at)
        z_diff   = np.abs(z_src - src_d_at)
        depth_ok = valid_d & (z_diff < epsilon)

        valid_pull = in_bounds & is_bg_in_src & depth_ok
        if not valid_pull.any():
            n_sources_tried += 1
            continue

        local_pull_idx = unfilled_idx[valid_pull]
        u_pull = u_safe_arr[valid_pull]
        v_pull = v_safe_arr[valid_pull]
        canvas[v_t[local_pull_idx], u_t[local_pull_idx]] = src_img[v_pull, u_pull]
        filled[local_pull_idx]      = True
        source_used[local_pull_idx] = src_idx
        n_sources_tried += 1

    n_pulled_raw = int(filled.sum())

    # ── Step 3.5: Strategy B — Photometric Validation ────────────
    # Background model: outer ring の真實背景像素 (median + MAD)
    bg_median, bg_mad = _build_bg_model(target_img, target_mask,
                                         ring_px=phot_ring_px)
    # 驗證每個 pulled pixel，退回污染者
    filled, n_contaminated = _photometric_validate(
        canvas, v_t, u_t, filled,
        bg_median, bg_mad,
        z_thresh=phot_z_thresh,
    )

    # ── Step 4: 統計（Strat-B 後更新）────────────────────────────
    n_filled = int(filled.sum())
    n_dead   = n_holes - n_filled
    pct_pull = 100 * n_filled / max(n_holes, 1)
    pct_dead = 100 * n_dead   / max(n_holes, 1)
    n_unique = len(np.unique(source_used[filled])) if n_filled > 0 else 0
    print(f"      📤 V_{target_idx:3d}: "
          f"raw_pull={n_pulled_raw:,} → after_B={n_filled:,}/{n_holes:,} "
          f"({pct_pull:.1f}%) from {n_unique} src | "
          f"dead={n_dead:,} ({pct_dead:.1f}%) (tried {n_sources_tried} src)")

    # ── Step 5: 死角填補（Strategy A: close 升級 7→11）───────────
    dead_mask = np.zeros((H, W), dtype=bool)
    dead_idx  = np.where(~filled)[0]
    if len(dead_idx) > 0:
        dead_mask[v_t[dead_idx], u_t[dead_idx]] = True

    if dead_mask.any():
        dead_u8 = dead_mask.astype(np.uint8) * 255
        close_kernel  = np.ones((11, 11), np.uint8)
        dead_u8       = cv2.morphologyEx(dead_u8, cv2.MORPH_CLOSE, close_kernel)
        dilate_kernel = np.ones((5, 5), np.uint8)
        dead_u8       = cv2.dilate(dead_u8, dilate_kernel, iterations=1)
        dead_mask_final = dead_u8 > 0

        # ── Patch B: 移除細碎假陽性 blob ────────────────────────
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dead_u8, connectivity=8
        )
        min_blob_px = 500
        clean_u8 = np.zeros_like(dead_u8)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_px:
                clean_u8[labels == lbl] = 255
        dead_mask_final = clean_u8 > 0
        # ────────────────────────────────────────────────────────

        # ── Patch A: 清空 canvas 所有 dead 區到 bg_median ────────
        # LaMa 雖然理論上忽略 mask 內部，但 border 效應可能漏看物體色
        # 把 dead 區全部清成背景顏色，確保 context 絕對乾淨
        bg_u8_fill = np.clip(np.round(bg_median), 0, 255).astype(np.uint8)
        canvas[dead_mask_final] = bg_u8_fill
        # ────────────────────────────────────────────────────────

        _ip = inpainter if inpainter is not None else _get_default_inpainter()
        context = {
            "target_idx":    target_idx,
            "depth":         target_depth,
            "scene_d_range": scene_d_range,
        }
        canvas = _ip.inpaint(canvas, dead_mask_final, context=context)
        if canvas.shape[:2] != (H, W):
            print(f"      ⚠️  Inpainter size mismatch {canvas.shape[:2]} → ({H},{W}), resizing")
            canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        dead_mask_final = dead_mask

    # ── Step 6: Fix 4 — Poisson seamless blending ─────────────────
    if use_poisson:
        canvas = _poisson_blend(canvas, target_img, target_mask)

    return canvas, dead_mask_final


# ============================================================================
# 主入口: drop-in replacement
# ============================================================================
def generative_multi_ref_propagation(
    ref_indices, target_idx, image_paths, mask_dir,
    raw_depth_maps, all_cam_to_world_mat, intrinsics,
    output_dir, ref_cache,
    mask_paths=None,
):
    """
    Backward Pull Harvesting Inpaint — Phase 1 + Strategy B

    eval_custom_360.py for-loop 之前注入 ref_cache:
      ref_cache["_src_dilation_px"] = 11    # Fix 1
      ref_cache["_tgt_dilation_px"] = 5     # Fix 2（forward-facing 建議 3-5）
      ref_cache["_use_poisson"]     = False # Fix 4（forward-facing 建議 False）
      ref_cache["_phot_z_thresh"]   = 3.0  # Strat B 門檻（調低=更嚴格，最低約 2.0）
      ref_cache["_phot_ring_px"]    = 20   # Strat B background 取樣圈寬
      ref_cache["_inpainter"]       = build_inpainter("lama")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_dilation_px = ref_cache.get("_src_dilation_px", 11)
    tgt_dilation_px = ref_cache.get("_tgt_dilation_px", 5)
    use_poisson     = ref_cache.get("_use_poisson", False)
    phot_z_thresh   = ref_cache.get("_phot_z_thresh", 3.0)
    phot_ring_px    = ref_cache.get("_phot_ring_px", 20)

    # ── 第一次呼叫: 載入並 cache ──────────────────────────────────
    if "_pullharvest_views" not in ref_cache:
        print("\n[Inpaint-360 Pull-Harvest] 首次呼叫: 載入 view 資料 ...")
        print(f"    Params: src_dil={src_dilation_px}px | tgt_dil={tgt_dilation_px}px | "
              f"poisson={use_poisson} | phot_z={phot_z_thresh} | ring={phot_ring_px}px")
        views = _load_all_views(
            image_paths, mask_paths, mask_dir,
            raw_depth_maps, intrinsics,
            src_mask_dilation_px=src_dilation_px,
        )
        view_dirs     = _compute_view_directions(views, all_cam_to_world_mat)
        scene_d_range = _scene_depth_range(views)
        ref_cache["_pullharvest_views"]     = views
        ref_cache["_pullharvest_view_dirs"] = view_dirs
        ref_cache["_pullharvest_d_range"]   = scene_d_range

        inpainter_obj = ref_cache.get("_inpainter")
        strategy = (getattr(inpainter_obj, "name", "cv2-fallback")
                    if inpainter_obj else "cv2-fallback")
        print(f"    ✅ 載入 {len(views)} views | depth range = {scene_d_range:.3f} | "
              f"dead-zone: [{strategy}]")

    views         = ref_cache["_pullharvest_views"]
    view_dirs     = ref_cache["_pullharvest_view_dirs"]
    scene_d_range = ref_cache["_pullharvest_d_range"]
    inpainter     = ref_cache.get("_inpainter", None)

    # ── Pull harvest + Strategy B ────────────────────────────────
    canvas, dead_mask = _harvest_target_view(
        target_idx, views, all_cam_to_world_mat,
        view_dirs, scene_d_range,
        inpainter=inpainter,
        tgt_dilation_px=tgt_dilation_px,
        use_poisson=use_poisson,
        phot_z_thresh=phot_z_thresh,
        phot_ring_px=phot_ring_px,
    )

    # ── 寫檔 ──────────────────────────────────────────────────────
    out_path = output_dir / f"inpainted_{target_idx}.png"
    cv2.imwrite(str(out_path), canvas)

    deadmask_dir = output_dir.parent / "deadmasks"
    deadmask_dir.mkdir(exist_ok=True)
    dead_path = deadmask_dir / f"deadmask_{target_idx}.png"
    cv2.imwrite(str(dead_path), dead_mask.astype(np.uint8) * 255)

    return int(dead_mask.sum()), 0.0