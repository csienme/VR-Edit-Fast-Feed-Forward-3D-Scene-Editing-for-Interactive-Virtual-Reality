"""
generative_inpaint_module_360.py  ── Phase 1 Patch (2025-05-11)
================================================================
Backward Pull Harvesting Pipeline

Phase 1 三項修正:
  [Fix 1] Source-side mask dilation
      ─ 在 source 端做 is_bg_in_src 判斷時，改用 mask_dilated（預膨脹 src_dilation_px px）
      ─ 目的：排除 SAM 切太緊殘留的 anti-aliased 邊緣像素 + 物體投射到地板的陰影
      ─ 原理：陰影在幾何上完全通過三重 filter（depth OK, mask外），只有 photometric
               方法才抓得到；擴張 source mask 是最直接的 proxy。

  [Fix 2] Target mask dilation before harvesting
      ─ 在 unproject mask interior 之前，先把 target mask 膨脹 tgt_dilation_px px
      ─ 目的：把 shadow halo 視為「需要修補的洞」而不是「已有的背景」
      ─ 之前做的是 pull 完之後才擴 dead_mask → LaMa context 已污染
      ─ Fix 2 = 在污染發生之前就把 shadow halo 列入修補名單

  [Fix 4] Poisson seamless blending (收尾)
      ─ 全部填補完畢（pull + LaMa）之後做一次 cv2.seamlessClone
      ─ 目的：處理不同 source view 之間 lighting / white balance 微小差異導致的色塊感
      ─ 使用 NORMAL_CLONE：保留 pulled 內容的 gradient，讓亮度 match 周圍背景

可調參數 (在 eval_custom_360.py for-loop 之前注入 ref_cache):
  ref_cache["_src_dilation_px"] = 11   # Fix 1: source mask 膨脹半徑 (px)
  ref_cache["_tgt_dilation_px"] = 11   # Fix 2: target mask 膨脹半徑 (px)
  ref_cache["_use_poisson"]     = True # Fix 4: 是否做 Poisson blend
  ref_cache["_inpainter"]       = build_inpainter("lama")  # dead zone 策略

核心思想 (反轉之前的 forward push):
  對每個 target view 的破洞 pixel:
    1. 用 target 自己的 VGGT depth 反投影成 3D 世界座標
    2. 投到每個 source view, 三重檢查:
       (a) 在 FOV 內
       (b) source 該位置在 dilated mask 之外 (真實背景, 不含 shadow halo)  ← Fix 1
       (c) Depth 與 source VGGT depth 一致
    3. 從通過檢查的 source 中選 viewing direction 最接近的, 取真實 RGB
    4. 真死角 → pluggable DeadZoneInpainter
    5. Poisson seamless blend 做光度收尾  ← Fix 4
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
    src_mask_dilation_px: 膨脹半徑，0 = 不膨脹（退回原始行為）。
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

        # ── Fix 1: 預計算膨脹版 source mask ──────────────────────
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
            "mask_dilated": mask_dilated,   # ← Fix 1 新增
            "depth":        depth,
            "K":            K,
            "H":            H,
            "W":            W,
            "idx":          idx,
        })
    return views


def _compute_view_directions(views, all_cam_to_world_mat):
    """每個 view 的 viewing direction (camera_center → scene_center)."""
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
    """所有 view 的有效 depth 範圍，用來算 epsilon."""
    all_d = np.concatenate([v["depth"][v["depth"] > 0].ravel() for v in views])
    return float(all_d.max() - all_d.min())


def _get_default_inpainter():
    """預設 dead-zone inpainter (cv2，向下相容)."""
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
    """
    Fix 4: Poisson seamless clone.

    canvas:     已完整填補的畫布 (BGR uint8)
    target_img: 原始 target 圖 (BGR uint8, 物體還在)
    blend_mask: bool (H, W), True = 需要光度 harmonize 的區域

    seamlessClone 以 canvas 為 src、target_img 為 dst：
      - 只有 blend_mask 內部用 Poisson equation 求解
      - 邊界條件 = target_img 在 mask 外緣的像素 (純淨背景)
      - 結果: 填補區域的 gradient 保留 (紋理), 亮度自動 match 周圍背景
    """
    if not blend_mask.any():
        return canvas

    H, W = canvas.shape[:2]
    mask_u8 = blend_mask.astype(np.uint8) * 255

    # Poisson clone 不能讓 mask 接觸圖邊 → 縮掉 1px border
    safe_mask = np.zeros_like(mask_u8)
    safe_mask[1:-1, 1:-1] = mask_u8[1:-1, 1:-1]
    if not safe_mask.any():
        print("      ⚠️  Poisson blend: mask 貼到圖邊，跳過")
        return canvas

    yc, xc = np.where(safe_mask > 0)
    cx, cy = int(np.mean(xc)), int(np.mean(yc))

    # center 也不能太靠邊
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
# 核心: 對單一 target view 做 backward pull harvesting
# ============================================================================
def _harvest_target_view(target_idx, views, all_cam_to_world_mat,
                         view_dirs, scene_d_range,
                         depth_eps_factor=0.05,
                         inpainter=None,
                         tgt_dilation_px: int = 11,
                         use_poisson: bool = True):
    """
    [Fix 2] 先膨脹 target mask → unproject 含 shadow halo 的大洞
    [Fix 1] pull 時用 mask_dilated 做 source 背景檢查
    [Fix 4] 填補完後做 Poisson blend
    """
    target    = views[target_idx]
    H, W      = target["H"], target["W"]
    K_t       = target["K"]
    w2c_t     = all_cam_to_world_mat[target_idx]
    c2w_t     = np.linalg.inv(w2c_t)
    target_img   = target["img"]
    target_depth = target["depth"]
    epsilon   = scene_d_range * depth_eps_factor

    # ── Fix 2: 膨脹 target mask，把 shadow halo 一起列入修補名單 ──
    target_mask_orig = target["mask"] > 0   # 原始 SAM mask，用於 Poisson blend
    if tgt_dilation_px > 0:
        tgt_k = np.ones((tgt_dilation_px, tgt_dilation_px), np.uint8)
        tgt_u8 = target_mask_orig.astype(np.uint8) * 255
        tgt_u8 = cv2.dilate(tgt_u8, tgt_k, iterations=1)
        target_mask = tgt_u8 > 0
    else:
        target_mask = target_mask_orig

    canvas = target_img.copy()
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
    pts_cam_t  = np.stack([x, y, z_t], axis=1)
    pts_h      = np.hstack([pts_cam_t, np.ones((n_holes, 1))])
    pts_world  = (c2w_t @ pts_h.T).T[:, :3]

    # ── Step 2: 排序 source views by viewing angle proximity ─────
    target_dir   = view_dirs[target_idx]
    sims         = view_dirs @ target_dir
    sims[target_idx] = -2.0
    source_order = np.argsort(-sims)

    # ── Step 3: Pull harvest (vectorized) ────────────────────────
    filled       = np.zeros(n_holes, dtype=bool)
    source_used  = np.full(n_holes, -1, dtype=np.int32)
    pts_world_h  = np.hstack([pts_world, np.ones((n_holes, 1))])
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

        # ── Fix 1: 用 mask_dilated 做背景判斷 ────────────────────
        src_mask_check = src_view["mask_dilated"] > 0   # ← Fix 1

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

        # (b) Fix 1: source 背景檢查用膨脹版 mask
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

    # ── Step 4: 統計 ──────────────────────────────────────────────
    n_filled  = int(filled.sum())
    n_dead    = n_holes - n_filled
    pct_pull  = 100 * n_filled / max(n_holes, 1)
    pct_dead  = 100 * n_dead   / max(n_holes, 1)
    n_unique  = len(np.unique(source_used[filled])) if n_filled > 0 else 0
    print(f"      📤 V_{target_idx:3d}: pull={n_filled:,}/{n_holes:,} "
          f"({pct_pull:.1f}%) from {n_unique} src, "
          f"dead={n_dead:,} ({pct_dead:.1f}%) "
          f"(tried {n_sources_tried} src)")

    # ── Step 5: 死角填補 ──────────────────────────────────────────
    dead_mask = np.zeros((H, W), dtype=bool)
    if n_dead > 0:
        dead_idx = np.where(~filled)[0]
        dead_mask[v_t[dead_idx], u_t[dead_idx]] = True

        # 形態學收整（與之前相同）
        dead_u8      = dead_mask.astype(np.uint8) * 255
        close_kernel = np.ones((7, 7), np.uint8)
        dead_u8      = cv2.morphologyEx(dead_u8, cv2.MORPH_CLOSE, close_kernel)
        dilate_kernel = np.ones((5, 5), np.uint8)
        dead_u8      = cv2.dilate(dead_u8, dilate_kernel, iterations=1)
        dead_mask_final = dead_u8 > 0

        _ip = inpainter if inpainter is not None else _get_default_inpainter()
        context = {
            "target_idx":   target_idx,
            "depth":        target_depth,
            "scene_d_range": scene_d_range,
        }
        canvas = _ip.inpaint(canvas, dead_mask_final, context=context)
    else:
        dead_mask_final = dead_mask  # all zeros

    # ── Step 6: Fix 4 — Poisson seamless blending ─────────────────
    # 用 target_mask（已膨脹的版本）對整塊修補區做光度 harmonize
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
    Backward Pull Harvesting Inpaint — Phase 1 Fixed

    在 eval_custom_360.py 的 for-loop 之前注入參數:
      ref_cache["_src_dilation_px"] = 11   # Fix 1
      ref_cache["_tgt_dilation_px"] = 11   # Fix 2
      ref_cache["_use_poisson"]     = True # Fix 4
      ref_cache["_inpainter"]       = build_inpainter("lama")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_dilation_px = ref_cache.get("_src_dilation_px", 11)
    tgt_dilation_px = ref_cache.get("_tgt_dilation_px", 11)
    use_poisson     = ref_cache.get("_use_poisson", True)

    # ── 第一次呼叫: 載入並 cache ──────────────────────────────────
    if "_pullharvest_views" not in ref_cache:
        print("\n[Inpaint-360 Pull-Harvest] 首次呼叫: 載入 view 資料 ...")
        print(f"    Phase 1 fixes: "
              f"src_dilation={src_dilation_px}px | "
              f"tgt_dilation={tgt_dilation_px}px | "
              f"poisson={use_poisson}")
        views = _load_all_views(
            image_paths, mask_paths, mask_dir,
            raw_depth_maps, intrinsics,
            src_mask_dilation_px=src_dilation_px,
        )
        view_dirs    = _compute_view_directions(views, all_cam_to_world_mat)
        scene_d_range = _scene_depth_range(views)
        ref_cache["_pullharvest_views"]    = views
        ref_cache["_pullharvest_view_dirs"] = view_dirs
        ref_cache["_pullharvest_d_range"]  = scene_d_range

        inpainter_obj = ref_cache.get("_inpainter")
        strategy = (getattr(inpainter_obj, "name", "cv2-fallback")
                    if inpainter_obj else "cv2-fallback")
        print(f"    ✅ 載入 {len(views)} views, scene depth range = {scene_d_range:.3f}")
        print(f"    dead-zone 策略: [{strategy}]")

    views        = ref_cache["_pullharvest_views"]
    view_dirs    = ref_cache["_pullharvest_view_dirs"]
    scene_d_range = ref_cache["_pullharvest_d_range"]
    inpainter    = ref_cache.get("_inpainter", None)

    # ── Pull harvest ──────────────────────────────────────────────
    canvas, dead_mask = _harvest_target_view(
        target_idx, views, all_cam_to_world_mat,
        view_dirs, scene_d_range,
        inpainter=inpainter,
        tgt_dilation_px=tgt_dilation_px,
        use_poisson=use_poisson,
    )

    # ── 寫檔 ──────────────────────────────────────────────────────
    out_path = output_dir / f"inpainted_{target_idx}.png"
    cv2.imwrite(str(out_path), canvas)

    deadmask_dir = output_dir.parent / "deadmasks"
    deadmask_dir.mkdir(exist_ok=True)
    dead_path = deadmask_dir / f"deadmask_{target_idx}.png"
    cv2.imwrite(str(dead_path), dead_mask.astype(np.uint8) * 255)

    return int(dead_mask.sum()), 0.0