"""
inpaint_360.py
==============
Backward Pull Harvesting Pipeline

核心思想 (反轉之前的 forward push):

對每個 target view 的破洞 pixel:
  1. 用 target 自己的 VGGT depth 反投影成 3D 世界座標
     (這是你 attention-modified VGGT 取代 3DGIC「先訓練 3DGS 再刪熊」的關鍵)
  2. 投到每個 source view, 三重檢查:
     (a) 在 FOV 內
     (b) source 該位置是 mask outside (真實背景, 不是物體)
     (c) Depth 與 source VGGT depth 一致 (沒被 source 自己的前景擋住)
  3. 從通過檢查的 source 中, 選 viewing direction 最接近的, 取真實 RGB
  4. 所有 source 都不行的真死角 → 交給可插拔的 DeadZoneInpainter

Dead-zone inpainter 切換方式 (在 eval_360.py 的 for-loop 之前設定):
  from eval.dead_zone_inpainter import build_inpainter
  global_ref_cache["_inpainter"] = build_inpainter("lama")  # or "cv2" / "sd"

預設值: cv2 (向下相容). 推薦改成 lama (品質好、確定性).
"""
import os
import cv2
import numpy as np
from pathlib import Path

# 可插拔 dead-zone inpainter (lazy import 保持向下相容)
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


def _load_all_views(image_paths, mask_paths, mask_dir, raw_depth_maps, intrinsics):
    """一次性載入所有 view 的 img/mask/depth/K. Cache 後重複使用."""
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

        depth_low = raw_depth_maps[idx]
        depth = cv2.resize(depth_low.astype(np.float32), (W, H),
                           interpolation=cv2.INTER_NEAREST)
        scale_x = W / depth_low.shape[1]
        scale_y = H / depth_low.shape[0]
        K = intrinsics[idx].copy().astype(np.float64)
        K[0, :] *= scale_x
        K[1, :] *= scale_y

        views.append({
            "img": img, "mask": mask, "depth": depth, "K": K,
            "H": H, "W": W, "idx": idx,
        })
    return views


def _compute_view_directions(views, all_cam_to_world_mat):
    """
    每個 view 的 viewing direction (camera_center → scene_center).
    對 360 場景, 相鄰 view 的 view direction 接近平行,
    180° 對側的 cosine similarity = -1.
    用來排序 source view by 角度接近度.
    """
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
    """所有 view 的有效 depth 範圍. 用來算 epsilon."""
    all_d = np.concatenate([v["depth"][v["depth"] > 0].ravel() for v in views])
    return float(all_d.max() - all_d.min())


def _get_default_inpainter():
    """取得預設 dead-zone inpainter (cv2, 向下相容)."""
    if _INPAINTER_MODULE_AVAILABLE:
        return CV2Inpainter()
    # 若 dead_zone_inpainter.py 不存在, 用 inline fallback
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


# ============================================================================
# 核心: 對單一 target view 做 backward pull harvesting
# ============================================================================
def _harvest_target_view(target_idx, views, all_cam_to_world_mat,
                         view_dirs, scene_d_range,
                         depth_eps_factor=0.05,
                         inpainter=None):
    """
    1. 反投影 target mask interior pixels → 3D world (用 target 自己 VGGT depth)
    2. 對 source view 排序 by viewing angle proximity
    3. 從最接近的 source 開始, vectorized 投影 + 三重檢查 + 取 RGB
       (early stopping when all filled)
    4. 真死角 → inpainter.inpaint() (cv2 / lama / sd, 由 caller 注入)
    """
    target = views[target_idx]
    H, W = target["H"], target["W"]
    K_t = target["K"]
    w2c_t = all_cam_to_world_mat[target_idx]
    c2w_t = np.linalg.inv(w2c_t)
    target_mask = target["mask"] > 0
    target_img = target["img"]
    target_depth = target["depth"]
    epsilon = scene_d_range * depth_eps_factor

    # ── Step 1: 反投影 mask interior 到 3D ─────────────────────
    canvas = target_img.copy()
    v_t, u_t = np.where(target_mask)
    n_holes = len(v_t)
    if n_holes == 0:
        return canvas, np.zeros((H, W), dtype=bool)

    # 取 target 自己的 VGGT depth (attention modified, 是背景 depth)
    z_t = target_depth[v_t, u_t]
    valid_z = (z_t > 0) & np.isfinite(z_t)
    if not valid_z.all():
        median_z = np.median(z_t[valid_z]) if valid_z.any() else 1.0
        z_t = np.where(valid_z, z_t, median_z)

    # 反投影到 target camera coordinate
    x = (u_t - K_t[0, 2]) * z_t / K_t[0, 0]
    y = (v_t - K_t[1, 2]) * z_t / K_t[1, 1]
    pts_cam_t = np.stack([x, y, z_t], axis=1)
    # 轉到 world coordinate
    pts_h = np.hstack([pts_cam_t, np.ones((n_holes, 1))])
    pts_world = (c2w_t @ pts_h.T).T[:, :3]

    # ── Step 2: 排序 source views by viewing angle proximity ─────
    target_dir = view_dirs[target_idx]
    sims = view_dirs @ target_dir
    sims[target_idx] = -2.0
    source_order = np.argsort(-sims)   # 由大到小: 最接近 target 的排前面

    # ── Step 3: 對每個 source 做 pull harvest ──────────────────
    filled = np.zeros(n_holes, dtype=bool)
    source_used = np.full(n_holes, -1, dtype=np.int32)
    pts_world_h = np.hstack([pts_world, np.ones((n_holes, 1))])

    n_sources_tried = 0
    for src_idx in source_order:
        if filled.all():
            break

        unfilled_idx = np.where(~filled)[0]
        if len(unfilled_idx) == 0:
            break

        src_view = views[src_idx]
        src_K = src_view["K"]
        src_w2c = all_cam_to_world_mat[src_idx]
        src_mask = src_view["mask"] > 0
        src_img = src_view["img"]
        src_depth = src_view["depth"]
        H_s, W_s = src_view["H"], src_view["W"]

        pts_check = pts_world_h[unfilled_idx]

        # 投影 unfilled points 到 source view
        pts_src_cam = (src_w2c @ pts_check.T).T[:, :3]
        z_src = pts_src_cam[:, 2]
        valid_z_src = z_src > 0.1
        z_safe = np.where(valid_z_src, z_src, 1.0)
        u_src = src_K[0, 0] * pts_src_cam[:, 0] / z_safe + src_K[0, 2]
        v_src = src_K[1, 1] * pts_src_cam[:, 1] / z_safe + src_K[1, 2]
        u_si = np.where(np.isfinite(u_src), np.round(u_src), -1).astype(np.int32)
        v_si = np.where(np.isfinite(v_src), np.round(v_src), -1).astype(np.int32)

        # ── 三重 Validity Filter ──
        # (a) FOV
        in_bounds = valid_z_src & (u_si >= 0) & (u_si < W_s) & (v_si >= 0) & (v_si < H_s)
        if not in_bounds.any():
            n_sources_tried += 1
            continue
        u_safe = np.clip(u_si, 0, W_s - 1)
        v_safe = np.clip(v_si, 0, H_s - 1)

        # (b) Source mask outside (真背景)
        is_bg_in_src = ~src_mask[v_safe, u_safe]

        # (c) Depth consistency (z_src ≈ source 那邊的 VGGT depth)
        src_d_at = src_depth[v_safe, u_safe]
        valid_d = (src_d_at > 0) & np.isfinite(src_d_at)
        z_diff = np.abs(z_src - src_d_at)
        depth_ok = valid_d & (z_diff < epsilon)

        valid_pull = in_bounds & is_bg_in_src & depth_ok
        if not valid_pull.any():
            n_sources_tried += 1
            continue

        # ── Pull RGB! ──
        local_pull_idx = unfilled_idx[valid_pull]
        u_pull = u_safe[valid_pull]
        v_pull = v_safe[valid_pull]
        canvas[v_t[local_pull_idx], u_t[local_pull_idx]] = src_img[v_pull, u_pull]
        filled[local_pull_idx] = True
        source_used[local_pull_idx] = src_idx
        n_sources_tried += 1

    # ── Step 4: 統計 ──
    n_filled = int(filled.sum())
    n_dead = n_holes - n_filled
    pct_pulled = 100 * n_filled / max(n_holes, 1)
    pct_dead = 100 * n_dead / max(n_holes, 1)
    n_unique_sources = len(np.unique(source_used[filled])) if n_filled > 0 else 0

    print(f"      📤 V_{target_idx:3d}: pull={n_filled:,}/{n_holes:,} "
          f"({pct_pulled:.1f}%) from {n_unique_sources} src, "
          f"dead={n_dead:,} ({pct_dead:.1f}%) "
          f"(tried {n_sources_tried} src)")

    # # ── Step 5: 真死角填補 (可插拔 inpainter) ──
    # dead_mask = np.zeros((H, W), dtype=bool)
    # if n_dead > 0:
    #     dead_idx = np.where(~filled)[0]
    #     dead_mask[v_t[dead_idx], u_t[dead_idx]] = True

    #     _ip = inpainter if inpainter is not None else _get_default_inpainter()
    #     context = {
    #         "target_idx": target_idx,
    #         "depth": target_depth,
    #         "scene_d_range": scene_d_range,
    #     }
    #     canvas = _ip.inpaint(canvas, dead_mask, context=context)
    # ── Step 5: 真死角填補 (可插拔 inpainter) ──
# ── Step 5: 真死角填補 (可插拔 inpainter) ──
    dead_mask = np.zeros((H, W), dtype=bool)
    if n_dead > 0:
        dead_idx = np.where(~filled)[0]
        dead_mask[v_t[dead_idx], u_t[dead_idx]] = True

        # 1. 消除內部碎點 (我們上一動做的事)
        dead_u8 = dead_mask.astype(np.uint8) * 255
        close_kernel = np.ones((7, 7), np.uint8)
        dead_u8 = cv2.morphologyEx(dead_u8, cv2.MORPH_CLOSE, close_kernel)
        
        # =========================================================
        # 🛡️ 2. 邊緣毒點清除 (Boundary Decontamination)
        # 目的：往外擴張 Mask，吃掉當初 SAM 切太緊殘留的物體深色邊緣，
        # 或是 Pull 過來的瑕疵邊緣，確保 LaMa 看到的都是純淨的背景。
        # =========================================================
        # kernel size 設為 5~9 之間通常最安全
        dilate_kernel = np.ones((5, 5), np.uint8) 
        dead_u8 = cv2.dilate(dead_u8, dilate_kernel, iterations=1)
        
        # 轉回 bool 給後續使用
        dead_mask_final = dead_u8 > 0 
        
        _ip = inpainter if inpainter is not None else _get_default_inpainter()
        context = {
            "target_idx": target_idx,
            "depth": target_depth,
            "scene_d_range": scene_d_range,
        }
        
        # 注意這裡傳進去的是膨脹過、吃掉毒邊緣的 dead_mask_final
        canvas = _ip.inpaint(canvas, dead_mask_final, context=context)
        
        # 將最終乾淨的 mask 回傳給外部
        return canvas, dead_mask_final



    return canvas, dead_mask


# ============================================================================
# 主入口: drop-in replacement (與舊版 generative_multi_ref_propagation 相容)
# ============================================================================
def generative_multi_ref_propagation(
    ref_indices, target_idx, image_paths, mask_dir,
    raw_depth_maps, all_cam_to_world_mat, intrinsics,
    output_dir, ref_cache,
    mask_paths=None,
):
    """
    Backward Pull Harvesting Inpaint (drop-in replacement)

    eval_360.py 只要把 import 換成這個模組:
      from eval.inpaint_360 import generative_multi_ref_propagation

    ref_indices 不再使用 (但保留參數相容).
    第一次呼叫時 cache 全部 view 資料, 後續呼叫只是從 cache 讀.

    Dead-zone inpainter 注入方式 (在 eval_360.py for-loop 之前):
      from eval.dead_zone_inpainter import build_inpainter
      global_ref_cache["_inpainter"] = build_inpainter("lama")
    若 cache 裡沒有 "_inpainter" 鍵, 自動 fallback 到 cv2.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 第一次呼叫: 載入並 cache ──
    if "_pullharvest_views" not in ref_cache:
        print("\n[Inpaint-360 Pull-Harvest] 首次呼叫: 載入 view 資料 ...")
        views = _load_all_views(image_paths, mask_paths, mask_dir,
                                raw_depth_maps, intrinsics)
        view_dirs = _compute_view_directions(views, all_cam_to_world_mat)
        scene_d_range = _scene_depth_range(views)
        ref_cache["_pullharvest_views"] = views
        ref_cache["_pullharvest_view_dirs"] = view_dirs
        ref_cache["_pullharvest_d_range"] = scene_d_range

        inpainter_obj = ref_cache.get("_inpainter")
        strategy = getattr(inpainter_obj, "name", "cv2-fallback") if inpainter_obj else "cv2-fallback"
        print(f"    ✅ 載入 {len(views)} views, scene depth range = {scene_d_range:.3f}")
        print(f"    架構: target 反投影 → pull 真實 source RGB → 死角用 [{strategy}]")

    views = ref_cache["_pullharvest_views"]
    view_dirs = ref_cache["_pullharvest_view_dirs"]
    scene_d_range = ref_cache["_pullharvest_d_range"]
    inpainter = ref_cache.get("_inpainter", None)

    # ── Pull harvest 該 target ──
    canvas, dead_mask = _harvest_target_view(
        target_idx, views, all_cam_to_world_mat,
        view_dirs, scene_d_range,
        inpainter=inpainter,
    )

    # ── 寫檔 ──
    out_path = output_dir / f"inpainted_{target_idx}.png"
    cv2.imwrite(str(out_path), canvas)

    # dead mask 放子資料夾，跟 inpainted_*.png 隔開
    deadmask_dir = output_dir.parent / "deadmasks"
    deadmask_dir.mkdir(exist_ok=True)
    dead_path = deadmask_dir / f"deadmask_{target_idx}.png"
    cv2.imwrite(str(dead_path), dead_mask.astype(np.uint8) * 255)

    return int(dead_mask.sum()), 0.0