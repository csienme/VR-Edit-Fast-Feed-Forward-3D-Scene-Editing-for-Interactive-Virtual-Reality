"""
inpaint_module_ablation_no_transfer.py  ── ABLATION: w/o Pixel Transfer (FIXED)
================================================================================
FORGE 的 "w/o pixel transfer" ablation 版本。

目的：證明 FORGE 的 FID 優勢來自「幾何引導的真實 RGB 重用 (pull harvest)」。
做法：移除整段 pull harvest（不從任何鄰近視角搬真實像素），
      將「整個 target mask 區域」全部視為 dead zone，直接交給 SD inpainter 生成。

與 Full pipeline 的唯一差異（單變量）：
  Full : mask 區 → pull harvest 搬真實 RGB → 剩餘 dead zone → SD
  本版 : mask 區 → (不搬) → 整個 mask = dead zone → SD

★ FIX (vs 初版)：補回 depth + scene_d_range 給 SD context。
   SD-ControlNet (use_controlnet=True) 需要 context["depth"] 建 control_image，
   否則 control_image=None 會報錯。depth 是 SD 的幾何條件輸入，不屬於
   pixel transfer，補回來不破壞 ablation 語意（唯一差別仍是「無 pull harvest」）。

刻意「保持一致」以維持單變量：
  • target mask 界定：同樣套用 tgt_dilation_px
  • dead_mask 形態學：同樣 close(11)+dilate(5)+CC 清理(min_blob 500)
  • SD context：同樣給 target_idx + depth + scene_d_range（與 Full 一致）
  • inpainter：同樣由 ref_cache["_inpainter"] 提供（sd）

刻意「移除」（沒有 pulled pixel 可作用）：
  pull harvest 主迴圈、Strategy B、Fix A、Fix B、Shadow M1/M2、Fix D、Poisson

函式簽章與原版完全相同 → drop-in。
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


def _resolve_mask_path(idx, image_paths, mask_paths, mask_dir):
    if mask_paths is not None:
        return str(mask_paths[idx])
    return os.path.join(str(mask_dir), os.path.basename(image_paths[idx]))


def _load_all_views(image_paths, mask_paths, mask_dir, raw_depth_maps, intrinsics,
                    src_mask_dilation_px: int = 11):
    """載入 img/mask/depth（depth 供 SD-ControlNet control 條件用，與 Full 一致）。"""
    views = []
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"讀不到 image: {img_path}")
        H, W = img.shape[:2]
        mask_path = _resolve_mask_path(idx, image_paths, mask_paths, mask_dir)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"讀不到 mask: {mask_path}")
        mask = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        depth_low = raw_depth_maps[idx]
        depth = cv2.resize(depth_low.astype(np.float32), (W, H),
                           interpolation=cv2.INTER_NEAREST)
        views.append({"img": img, "mask": mask, "depth": depth,
                      "H": H, "W": W, "idx": idx})
    return views


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


def _inpaint_target_view(target_idx, views, scene_d_range,
                         inpainter=None, tgt_dilation_px: int = 5):
    target       = views[target_idx]
    H, W         = target["H"], target["W"]
    target_img   = target["img"]
    target_depth = target["depth"]

    target_mask_orig = target["mask"] > 0
    if tgt_dilation_px > 0:
        tgt_k  = np.ones((tgt_dilation_px, tgt_dilation_px), np.uint8)
        tgt_u8 = target_mask_orig.astype(np.uint8) * 255
        tgt_u8 = cv2.dilate(tgt_u8, tgt_k, iterations=1)
        target_mask = tgt_u8 > 0
    else:
        target_mask = target_mask_orig

    canvas  = target_img.copy()
    n_holes = int(target_mask.sum())
    if n_holes == 0:
        return canvas, np.zeros((H, W), dtype=bool)

    dead_mask = target_mask.copy()
    print(f"      V_{target_idx:3d}: [w/o pixel transfer] "
          f"整個 mask = dead zone ({n_holes:,} px) -> SD inpaint")

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

    if not dead_mask_final.any():
        return canvas, dead_mask_final

    _ip = inpainter if inpainter is not None else _get_default_inpainter()
    context = {
        "target_idx":    target_idx,
        "depth":         target_depth,
        "scene_d_range": scene_d_range,
    }
    canvas = _ip.inpaint(canvas, dead_mask_final, context=context)
    if canvas.shape[:2] != (H, W):
        print(f"      Size mismatch {canvas.shape[:2]} -> ({H},{W}), resizing")
        canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)

    return canvas, dead_mask_final


def generative_multi_ref_propagation(
    ref_indices, target_idx, image_paths, mask_dir,
    raw_depth_maps, all_cam_to_world_mat, intrinsics,
    output_dir, ref_cache, mask_paths=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_dilation_px = ref_cache.get("_src_dilation_px", 11)
    tgt_dilation_px = ref_cache.get("_tgt_dilation_px", 5)

    if "_pullharvest_views" not in ref_cache:
        print("\n[Inpaint-360 ABLATION: w/o pixel transfer] 首次呼叫: 載入 view 資料 ...")
        print(f"    tgt_dil={tgt_dilation_px}px | 整個 mask 直接走 SD（無 pull harvest）")
        views = _load_all_views(image_paths, mask_paths, mask_dir,
                                raw_depth_maps, intrinsics,
                                src_mask_dilation_px=src_dilation_px)
        ref_cache["_pullharvest_views"]   = views
        ref_cache["_pullharvest_d_range"] = _scene_depth_range(views)
        inpainter_obj = ref_cache.get("_inpainter")
        strategy = (getattr(inpainter_obj, "name", "cv2-fallback")
                    if inpainter_obj else "cv2-fallback")
        print(f"    {len(views)} views | dead-zone=[{strategy}] | "
              f"ABLATION MODE: pixel transfer DISABLED")

    views         = ref_cache["_pullharvest_views"]
    scene_d_range = ref_cache["_pullharvest_d_range"]
    inpainter     = ref_cache.get("_inpainter", None)

    canvas, dead_mask = _inpaint_target_view(
        target_idx, views, scene_d_range,
        inpainter=inpainter, tgt_dilation_px=tgt_dilation_px,
    )

    out_path = output_dir / f"inpainted_{target_idx}.png"
    cv2.imwrite(str(out_path), canvas)

    deadmask_dir = output_dir.parent / "deadmasks"
    deadmask_dir.mkdir(exist_ok=True)
    dead_path = deadmask_dir / f"deadmask_{target_idx}.png"
    cv2.imwrite(str(dead_path), dead_mask.astype(np.uint8) * 255)

    return int(dead_mask.sum()), 0.0