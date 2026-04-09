import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
from typing import Optional, List, Tuple

try:
    from simple_lama_inpainting import SimpleLama
    print("⏳ 正在載入 LaMa 模型至 GPU (Geometry Grounded Pipeline)...")
    lama_model = SimpleLama()
    print("✅ LaMa 模型載入完成")
except ImportError:
    print("⚠️  simple-lama-inpainting 未安裝，將無法執行死角填補")
    lama_model = None

# ==============================================================================
# I. 基礎工具與投影
# ==============================================================================
def _lama_fill(canvas_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    if lama_model is None or not mask_u8.any(): return canvas_bgr.copy()
    H, W = canvas_bgr.shape[:2]
    img_pil  = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask_u8).convert('L')
    result   = lama_model(img_pil, mask_pil)
    out = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    if out.shape[:2] != (H, W): out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
    return out

def _project_pts_to_frame(pts_world: np.ndarray, w2c: np.ndarray, K: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_h = np.c_[pts_world, np.ones(len(pts_world))]
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    valid = pts_cam[:, 2] > 0.05
    uv = (K @ pts_cam.T).T
    z  = pts_cam[:, 2].clip(min=1e-6) 
    u  = (uv[:, 0] / z)
    v  = (uv[:, 1] / z)
    # 允許些微超出邊界以防止邊緣裁切
    valid &= (u >= -5) & (u < W + 5) & (v >= -5) & (v < H + 5)
    return u, v, valid

# ==============================================================================
# II. 核心 SOTA：邊界深度對齊 (Boundary-Guided Depth Alignment)
# ==============================================================================
def boundary_guided_depth_alignment(
    lama_depth: np.ndarray,      
    vggt_depth: np.ndarray,      
    mask: np.ndarray,            
    border_width: int = 15       
) -> np.ndarray:
    """
    用 mask 邊界的 VGGT 真實深度，線性回歸算出 scale 和 shift，
    強制 LaMa 生成的深度對齊 VGGT 幾何空間。
    """
    dilated = cv2.dilate(mask, np.ones((border_width, border_width), np.uint8))
    border_ring = (dilated > 0) & (mask == 0)

    valid = border_ring & ~np.isnan(vggt_depth) & (vggt_depth > 0) & ~np.isnan(lama_depth)
    if valid.sum() < 10:
        return lama_depth  # 邊界點不夠，不做對齊

    D_vggt = vggt_depth[valid].flatten()
    D_lama = lama_depth[valid].flatten()

    # Least Squares: D_vggt ≈ scale * D_lama + shift
    A = np.c_[D_lama, np.ones_like(D_lama)]
    result, _, _, _ = np.linalg.lstsq(A, D_vggt, rcond=None)
    scale, shift = result[0], result[1]

    print(f"    📐 [Depth Align] 深度幾何校正: scale={scale:.4f}, shift={shift:.4f}")

    aligned = scale * lama_depth + shift
    
    # 防呆：避免離群值暴衝
    min_val, max_val = np.nanmin(vggt_depth), np.nanmax(vggt_depth)
    aligned = np.clip(aligned, min_val * 0.2, max_val * 3.0)
    
    return aligned

# ==============================================================================
# III. 基礎修補模組
# ==============================================================================
def run_generative_depth_inpaint(dense_depth, mask_2d):
    valid_mask = ~np.isnan(dense_depth) & ~np.isinf(dense_depth) & (dense_depth > 0)
    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(mask_2d, kernel, iterations=1)

    full_inpaint_mask = np.zeros_like(mask_2d, dtype=np.uint8)
    full_inpaint_mask[~valid_mask] = 255
    full_inpaint_mask[dilated_mask > 0] = 255

    safe_bg_mask = valid_mask & (full_inpaint_mask == 0)
    if not np.any(safe_bg_mask): return dense_depth, None, None 
    
    depth_min, depth_max = np.percentile(dense_depth[safe_bg_mask], 1), np.percentile(dense_depth[safe_bg_mask], 99)
    depth_norm = np.zeros_like(dense_depth, dtype=np.float32)
    depth_norm[safe_bg_mask] = dense_depth[safe_bg_mask]
    depth_img_8u = np.clip((depth_norm - depth_min) / (depth_max - depth_min + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    if lama_model is not None:
        d3c = cv2.cvtColor(depth_img_8u, cv2.COLOR_GRAY2RGB)
        res = lama_model(Image.fromarray(d3c), Image.fromarray(full_inpaint_mask).convert('L'))
        inpainted_depth_8u = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2GRAY)
        H, W = mask_2d.shape
        if inpainted_depth_8u.shape[:2] != (H, W):
            inpainted_depth_8u = cv2.resize(inpainted_depth_8u, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        inpainted_depth_8u = cv2.inpaint(depth_img_8u, full_inpaint_mask, 5, cv2.INPAINT_TELEA)

    inpainted_depth_8u = cv2.medianBlur(inpainted_depth_8u, 7)
    
    # 轉回 Metric Scale
    depth_inpainted_metric = inpainted_depth_8u.astype(np.float32) / 255.0 * (depth_max - depth_min) + depth_min

    # 💥 專家修正：邊界導向深度校正
    aligned_depth_metric = boundary_guided_depth_alignment(
        lama_depth=depth_inpainted_metric,
        vggt_depth=dense_depth,
        mask=mask_2d,
        border_width=15
    )

    final_depth = dense_depth.copy()
    final_depth[full_inpaint_mask > 0] = aligned_depth_metric[full_inpaint_mask > 0]
    
    return final_depth, depth_img_8u, inpainted_depth_8u

def run_generative_rgb_inpaint(img_path, mask_2d):
    kernel = np.ones((15, 15), np.uint8)
    mask_dilated = cv2.dilate(mask_2d, kernel, iterations=1)

    if lama_model is not None:
        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.fromarray(mask_dilated).convert('L')
        result_pil = lama_model(img_pil, mask_pil)
        inpainted_rgb = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        H, W = mask_2d.shape
        if inpainted_rgb.shape[:2] != (H, W):
            inpainted_rgb = cv2.resize(inpainted_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        inpainted_rgb = cv2.inpaint(cv2.imread(img_path), mask_dilated, 10, cv2.INPAINT_TELEA)
    return inpainted_rgb

# ==============================================================================
# IV. 主調用端 (完美幾何對齊 + Z-Buffer 銳利覆寫)
# ==============================================================================
def generative_multi_ref_propagation(
    ref_indices: List[int], target_idx: int, image_paths: List[str], mask_dir: str, 
    dense_depth_maps: List[np.ndarray], all_cam_to_world_mat: List[np.ndarray], intrinsics: np.ndarray, 
    output_dir: Path, ref_cache: dict,
    # 預留 kwargs 讓 eval_custom.py 傳入特徵時不報錯
    **kwargs
) -> Tuple[int, float]:
    
    print(f"\n[Geometry Prop] 啟動純幾何深度對齊融合: Target V_{target_idx} ← Refs {ref_indices}")

    target_img_path = image_paths[target_idx]
    target_img = cv2.imread(target_img_path)
    H, W = target_img.shape[:2]
    
    target_mask_path = os.path.join(mask_dir, os.path.basename(target_img_path))
    mask_tgt = cv2.resize(cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST)

    final_canvas = target_img.copy()
    remaining_hole_mask = (mask_tgt > 0).copy()
    
    kernel_expand_tgt = np.ones((25, 25), np.uint8) 
    expanded_mask_tgt = cv2.dilate(mask_tgt, kernel_expand_tgt, iterations=1)
    boundary_mses = []

    target_c2w = np.linalg.inv(all_cam_to_world_mat[target_idx])
    target_pos = target_c2w[:3, 3]
    ref_distances = []
    
    for r_idx in ref_indices:
        r_c2w = np.linalg.inv(all_cam_to_world_mat[r_idx])
        dist = np.linalg.norm(r_c2w[:3, 3] - target_pos)
        ref_distances.append((r_idx, dist))
        
    sorted_ref_indices = [x[0] for x in sorted(ref_distances, key=lambda x: x[1])]

    for ref_idx in sorted_ref_indices:
        if not np.any(remaining_hole_mask): break
        print(f"  -> 擷取 Ref V_{ref_idx} 進行精確 3D 幾何映射...")

        ref_img_path = image_paths[ref_idx]
        ref_mask_path = os.path.join(mask_dir, os.path.basename(ref_img_path))
        mask_ref = cv2.resize(cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST)
        
        if ref_idx not in ref_cache:
            d_ref_scaled = cv2.resize(dense_depth_maps[ref_idx].astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
            rgb_ref_filled = run_generative_rgb_inpaint(ref_img_path, mask_ref)
            depth_ref_filled, _, _ = run_generative_depth_inpaint(d_ref_scaled, mask_ref)
            ref_cache[ref_idx] = (rgb_ref_filled, depth_ref_filled)
            
        inpainted_rgb_ref, inpainted_depth_ref = ref_cache[ref_idx]

        expanded_mask_ref = cv2.dilate(mask_ref, np.ones((25, 25), np.uint8), iterations=1)
        v_ref, u_ref = np.where(expanded_mask_ref > 0)
        
        Z_ref = inpainted_depth_ref[v_ref, u_ref]
        valid_z = ~np.isnan(Z_ref) & ~np.isinf(Z_ref) & (Z_ref > 0)
        v_ref, u_ref, Z_ref = v_ref[valid_z], u_ref[valid_z], Z_ref[valid_z]
        colors = inpainted_rgb_ref[v_ref, u_ref]

        # 💥 專家修正：絕對精確的 Intrinsics Scaling
        orig_h, orig_w = dense_depth_maps[ref_idx].shape[:2]
        sx, sy = W / orig_w, H / orig_h
        
        K_ref = intrinsics[ref_idx].copy()
        K_ref[0, :] *= sx; K_ref[1, :] *= sy
        K_tgt = intrinsics[target_idx].copy()
        K_tgt[0, :] *= sx; K_tgt[1, :] *= sy

        X = (u_ref - K_ref[0, 2]) * Z_ref / K_ref[0, 0]
        Y = (v_ref - K_ref[1, 2]) * Z_ref / K_ref[1, 1]
        c2w_ref = np.linalg.inv(all_cam_to_world_mat[ref_idx])
        pts_w = (c2w_ref @ np.c_[X, Y, Z_ref, np.ones(len(Z_ref))].T).T[:, :3]

        u_t, v_t, valid = _project_pts_to_frame(pts_w, all_cam_to_world_mat[target_idx], K_tgt, H, W)
        u_t, v_t = u_t[valid], v_t[valid]
        colors = colors[valid]
        Z_ref_valid = Z_ref[valid]

        # 由於 3D 幾何已經完美校正，拔除所有 2D dx, dy 的補償，直接交給 Z-Buffer！
        # 引入 stable sort 保留 2D 晶格連貫性
        sort_idx = np.argsort(-Z_ref_valid, kind='stable')
        u_sorted, v_sorted = np.round(u_t[sort_idx]).astype(np.int32), np.round(v_t[sort_idx]).astype(np.int32)
        colors_sorted = colors[sort_idx]

        warp_canvas = np.zeros_like(target_img)
        valid_warp_mask = np.zeros((H, W), dtype=np.uint8)
        
        # 銳利繪製
        for u, v, c in zip(u_sorted, v_sorted, colors_sorted):
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(warp_canvas, (u, v), 1, c.tolist(), -1)
                cv2.circle(valid_warp_mask, (u, v), 1, 255, -1)
            
        valid_warp_mask_smoothed = cv2.morphologyEx(valid_warp_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        current_eval_ring = (expanded_mask_tgt > 0) & (mask_tgt == 0) & (valid_warp_mask_smoothed > 0)
        if np.any(current_eval_ring):
            diff = warp_canvas[current_eval_ring].astype(np.float32) - target_img[current_eval_ring].astype(np.float32)
            patch_mse = float(np.mean(np.square(diff)))
            boundary_mses.append(patch_mse)
            print(f"    📊 幾何映射後，補丁邊界 MSE 誤差: {patch_mse:.2f}")

        paste_mask = (valid_warp_mask_smoothed > 0) & remaining_hole_mask
        if paste_mask.any():
            final_canvas[paste_mask] = warp_canvas[paste_mask]
            remaining_hole_mask[paste_mask] = False
        
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask_smoothed = cv2.morphologyEx((remaining_hole_mask * 255).astype(np.uint8), cv2.MORPH_OPEN, morph_kernel)
    red_y, red_x = np.where(red_mask_smoothed > 0)
    
    final_canvas_u8 = np.clip(final_canvas, 0, 255).astype(np.uint8)
    final_canvas_u8[red_y, red_x] = [0, 0, 255]
    red_area = len(red_y)
    
    if red_area > 0 and lama_model is not None:
        print(f"    🌟 啟動 LaMa 死角填補 (面積: {red_area})...")
        final_canvas_u8 = _lama_fill(final_canvas_u8, red_mask_smoothed)
    
    final_boundary_mse = max(boundary_mses) if boundary_mses else 0.0
    
    os.makedirs(output_dir / "gen_3d_prop_new", exist_ok=True)
    cv2.imwrite(str(output_dir / "gen_3d_prop_new" / f"inpainted_{target_idx}.png"), final_canvas_u8)
    
    print(f"✅ V_{target_idx} 處理完成！剩餘死角: {red_area} px，最高 MSE: {final_boundary_mse:.2f}")
    return red_area, final_boundary_mse