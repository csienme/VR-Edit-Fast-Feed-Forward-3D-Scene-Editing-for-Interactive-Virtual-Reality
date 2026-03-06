import cv2
import numpy as np
import torch
import os
from PIL import Image




try:
    from simple_lama_inpainting import SimpleLama
    print("⏳ 正在載入 LaMa 模型至 GPU...")
    lama_model = SimpleLama()
    print("✅ LaMa 模型載入完成！")
except ImportError:
    print("⚠️ 尚未安裝 simple-lama-inpainting，請執行 pip install simple-lama-inpainting")
    lama_model = None

# ==============================================================================
# [模組 1] 2D 深度圖生成式修補 (LaMa RGB-D 雙軌管線)
# ==============================================================================
# ==============================================================================
# [模組 1] 2D 深度圖生成式修補 (LaMa RGB-D 雙軌管線)
# ==============================================================================
def run_generative_depth_inpaint(dense_depth, mask_2d):
    """
    將 VGGT 絕對深度轉換為視覺圖像，讓 LaMa 進行高頻結構修補，再反推回物理深度。
    """
    valid_mask = ~np.isnan(dense_depth) & ~np.isinf(dense_depth) & (dense_depth > 0)
    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(mask_2d, kernel, iterations=1)

    full_inpaint_mask = np.zeros_like(mask_2d, dtype=np.uint8)
    full_inpaint_mask[~valid_mask] = 255
    full_inpaint_mask[dilated_mask > 0] = 255

    safe_bg_mask = valid_mask & (full_inpaint_mask == 0)
    if not np.any(safe_bg_mask):
        return dense_depth, None, None 
    
    depth_min, depth_max = np.percentile(dense_depth[safe_bg_mask], 1), np.percentile(dense_depth[safe_bg_mask], 99)

    depth_norm = np.zeros_like(dense_depth, dtype=np.float32)
    depth_norm[safe_bg_mask] = dense_depth[safe_bg_mask]
    depth_img_8u = np.clip((depth_norm - depth_min) / (depth_max - depth_min + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    depth_img_3c = cv2.cvtColor(depth_img_8u, cv2.COLOR_GRAY2RGB)

    if lama_model is not None:
        img_pil = Image.fromarray(depth_img_3c)
        mask_pil = Image.fromarray(full_inpaint_mask).convert('L')
        result_pil = lama_model(img_pil, mask_pil)
        inpainted_depth_8u = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2GRAY)
        
        # 💥【關鍵修復】：LaMa 可能會填充圖片為 8 的倍數，強制改回原本長寬！
        H, W = mask_2d.shape
        if inpainted_depth_8u.shape[:2] != (H, W):
            inpainted_depth_8u = cv2.resize(inpainted_depth_8u, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        inpainted_depth_8u = cv2.inpaint(depth_img_8u, full_inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    inpainted_depth_8u = cv2.medianBlur(inpainted_depth_8u, 7)
    depth_inpainted_metric = inpainted_depth_8u.astype(np.float32) / 255.0 * (depth_max - depth_min) + depth_min

    final_depth = dense_depth.copy()
    final_depth[full_inpaint_mask > 0] = depth_inpainted_metric[full_inpaint_mask > 0]

    return final_depth, depth_img_8u, inpainted_depth_8u

# ==============================================================================
# [模組 2] 2D RGB 生成式修補 (LaMa / Stable Diffusion Wrapper)
# ==============================================================================
# ==============================================================================
# [模組 2] 2D RGB 生成式修補 (LaMa / Stable Diffusion Wrapper)
# ==============================================================================
def run_generative_rgb_inpaint(img_path, mask_2d):
    """
    呼叫 LaMa 進行修補，並預先膨脹 Mask 以消除陰影殘留與邊界洩漏。
    """
    kernel = np.ones((15, 15), np.uint8)
    mask_dilated = cv2.dilate(mask_2d, kernel, iterations=1)

    if lama_model is not None:
        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.fromarray(mask_dilated).convert('L')
        result_pil = lama_model(img_pil, mask_pil)
        inpainted_rgb = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        
        # 💥【關鍵修復】：確保 RGB 修補出來的圖片尺寸與原圖完美吻合
        H, W = mask_2d.shape
        if inpainted_rgb.shape[:2] != (H, W):
            inpainted_rgb = cv2.resize(inpainted_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.imread(img_path)
        inpainted_rgb = cv2.inpaint(img, mask_dilated, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        
    return inpainted_rgb
# ==============================================================================
# [模組 3] 核心：Image-to-PointCloud 昇維投影
# ==============================================================================
# ==============================================================================
# [模組 3] 核心：Image-to-PointCloud 昇維投影 (LaMa RGB-D 雙軌版)
# ==============================================================================
# ==============================================================================
# [模組 3] 核心：Image-to-PointCloud 逆向映射 (Target-Driven Backward Warping)
# ==============================================================================
# ==============================================================================
# [模組 3] 核心：多參考視角逆向映射 (Multi-Reference Backward Warping)
# ==============================================================================
# ==============================================================================
# [模組 3] 核心：多參考視角前向融合 (Multi-Ref Forward Splatting + Local Registration)
# ==============================================================================
def generative_multi_ref_propagation(
    ref_indices, target_idx, image_paths, mask_dir, 
    dense_depth_maps, all_cam_to_world_mat, intrinsics, 
    output_dir, ref_cache
):
    print(f"\n[Gen-3D Prop] 啟動多視角前向融合: Target V_{target_idx} <- Refs {ref_indices}")

    # ==========================================
    # 0. 讀取 Target 基礎資料
    # ==========================================
    target_img_path = image_paths[target_idx]
    target_img = cv2.imread(target_img_path)
    H, W = target_img.shape[:2]
    
    target_mask_path = os.path.join(mask_dir, os.path.basename(target_img_path))
    mask_tgt = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
    mask_tgt = cv2.resize(mask_tgt, (W, H), interpolation=cv2.INTER_NEAREST)

    w2c_tgt = all_cam_to_world_mat[target_idx]
    depth_vt_lowres = dense_depth_maps[target_idx]
    scale_x, scale_y = W / depth_vt_lowres.shape[1], H / depth_vt_lowres.shape[0]
    K_tgt = intrinsics[target_idx].copy()
    K_tgt[0, :] *= scale_x
    K_tgt[1, :] *= scale_y

    final_canvas = target_img.copy()
    remaining_hole_mask = (mask_tgt > 0).copy()
    
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 建立 Target 參考環 (供後續方案一對齊使用)
    kernel_expand_tgt = np.ones((25, 25), np.uint8) 
    expanded_mask_tgt = cv2.dilate(mask_tgt, kernel_expand_tgt, iterations=1)

    # ==========================================
    # 1. 貪婪圖層覆蓋：依序從多個 Ref 進行前向投影
    # ==========================================
    for ref_idx in ref_indices:
        if not np.any(remaining_hole_mask):
            print(f"🎉 Target V_{target_idx} 的死角已被先前的 Ref 完美填滿！")
            break

        print(f"  -> 正在從 Ref V_{ref_idx} 擷取並映射 3D 補丁...")

        # (A) 從快取取得 Ref 的 RGB-D
        ref_img_path = image_paths[ref_idx]
        ref_mask_path = os.path.join(mask_dir, os.path.basename(ref_img_path))
        mask_ref = cv2.resize(cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST)
        
        if ref_idx not in ref_cache:
            d_ref = cv2.resize(dense_depth_maps[ref_idx], (W, H), interpolation=cv2.INTER_NEAREST)
            rgb_ref = run_generative_rgb_inpaint(ref_img_path, mask_ref)
            depth_ref, _, _ = run_generative_depth_inpaint(d_ref, mask_ref)
            ref_cache[ref_idx] = (rgb_ref, depth_ref)
            
        inpainted_rgb_ref, inpainted_depth_ref = ref_cache[ref_idx]

        # (B) 擷取 Ref 的 3D 實體補丁
        kernel_expand_ref = np.ones((25, 25), np.uint8) 
        expanded_mask_ref = cv2.dilate(mask_ref, kernel_expand_ref, iterations=1)
        v_ref, u_ref = np.where(expanded_mask_ref > 0)
        
        Z_ref = inpainted_depth_ref[v_ref, u_ref]
        valid_z = ~np.isnan(Z_ref) & ~np.isinf(Z_ref) & (Z_ref > 0)
        u_ref, v_ref, Z_ref = u_ref[valid_z], v_ref[valid_z], Z_ref[valid_z]
        colors_ref = inpainted_rgb_ref[v_ref, u_ref]

        # (C) 逆投影至 World Space
        c2w_ref = np.linalg.inv(all_cam_to_world_mat[ref_idx])
        K_ref = intrinsics[ref_idx].copy()
        K_ref[0, :] *= scale_x
        K_ref[1, :] *= scale_y

        X_cam = (u_ref - K_ref[0, 2]) * Z_ref / K_ref[0, 0]
        Y_cam = (v_ref - K_ref[1, 2]) * Z_ref / K_ref[1, 1]
        pts_cam = np.column_stack((X_cam, Y_cam, Z_ref))
        pts_cam_homo = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
        pts_world = (c2w_ref @ pts_cam_homo.T).T[:, :3]

        # (D) 前向濺射至 Target 視角
        pts_world_homo = np.hstack((pts_world, np.ones((pts_world.shape[0], 1))))
        pts_tgt_cam = (w2c_tgt @ pts_world_homo.T).T[:, :3]
        
        valid_z_tgt = pts_tgt_cam[:, 2] > 0.1
        pts_tgt_cam = pts_tgt_cam[valid_z_tgt]
        final_colors = colors_ref[valid_z_tgt]
        
        pts_2d_tgt = (K_tgt @ pts_tgt_cam.T).T
        Z_tgt_proj = pts_tgt_cam[:, 2]
        u_tgt = (pts_2d_tgt[:, 0] / Z_tgt_proj).astype(np.int32)
        v_tgt = (pts_2d_tgt[:, 1] / Z_tgt_proj).astype(np.int32)
        
        valid_uv = (u_tgt >= 0) & (u_tgt < W) & (v_tgt >= 0) & (v_tgt < H)
        u_tgt, v_tgt, final_colors = u_tgt[valid_uv], v_tgt[valid_uv], final_colors[valid_uv]

        # (E) 渲染暫存畫布與形態學平滑
        warp_canvas = np.zeros_like(target_img)
        valid_warp_mask = np.zeros((H, W), dtype=np.uint8)
        
        for v, u, color in zip(v_tgt, u_tgt, final_colors):
            cv2.circle(warp_canvas, (u, v), 3, color.tolist(), -1)
            cv2.circle(valid_warp_mask, (u, v), 3, 255, -1)
            
        valid_warp_mask_smoothed = cv2.morphologyEx(valid_warp_mask, cv2.MORPH_OPEN, morph_kernel)
        valid_warp_mask_bool = valid_warp_mask_smoothed > 0

        # ==========================================
        # 2. 【方案一】 局部幾何配準 (Local Registration)
        # ==========================================
        ring_mask = (expanded_mask_tgt > 0) & (mask_tgt == 0) & valid_warp_mask_bool
        best_dx, best_dy = 0, 0
        
        if np.any(ring_mask):
            y_idx, x_idx = np.where(ring_mask)
            y_min, y_max = max(0, y_idx.min()-15), min(H, y_idx.max()+15)
            x_min, x_max = max(0, x_idx.min()-15), min(W, x_idx.max()+15)
            
            tgt_crop = target_img[y_min:y_max, x_min:x_max].astype(np.float32)
            warp_crop = warp_canvas[y_min:y_max, x_min:x_max].astype(np.float32)
            ring_crop = ring_mask[y_min:y_max, x_min:x_max]
            
            min_error = float('inf')
            for dy in range(-10, 11):
                for dx in range(-10, 11):
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    shifted_warp = cv2.warpAffine(warp_crop, M, (warp_crop.shape[1], warp_crop.shape[0]))
                    diff = tgt_crop[ring_crop] - shifted_warp[ring_crop]
                    error = np.mean(np.square(diff))
                    if error < min_error:
                        min_error = error
                        best_dx, best_dy = dx, dy
            print(f"    🎯 [方案一觸發] 對齊偏移量 (dx={best_dx}, dy={best_dy})")

        # (F) 施加平移並填補「剩餘的洞」
        M_best = np.float32([[1, 0, best_dx], [0, 1, best_dy]])
        shifted_canvas = cv2.warpAffine(warp_canvas, M_best, (W, H))
        shifted_mask = cv2.warpAffine(valid_warp_mask_smoothed, M_best, (W, H))
        
        paste_mask = (shifted_mask > 0) & remaining_hole_mask
        paste_y, paste_x = np.where(paste_mask)
        
        final_canvas[paste_y, paste_x] = shifted_canvas[paste_y, paste_x]
        
        # 標記已填補區域，供下一輪 Ref 參考
        remaining_hole_mask[paste_y, paste_x] = False
        
    # ==========================================
    # 3. 絕對死角結算
    # ==========================================
    red_mask_smoothed = cv2.morphologyEx((remaining_hole_mask * 255).astype(np.uint8), cv2.MORPH_OPEN, morph_kernel)
    red_y, red_x = np.where(red_mask_smoothed > 0)
    final_canvas[red_y, red_x] = [0, 0, 255]
    
    os.makedirs(output_dir / "gen_3d_prop", exist_ok=True)
    cv2.imwrite(str(output_dir / "gen_3d_prop" / f"inpainted_{target_idx}.png"), final_canvas)
    
    red_area = len(red_y)
    print(f"✅ V_{target_idx} 處理完成！剩餘紅色死角面積: {red_area} 像素\n")
    
    return red_area