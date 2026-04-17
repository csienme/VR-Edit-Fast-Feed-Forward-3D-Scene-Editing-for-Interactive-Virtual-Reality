import cv2
import numpy as np
import torch
import os
from PIL import Image
from sklearn.linear_model import RANSACRegressor

# ==============================================================================
# [模組 1] 🦴 核心魔法：3D 幾何外推與直角重建 (3D Geometry Extrapolation)
# ==============================================================================
def extrapolate_3d_geometry(raw_depth, mask_2d, K):
    """
    純粹基於 3D 數學，利用洞口邊緣的點雲，延伸並重建出帶有銳利直角的幾何骨架。
    """
    H, W = mask_2d.shape

    # 確保 mask 是 0/255 格式
    mask_2d_255 = (mask_2d > 0).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(mask_2d_255, kernel, iterations=1)
    
    # 邊界 = 膨脹後的範圍 減去 原本的破洞 且 深度數值有效
    boundary_mask = (dilated_mask > 0) & (mask_2d == 0) & (raw_depth > 0) & ~np.isnan(raw_depth)
    v_b, u_b = np.where(boundary_mask)
    z_b = raw_depth[v_b, u_b]

    # 防呆機制
    if len(z_b) < 50:
        print("    ⚠️ 邊界點不足，跳過 3D 幾何外推！")
        return raw_depth.copy()

    # 將 2D 邊界反投影到 3D 空間
    x_b = (u_b - K[0, 2]) * z_b / K[0, 0]
    y_b = (v_b - K[1, 2]) * z_b / K[1, 1]
    pts_3d = np.column_stack((x_b, y_b, z_b))

    # 雙平面擬合 (Two-Plane Fitting via RANSAC)
    XY = pts_3d[:, :2]
    Z = pts_3d[:, 2]

    ransac1 = RANSACRegressor(residual_threshold=0.05, random_state=42)
    ransac1.fit(XY, Z)
    inlier_mask1 = ransac1.inlier_mask_

    XY_rem = XY[~inlier_mask1]
    Z_rem = Z[~inlier_mask1]
    

    global has_plane2
    has_plane2 = False

    if len(XY_rem) > 20:
        ransac2 = RANSACRegressor(residual_threshold=0.05, random_state=42)
        ransac2.fit(XY_rem, Z_rem)
        has_plane2 = True
        print(f"    🦴 [幾何重建] 成功擷取雙平面結構 (長椅與地板)！")
    else:
        print(f"    🦴 [幾何重建] 僅偵測到單一連續平面！")

    v_h, u_h = np.where(mask_2d > 0)
    ray_x = (u_h - K[0, 2]) / K[0, 0]
    ray_y = (v_h - K[1, 2]) / K[1, 1]
    rays_xy = np.column_stack((ray_x, ray_y))

    def compute_z_on_plane(ransac_model, rays):
        a, b = ransac_model.estimator_.coef_
        c = ransac_model.estimator_.intercept_
        denominator = 1.0 - (a * rays[:, 0] + b * rays[:, 1])
        denominator[np.abs(denominator) < 1e-6] = 1e-6 
        return c / denominator

    z_plane1 = compute_z_on_plane(ransac1, rays_xy)

    if has_plane2:
        z_plane2 = compute_z_on_plane(ransac2, rays_xy)
        # 銳利直角的產生：取距離相機較近的面
        z_hole = np.minimum(z_plane1, z_plane2)
    else:
        z_hole = z_plane1

    scaffold_depth = raw_depth.copy()
    scaffold_depth[v_h, u_h] = z_hole

    return scaffold_depth

# ==============================================================================
# ⏳ 載入 2D 擴散模型 (Stable Diffusion Inpaint + ControlNet Depth)
# ==============================================================================
try:
    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
    print("⏳ 正在載入 ControlNet-Depth 聯合修補模型至 GPU (這可能需要幾分鐘下載權重)...")
    
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", 
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        controlnet=controlnet, 
        torch_dtype=torch.float16,
        use_safetensors=True, 
        variant="fp16"        
    ).to("cuda")
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload() 
    print("✅ 幾何條件擴散模型 (Diffusion) 載入完成！")
except ImportError:
    print("⚠️ 尚未安裝 diffusers，請執行: pip install diffusers accelerate transformers")
    pipe = None



try:
    from simple_lama_inpainting import SimpleLama
    print("⏳ 正在載入 LaMa 模型至 GPU...")
    lama_model = SimpleLama()
    print("✅ LaMa 模型載入完成！")
except ImportError:
    print("⚠️ 尚未安裝 simple-lama-inpainting，請執行 pip install simple-lama-inpainting")
    lama_model = None

# ==============================================================================
# [模組 2] 🎨 幾何條件約束的紋理生成 (Depth-Conditioned Texture Hallucination)
# ==============================================================================
def run_diffusion_texture_generation(img_path, mask_2d, scaffold_depth,
                                     prev_inpainted_bgr=None,
                                     first_ref_depth=None):
    """
    prev_inpainted_bgr: 第一個 ref view 的 inpainted 結果（Visual Prompting）。
                        提供時 SD 繼承前人確立的紋理風格，確保多視角一致性。
    """
    if pipe is None:
        raise RuntimeError("需要 diffusers 套件來執行紋理生成！")

    # ── mask ───────────────────────────────────────────────────────
    mask_2d_255  = (mask_2d > 0).astype(np.uint8) * 255
    kernel       = np.ones((9, 9), np.uint8)
    mask_dilated = cv2.dilate(mask_2d_255, kernel, iterations=1)
    mask_blurred = cv2.GaussianBlur(mask_dilated, (15, 15), 0)
    mask_pil     = Image.fromarray(mask_blurred).convert('L')

    # ── base image ─────────────────────────────────────────────────
    if prev_inpainted_bgr is not None:
        img_np = cv2.cvtColor(prev_inpainted_bgr, cv2.COLOR_BGR2RGB)
        H, W   = img_np.shape[:2]
        print("    ♻️  [Visual Prompting] 繼承第一個 ref view 的紋理風格")
    else:
        img_np = np.array(Image.open(img_path).convert('RGB'))
        H, W   = img_np.shape[:2]
        img_np[mask_dilated > 0] = [127, 127, 127]
    img_pil = Image.fromarray(img_np)

    # ── depth control ───────────────────────────────────────────────
    valid_depth = scaffold_depth[scaffold_depth > 0]
    if len(valid_depth) == 0:
        valid_depth = np.array([1.0])
    min_d, max_d = valid_depth.min(), valid_depth.max()
    depth_norm   = 255.0 * (max_d - scaffold_depth) / (max_d - min_d + 1e-5)


    # 若提供 first_ref_depth，與當前 depth 各 50% 混合送 ControlNet
    if first_ref_depth is not None:
        def norm01(d):
            v = d[(d > 0) & ~np.isnan(d)]
            if len(v) == 0: return np.zeros_like(d, dtype=np.float32)
            return np.clip((d - v.min()) / (v.max() - v.min() + 1e-8), 0, 1).astype(np.float32)
        cur_n   = norm01(scaffold_depth)
        first_n = norm01(first_ref_depth)
        if first_n.shape != cur_n.shape:
            first_n = cv2.resize(first_n, (cur_n.shape[1], cur_n.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        depth_norm = ((0.5 * cur_n + 0.5 * first_n) * 255).astype(np.uint8)
        print("    🔀 [Depth Blend] 與第一個 ref depth 50/50 混合送 ControlNet")
    # （else 維持原本的 depth_norm 計算不動）

    control_image = Image.fromarray(
        cv2.cvtColor(np.clip(depth_norm, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    )

    # SD 輸入對齊 8 的倍數
    sd_w = (W // 8) * 8
    sd_h = (H // 8) * 8
    img_sd     = img_pil.resize((sd_w, sd_h), Image.LANCZOS)
    mask_sd    = mask_pil.resize((sd_w, sd_h), Image.NEAREST)
    control_sd = control_image.resize((sd_w, sd_h), Image.LANCZOS)

    prompt = (
        "seamless empty background, perfectly matching surrounding textures, "
        "continuous texture"
        "continuous surface, clean and uncluttered space, empty scenery, "
        "background only, highly detailed, photorealistic"
    )
    negative_prompt = (
        "cardboard box, suitcase, luggage, bag, backpack, box, crate, "
        "bin, trash can, garbage can, transparent object, glass, ghost, "
        "semi-transparent, translucent, blurry, artifacts, unnatural boundaries, "
        "distorted geometry, low quality, floating, duplicate, extra object"
    )

    # scale=0.25 (low)：scaffold depth 仍有誤差，low scale 讓 SD 自由發揮 2D prior
    # seed=42：所有 ref view 共用相同 initial noise → 紋理風格一致，無 ghosting



    generator = torch.Generator(device="cuda").manual_seed(42)
    print("    🎨 [單次生成] scale=0.25, seed=42...")
    result_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img_sd,
        mask_image=mask_sd,
        control_image=control_sd,
        height=sd_h,
        width=sd_w,
        num_inference_steps=25,
        controlnet_conditioning_scale=0.4,
        guidance_scale=8.5,
        generator=generator,
    ).images[0].resize((W, H), Image.LANCZOS)

    return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

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
        img = cv2.imread(img_path)
        inpainted_rgb = cv2.inpaint(img, mask_dilated, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        
    return inpainted_rgb


# ==============================================================================
# [模組 3] 🚀 多參考視角前向融合與動態評估 (Geometry-First 終極版)
# ==============================================================================
def generative_multi_ref_propagation(
    ref_indices, target_idx, image_paths, mask_dir, 
    raw_depth_maps, all_cam_to_world_mat, intrinsics, 
    output_dir, ref_cache
):
    print(f"\n[Gen-3D Prop] 啟動幾何優先多視角融合: Target V_{target_idx} <- Refs {ref_indices}", end="\r")

    target_img_path = image_paths[target_idx]
    target_img = cv2.imread(target_img_path)
    H, W = target_img.shape[:2]
    
    target_mask_path = os.path.join(mask_dir, os.path.basename(target_img_path))
    mask_tgt = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
    mask_tgt = cv2.resize(mask_tgt, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_tgt = (mask_tgt > 0).astype(np.uint8) * 255 # 🟢 確保 Target Mask 是 255

    w2c_tgt = all_cam_to_world_mat[target_idx]
    c2w_tgt = np.linalg.inv(w2c_tgt) 
    
    depth_vt_lowres = raw_depth_maps[target_idx]
    scale_x, scale_y = W / depth_vt_lowres.shape[1], H / depth_vt_lowres.shape[0]
    K_tgt = intrinsics[target_idx].copy()
    K_tgt[0, :] *= scale_x
    K_tgt[1, :] *= scale_y

    final_canvas = target_img.copy()
    remaining_hole_mask = (mask_tgt > 0).copy()
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    kernel_expand_tgt = np.ones((25, 25), np.uint8) 
    expanded_mask_tgt = cv2.dilate(mask_tgt, kernel_expand_tgt, iterations=1)

    boundary_mses = []
    first_ref_bgr = None   # 第 1 個 ref view 結果，供後續 ref 繼承
    first_ref_depth = None

    target_pos = c2w_tgt[:3, 3]
    ref_distances = []
    
    for r_idx in ref_indices:
        r_c2w = np.linalg.inv(all_cam_to_world_mat[r_idx])
        dist = np.linalg.norm(r_c2w[:3, 3] - target_pos)
        ref_distances.append((r_idx, dist))
        
    sorted_ref_indices = [x[0] for x in sorted(ref_distances, key=lambda x: x[1])]

    for ref_idx in sorted_ref_indices:
        if not np.any(remaining_hole_mask):
            print(f"🎉 Target V_{target_idx} 的死角已被完美填滿！", end="\r")
            break

        print(f"  -> 正在從 Ref V_{ref_idx} 擷取並映射 3D 補丁...", end="\r")

        ref_img_path = image_paths[ref_idx]
        ref_mask_path = os.path.join(mask_dir, os.path.basename(ref_img_path))
        mask_ref = cv2.resize(cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_ref = (mask_ref > 0).astype(np.uint8) * 255 # 🟢 確保 Ref Mask 是 255
        
        if ref_idx not in ref_cache:
            vggt_raw_depth = cv2.resize(raw_depth_maps[ref_idx], (W, H), interpolation=cv2.INTER_NEAREST)
            
            K_ref = intrinsics[ref_idx].copy()
            K_ref[0, :] *= scale_x
            K_ref[1, :] *= scale_y

            scaffold_depth = extrapolate_3d_geometry(vggt_raw_depth, mask_ref, K_ref)


                # mask 區域內的 depth 值
            mask_depths = scaffold_depth[mask_ref > 0]
            mask_depths = mask_depths[~np.isnan(mask_depths) & (mask_depths > 0)]

            if len(mask_depths) > 10:
                depth_range = mask_depths.max() - mask_depths.min()
                depth_std   = np.std(mask_depths)
                relative_spread = depth_std / (depth_range + 1e-8)

                # 額外檢查：小平面佔比必須超過 20%，才算真正跨兩平面
                # 用 median 切兩組，取小的那組的佔比
                median_d   = np.median(mask_depths)
                n_lower    = (mask_depths < median_d).sum()
                n_upper    = (mask_depths >= median_d).sum()
                minor_ratio = min(n_lower, n_upper) / len(mask_depths)

                mask_spans_two_planes = (
                    has_plane2
                    and (relative_spread > 0.2)
                    and (minor_ratio > 0.3)    # 小平面至少佔 20%
                )
            else:
                mask_spans_two_planes = False

            print(f"    📊 relative_spread={relative_spread:.3f}  minor_ratio={minor_ratio:.3f}  spans_two={mask_spans_two_planes}")

            if mask_spans_two_planes:

                # 第 1 個 ref → 獨立生成建立風格基準；第 2、3 個 ref → 繼承第 1 個的結果
                print("    🖌️ [Diffusion] 跨雙平面，使用 ControlNet-Depth 條件擴散生成紋理...")
                inpainted_rgb_ref = run_diffusion_texture_generation(
                    ref_img_path, mask_ref, scaffold_depth,
                    prev_inpainted_bgr=first_ref_bgr,
                    first_ref_depth=first_ref_depth
                )

            else:
                print("    🖌️ [LaMa] 單平面，使用 LaMa 穩定填補...")
                inpainted_rgb_ref = run_generative_rgb_inpaint(ref_img_path, mask_ref)




            if first_ref_bgr is None:
                first_ref_bgr = inpainted_rgb_ref.copy()
                first_ref_depth = scaffold_depth.copy()
                print(f"    📌 V_{ref_idx} 建立風格基準，後續 ref 將繼承")

            ref_cache[ref_idx] = (inpainted_rgb_ref, scaffold_depth)
            
        inpainted_rgb_ref, scaffold_depth = ref_cache[ref_idx]

        kernel_expand_ref = np.ones((25, 25), np.uint8) 
        expanded_mask_ref = cv2.dilate(mask_ref, kernel_expand_ref, iterations=1)
        v_ref, u_ref = np.where(expanded_mask_ref > 0)
        
        Z_ref = scaffold_depth[v_ref, u_ref]
        valid_z = ~np.isnan(Z_ref) & ~np.isinf(Z_ref) & (Z_ref > 0)
        u_ref, v_ref, Z_ref = u_ref[valid_z], v_ref[valid_z], Z_ref[valid_z]
        colors_ref = inpainted_rgb_ref[v_ref, u_ref]

        c2w_ref = np.linalg.inv(all_cam_to_world_mat[ref_idx])
        K_ref = intrinsics[ref_idx].copy()
        K_ref[0, :] *= scale_x
        K_ref[1, :] *= scale_y

        X_cam = (u_ref - K_ref[0, 2]) * Z_ref / K_ref[0, 0]
        Y_cam = (v_ref - K_ref[1, 2]) * Z_ref / K_ref[1, 1]
        pts_cam = np.column_stack((X_cam, Y_cam, Z_ref))
        pts_cam_homo = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
        pts_world = (c2w_ref @ pts_cam_homo.T).T[:, :3]

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

        warp_canvas = np.zeros_like(target_img)
        valid_warp_mask = np.zeros((H, W), dtype=np.uint8)
        
        for v, u, color in zip(v_tgt, u_tgt, final_colors):
            cv2.circle(warp_canvas, (u, v), 3, color.tolist(), -1)
            cv2.circle(valid_warp_mask, (u, v), 3, 255, -1)
            
        valid_warp_mask_smoothed = cv2.morphologyEx(valid_warp_mask, cv2.MORPH_OPEN, morph_kernel)
        valid_warp_mask_bool = valid_warp_mask_smoothed > 0

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

        M_best = np.float32([[1, 0, best_dx], [0, 1, best_dy]])
        shifted_canvas = cv2.warpAffine(warp_canvas, M_best, (W, H))
        shifted_mask = cv2.warpAffine(valid_warp_mask_smoothed, M_best, (W, H))
        
        current_eval_ring = (expanded_mask_tgt > 0) & (mask_tgt == 0) & (shifted_mask > 0)
        
        if np.any(current_eval_ring):
            diff = shifted_canvas[current_eval_ring].astype(np.float32) - target_img[current_eval_ring].astype(np.float32)
            patch_mse = float(np.mean(np.square(diff)))
            boundary_mses.append(patch_mse)

        paste_mask = (shifted_mask > 0) & remaining_hole_mask
        paste_y, paste_x = np.where(paste_mask)
        
        final_canvas[paste_y, paste_x] = shifted_canvas[paste_y, paste_x]
        remaining_hole_mask[paste_y, paste_x] = False
        
    red_mask_smoothed = cv2.morphologyEx((remaining_hole_mask * 255).astype(np.uint8), cv2.MORPH_OPEN, morph_kernel)
    red_y, red_x = np.where(red_mask_smoothed > 0)
    final_canvas[red_y, red_x] = [0, 0, 255]
    red_area = len(red_y)
    
    final_boundary_mse = max(boundary_mses) if boundary_mses else 0.0
    
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(str(output_dir / f"inpainted_{target_idx}.png"), final_canvas)
    
    # print(f"✅ V_{target_idx} 處理完成！")
    # print(f"   - 剩餘紅色死角面積: {red_area} 像素")
    # print(f"   - 最高邊界紋理撕裂誤差 (MSE): {final_boundary_mse:.2f}\n")
    
    return red_area, final_boundary_mse