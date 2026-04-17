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
    Per-plane Nearest Neighbor 深度傳播：
      取代 RANSAC，直接用 VGGT 已預測的 valid depth 值做幾何填充。

    策略：
      1. 對 valid depth 做 Laplacian → 找平面邊界（高梯度 = 幾何斷層）
      2. Connected Components 切分平面 region
      3. 每個 NaN pixel 向所有 plane region 找最近鄰 valid depth
         → 取 minimum（近的面優先，自然產生銳利直角而不是斜坡）
      4. 真正死角（所有 plane 都太遠）→ OpenCV inpaint 保底

    不使用 intrinsics K（保留參數以維持呼叫端相容性）。
    """
    H, W = mask_2d.shape
    mask_bool = (mask_2d > 0)

    # ── Step 1: 找 valid depth（VGGT 在 mask 內已同化的背景幾何）───
    valid_bool = (~np.isnan(raw_depth)) & (raw_depth > 0)
    valid_in_mask    = valid_bool &  mask_bool   # mask 內的有效點
    valid_outside    = valid_bool & ~mask_bool   # mask 外（真實背景）

    n_valid_in = valid_in_mask.sum()
    print(f"    🔍 [VGGT Depth] mask 內 valid pixels: {n_valid_in} / {mask_bool.sum()}")

    # ── Step 2: Laplacian 找幾何斷層線 ──────────────────────────────
    depth_clean = raw_depth.copy()
    depth_clean[np.isnan(depth_clean)] = 0.0
    laplacian  = cv2.Laplacian(depth_clean.astype(np.float32), cv2.CV_32F)
    disc_abs   = np.abs(laplacian)

    # 自適應 threshold：取 valid 區域梯度的 75th percentile
    valid_grad = disc_abs[valid_bool]
    disc_thresh = np.percentile(valid_grad, 75) if len(valid_grad) > 0 else 0.01
    boundary_map = (disc_abs > disc_thresh).astype(np.uint8)
    print(f"    🔍 [Disc] threshold={disc_thresh:.4f}")

    # ── Step 3: Connected Components 切分平面 region ────────────────
    # valid & 非斷層 → 獨立平面內部
    plane_seed = (valid_bool & (boundary_map == 0)).astype(np.uint8)
    num_labels, label_map = cv2.connectedComponents(plane_seed)
    n_planes = num_labels - 1   # 扣掉背景 label 0
    print(f"    🔍 [Planes] detected {n_planes} plane regions")

    # ── Step 4: Per-plane Nearest Neighbor 填充 NaN ─────────────────
    nan_in_mask = mask_bool & (~valid_bool)   # 需要填充的 NaN 像素

    if nan_in_mask.sum() == 0:
        print("    ✅ 無 NaN 需填充，直接使用 VGGT depth")
        return raw_depth.copy()

    if n_planes == 0:
        # 完全沒有 valid 點，用 mask 外的最近鄰保底
        print("    ⚠️ 無 valid plane，使用 mask 外鄰居填充")
        scaffold = raw_depth.copy()
        scaffold_u8 = np.zeros((H, W), dtype=np.uint8)
        inpaint_mask = nan_in_mask.astype(np.uint8) * 255
        # 用外部 valid depth 的 mean 填充（最後保底）
        mean_d = raw_depth[valid_outside].mean() if valid_outside.sum() > 0 else 1.0
        scaffold[nan_in_mask] = mean_d
        return scaffold

    # 對每個 NaN pixel 計算各平面的最近距離 → 取 minimum depth
    nan_v, nan_u = np.where(nan_in_mask)
    nan_coords   = np.column_stack((nan_v, nan_u)).astype(np.float32)   # (N, 2)

    scaffold = raw_depth.copy()
    candidate_depths = np.full(len(nan_coords), np.inf)   # 記錄每個 NaN 的候選深度

    DIST_THRESHOLD = min(H, W) * 0.3   # 超過此距離的 plane 視為太遠，不採用

    for label_id in range(1, num_labels):
        plane_mask = (label_map == label_id)
        if plane_mask.sum() < 5:
            continue   # 忽略太小的 noise region

        plane_v, plane_u = np.where(plane_mask)
        plane_coords = np.column_stack((plane_v, plane_u)).astype(np.float32)  # (M, 2)
        plane_depths = raw_depth[plane_v, plane_u]   # (M,) 這個 plane 的 depth 值

        # 對每個 NaN 找此 plane 的最近鄰距離和對應 depth
        # 用 KD-tree 加速（scipy）
        from scipy.spatial import cKDTree
        tree = cKDTree(plane_coords)
        dists, idxs = tree.query(nan_coords, k=1)

        # 只採用距離夠近的 plane
        close_enough = dists < DIST_THRESHOLD
        nn_depth     = plane_depths[idxs]   # 最近鄰的 depth 值

        # minimum depth：哪個 plane 離得近且深度最小，就用那個
        # （近的面遮擋遠的面，自然形成直角而非斜坡）
        update_mask = close_enough & (nn_depth < candidate_depths)
        candidate_depths[update_mask] = nn_depth[update_mask]

    # 把有效候選 depth 填入
    has_candidate = candidate_depths < np.inf
    scaffold[nan_v[has_candidate], nan_u[has_candidate]] = candidate_depths[has_candidate]

    n_filled    = has_candidate.sum()
    n_remaining = (~has_candidate).sum()
    print(f"    ✅ Per-plane NN filled: {n_filled}  remaining NaN (true dead zone): {n_remaining}")

    # ── Step 5: 真正死角保底 → OpenCV inpaint ───────────────────────
    if n_remaining > 0:
        still_nan = nan_in_mask.copy()
        still_nan[nan_v[has_candidate], nan_u[has_candidate]] = False

        # opencv inpaint 需要 uint8，先線性 normalize
        valid_vals = scaffold[valid_bool | has_candidate]
        d_min, d_max = valid_vals.min(), valid_vals.max()
        depth_u8 = np.clip(
            255.0 * (scaffold - d_min) / (d_max - d_min + 1e-8), 0, 255
        ).astype(np.uint8)
        inpaint_mask_u8 = still_nan.astype(np.uint8) * 255
        filled_u8 = cv2.inpaint(depth_u8, inpaint_mask_u8, inpaintRadius=5,
                                 flags=cv2.INPAINT_TELEA)
        # 反 normalize 回 depth
        filled_depth = filled_u8.astype(np.float32) / 255.0 * (d_max - d_min) + d_min
        scaffold[still_nan] = filled_depth[still_nan]
        print(f"    🛟 OpenCV inpaint 保底填充 {n_remaining} dead-zone pixels")

    # ── Debug: 儲存填充後的深度圖 ────────────────────────────────────
    _dbg_dir = "debug_inpaint"
    os.makedirs(_dbg_dir, exist_ok=True)
    import time, hashlib
    _tag = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    d_min2, d_max2 = scaffold[~np.isnan(scaffold)].min(), scaffold[~np.isnan(scaffold)].max()
    scaffold_vis = np.clip(
        255.0 * (d_max2 - scaffold) / (d_max2 - d_min2 + 1e-8), 0, 255
    ).astype(np.uint8)
    cv2.imwrite(os.path.join(_dbg_dir, f"{_tag}_scaffold_new.png"), scaffold_vis)
    print(f"    [DEBUG] New scaffold depth → {_dbg_dir}/{_tag}_scaffold_new.png")

    return scaffold


# ==============================================================================
# [NEW] 📐 幾何訊號圖建立 (Geometry Signal Maps)
#
# 產生兩張 per-pixel 訊號圖，用於後續 adaptive blending：
#   density_map  : VGGT point 密度（高 = 有可信 3D 背景先驗）
#   disc_map     : 深度不連續性（高 = 平面邊界，直角需要保留）
# ==============================================================================
def build_geometry_signal_maps(scaffold_depth, raw_depth_lowres, mask_2d, H, W):
    """
    Parameters
    ----------
    scaffold_depth   : (H, W) float, RANSAC 外推後的 depth
    raw_depth_lowres : (h, w) float, VGGT 原始低解析度 depth（可含 NaN）
    mask_2d          : (H, W) uint8, 0/255
    H, W             : 目標解析度

    Returns
    -------
    blend_alpha : (H, W) float in [0, 1]
        0 = 完全信任 2D diffusion 結果
        1 = 完全信任 3D-constrained 結果
    """
    mask_bool = mask_2d > 0

    # ── Signal 1: VGGT point density ──────────────────────────────────
    # valid pixel in raw_depth = VGGT 有信心的 3D 點（含 mask 內被同化的背景）
    d_resized = cv2.resize(
        raw_depth_lowres.astype(np.float32), (W, H),
        interpolation=cv2.INTER_NEAREST
    )
    valid_pts = (~np.isnan(d_resized)) & (d_resized > 0)
    # Gaussian blur → density field
    density_map = cv2.GaussianBlur(
        valid_pts.astype(np.float32), (31, 31), 0
    )
    density_norm = density_map / (density_map.max() + 1e-8)

    # ── Signal 2: depth discontinuity from scaffold ────────────────────
    depth_clean = scaffold_depth.copy()
    depth_clean[np.isnan(depth_clean)] = 0
    laplacian  = cv2.Laplacian(depth_clean.astype(np.float32), cv2.CV_32F)
    disc_map   = np.abs(laplacian)
    # 平滑一點，避免單像素噪聲影響 blending
    disc_map   = cv2.GaussianBlur(disc_map, (9, 9), 0)
    disc_norm  = disc_map / (disc_map.max() + 1e-8)

    # ── Combine ────────────────────────────────────────────────────────
    # 高密度 OR 高不連續 → alpha 高 → 多信任 3D
    alpha_raw = 0.5 * density_norm + 0.5 * disc_norm
    alpha_raw = np.clip(alpha_raw, 0.0, 1.0)

    # 只在 mask 內有效；mask 外固定給中間值（不影響結果）
    blend_alpha = np.full((H, W), 0.5, dtype=np.float32)
    blend_alpha[mask_bool] = alpha_raw[mask_bool]

    # 邊界過渡帶用 distance transform 做 feathering，避免硬邊
    dist = cv2.distanceTransform(
        mask_2d.astype(np.uint8), cv2.DIST_L2, 5
    )
    max_dist = dist.max() if dist.max() > 0 else 1.0
    feather = np.clip(dist / (max_dist * 0.3), 0.0, 1.0)   # 邊緣 30% 做過渡
    blend_alpha = blend_alpha * feather + 0.5 * (1 - feather)

    print(f"    📐 [Adaptive] density_mean={density_norm[mask_bool].mean():.3f}  "
          f"disc_mean={disc_norm[mask_bool].mean():.3f}  "
          f"alpha_mean={blend_alpha[mask_bool].mean():.3f}")

    return blend_alpha.astype(np.float32)


# ==============================================================================
# ⏳ 載入 2D 擴散模型 (Stable Diffusion Inpaint + ControlNet Depth)
# ==============================================================================
try:
    from diffusers import (StableDiffusionControlNetInpaintPipeline,
                           ControlNetModel, UniPCMultistepScheduler)
    print("⏳ 正在載入 ControlNet-Depth 聯合修補模型至 GPU...")

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


# ==============================================================================
# [模組 2] 🎨 Adaptive 幾何條件約束紋理生成
#
# 改動說明：
#   新增 raw_depth_lowres 參數，用來建立 geometry signal maps。
#   跑「兩次」 diffusion（scale_low / scale_high），
#   再依 blend_alpha 做 per-pixel weighted blend：
#     alpha=0 → 信任 2D（用 scale_low 結果，適合草叢等自由紋理）
#     alpha=1 → 信任 3D（用 scale_high 結果，適合平面邊界直角）
# ==============================================================================
def run_diffusion_texture_generation(img_path, mask_2d, scaffold_depth,
                                     raw_depth_lowres=None):
    """
    Parameters
    ----------
    img_path          : str, 參考影像路徑
    mask_2d           : (H, W) uint8, 0/255
    scaffold_depth    : (H, W) float, RANSAC 外推的幾何鷹架深度
    raw_depth_lowres  : (h, w) float or None
                        VGGT 原始低解析度深度（含 NaN）
                        None → 退化成舊行為（單一固定 scale）
    """
    if pipe is None:
        raise RuntimeError("需要 diffusers 套件來執行紋理生成！")

    # ── 準備輸入影像與遮罩 ──────────────────────────────────────────
    img_pil = Image.open(img_path).convert('RGB')
    img_np  = np.array(img_pil)
    H, W    = img_np.shape[:2]

    # 防線一：先算 dilated mask，再用它塗銷（比原始 mask 更大）
    # 關鍵：SD 的 inpaint 區域是 mask_dilated，所以灰色必須覆蓋整個 dilated 範圍
    # 否則膨脹出去的那一圈會露出原始物體邊緣，SD 會把物體「補完」成幽靈
    mask_2d_255  = (mask_2d > 0).astype(np.uint8) * 255
    kernel       = np.ones((9, 9), np.uint8)
    mask_dilated = cv2.dilate(mask_2d_255, kernel, iterations=1)
    mask_blurred = cv2.GaussianBlur(mask_dilated, (15, 15), 0)
    mask_pil     = Image.fromarray(mask_blurred).convert('L')

    # 防線二：塗銷整個 dilated 區域（不只是原始 mask）
    img_np[mask_dilated > 0] = [127, 127, 127]
    img_pil = Image.fromarray(img_np)

    # ── ControlNet 深度圖（MiDaS 視差格式）────────────────────────
    valid_depth = scaffold_depth[scaffold_depth > 0]
    min_d, max_d = valid_depth.min(), valid_depth.max()
    depth_norm   = 255.0 * (max_d - scaffold_depth) / (max_d - min_d + 1e-5)
    depth_norm   = np.clip(depth_norm, 0, 255).astype(np.uint8)
    control_image = Image.fromarray(cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB))

    # ── [DEBUG] 儲存中間結果供診斷 ────────────────────────────────
    import tempfile, hashlib, time
    _dbg_tag = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    _dbg_dir = "debug_inpaint"
    os.makedirs(_dbg_dir, exist_ok=True)

    # Debug 1: SD 實際看到的輸入圖（灰色填充後）
    img_pil.save(os.path.join(_dbg_dir, f"{_dbg_tag}_1_sd_input.png"))
    print(f"    [DEBUG] SD input saved → {_dbg_dir}/{_dbg_tag}_1_sd_input.png")

    # Debug 2: Dilated mask（SD 的 inpaint 範圍）
    mask_pil.save(os.path.join(_dbg_dir, f"{_dbg_tag}_2_mask_dilated.png"))
    print(f"    [DEBUG] Mask saved     → {_dbg_dir}/{_dbg_tag}_2_mask_dilated.png")

    # Debug 3: ControlNet 深度圖（確認 RANSAC scaffold 形狀）
    control_image.save(os.path.join(_dbg_dir, f"{_dbg_tag}_3_depth_control.png"))
    print(f"    [DEBUG] Depth saved    → {_dbg_dir}/{_dbg_tag}_3_depth_control.png")
    # ────────────────────────────────────────────────────────────

    # SD 輸入尺寸對齊 8 的倍數
    sd_w = (W // 8) * 8
    sd_h = (H // 8) * 8
    img_sd     = img_pil.resize((sd_w, sd_h), Image.LANCZOS)
    mask_sd    = mask_pil.resize((sd_w, sd_h), Image.NEAREST)
    control_sd = control_image.resize((sd_w, sd_h), Image.LANCZOS)

    prompt = (
        "realistic background, seamless surface texture, "
        "consistent lighting, photorealistic"
    )
    # 明確列出場景中可能出現的物體類型，防止 SD 幻覺補回原始物體
    negative_prompt = (
        "cardboard box, suitcase, luggage, bag, backpack, box, crate, "
        "bin, trash can, garbage can, transparent object, glass, ghost, "
        "semi-transparent, translucent, blurry, artifacts, unnatural boundaries, "
        "distorted geometry, low quality, floating, duplicate, extra object"
    )

    # ── 判斷是否啟用 Adaptive 雙次生成 ───────────────────────────
    use_adaptive = (raw_depth_lowres is not None)

    if use_adaptive:
        # ── 建立 blend_alpha map ──────────────────────────────────
        blend_alpha = build_geometry_signal_maps(
            scaffold_depth, raw_depth_lowres, mask_2d, H, W
        )

        SCALE_LOW  = 0.2   # 2D 自由區（草叢、無明確 3D prior 的死角）
        SCALE_HIGH = 0.65  # 3D 約束區（平面邊界、密集 point 區）

        # Pass 1：低 scale，2D 充分發揮（草叢立體感）
        print("    🎨 [Pass 1/2] scale_low=0.2 (2D freedom)...")
        result_low = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            image=img_sd, mask_image=mask_sd, control_image=control_sd,
            height=sd_h, width=sd_w,
            num_inference_steps=25,
            controlnet_conditioning_scale=SCALE_LOW,
            guidance_scale=7.5
        ).images[0].resize((W, H), Image.LANCZOS)
        # Debug 4: Pass1 原始輸出
        result_low.save(os.path.join(_dbg_dir, f"{_dbg_tag}_4_pass1_low.png"))
        print(f"    [DEBUG] Pass1 saved    → {_dbg_dir}/{_dbg_tag}_4_pass1_low.png")

        # Pass 2：高 scale，3D 結構強制約束（直角、邊界）
        print("    🎨 [Pass 2/2] scale_high=0.65 (3D constraint)...")
        result_high = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            image=img_sd, mask_image=mask_sd, control_image=control_sd,
            height=sd_h, width=sd_w,
            num_inference_steps=25,
            controlnet_conditioning_scale=SCALE_HIGH,
            guidance_scale=7.5
        ).images[0].resize((W, H), Image.LANCZOS)
        # Debug 5: Pass2 原始輸出
        result_high.save(os.path.join(_dbg_dir, f"{_dbg_tag}_5_pass2_high.png"))
        print(f"    [DEBUG] Pass2 saved    → {_dbg_dir}/{_dbg_tag}_5_pass2_high.png")

        # ── Per-pixel Weighted Blend ──────────────────────────────
        low_np  = np.array(result_low,  dtype=np.float32)   # (H, W, 3)
        high_np = np.array(result_high, dtype=np.float32)   # (H, W, 3)
        alpha   = blend_alpha[:, :, np.newaxis]              # (H, W, 1)

        blended_np = (1.0 - alpha) * low_np + alpha * high_np
        blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)

        result_bgr = cv2.cvtColor(blended_np, cv2.COLOR_RGB2BGR)

        # Debug 6: blend alpha map 視覺化（越亮=越信任3D）
        alpha_vis = (blend_alpha * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(_dbg_dir, f"{_dbg_tag}_6_blend_alpha.png"), alpha_vis)
        # Debug 7: 最終 blended 結果
        Image.fromarray(blended_np).save(os.path.join(_dbg_dir, f"{_dbg_tag}_7_blended.png"))
        print(f"    [DEBUG] Alpha map saved → {_dbg_dir}/{_dbg_tag}_6_blend_alpha.png")
        print(f"    [DEBUG] Blended saved   → {_dbg_dir}/{_dbg_tag}_7_blended.png")
        print("    ✅ Adaptive blend 完成。")

    else:
        # ── 退化模式：舊行為，單一固定 scale ─────────────────────
        print("    🎨 [單次生成] scale=0.3 (legacy mode)...")
        result_pil = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            image=img_sd, mask_image=mask_sd, control_image=control_sd,
            height=sd_h, width=sd_w,
            num_inference_steps=25,
            controlnet_conditioning_scale=0.3,
            guidance_scale=7.5
        ).images[0].resize((W, H), Image.LANCZOS)
        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    return result_bgr


# ==============================================================================
# [模組 3] 🚀 多參考視角前向融合與動態評估
# ==============================================================================
def generative_multi_ref_propagation(
    ref_indices, target_idx, image_paths, mask_dir,
    raw_depth_maps, all_cam_to_world_mat, intrinsics,
    output_dir, ref_cache
):
    print(f"\n[Gen-3D Prop] 啟動幾何優先多視角融合: Target V_{target_idx} <- Refs {ref_indices}")

    target_img_path = image_paths[target_idx]
    target_img = cv2.imread(target_img_path)
    H, W = target_img.shape[:2]

    target_mask_path = os.path.join(mask_dir, os.path.basename(target_img_path))
    mask_tgt = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
    mask_tgt = cv2.resize(mask_tgt, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_tgt = (mask_tgt > 0).astype(np.uint8) * 255

    w2c_tgt = all_cam_to_world_mat[target_idx]
    c2w_tgt = np.linalg.inv(w2c_tgt)

    depth_vt_lowres = raw_depth_maps[target_idx]
    scale_x = W / depth_vt_lowres.shape[1]
    scale_y = H / depth_vt_lowres.shape[0]
    K_tgt = intrinsics[target_idx].copy()
    K_tgt[0, :] *= scale_x
    K_tgt[1, :] *= scale_y

    final_canvas       = target_img.copy()
    remaining_hole_mask = (mask_tgt > 0).copy()
    morph_kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    kernel_expand_tgt  = np.ones((25, 25), np.uint8)
    expanded_mask_tgt  = cv2.dilate(mask_tgt, kernel_expand_tgt, iterations=1)

    boundary_mses = []

    target_pos    = c2w_tgt[:3, 3]
    ref_distances = []
    for r_idx in ref_indices:
        r_c2w = np.linalg.inv(all_cam_to_world_mat[r_idx])
        dist  = np.linalg.norm(r_c2w[:3, 3] - target_pos)
        ref_distances.append((r_idx, dist))
    sorted_ref_indices = [x[0] for x in sorted(ref_distances, key=lambda x: x[1])]

    for ref_idx in sorted_ref_indices:
        if not np.any(remaining_hole_mask):
            print(f"🎉 Target V_{target_idx} 的死角已被完美填滿！")
            break

        print(f"  -> 正在從 Ref V_{ref_idx} 擷取並映射 3D 補丁...")

        ref_img_path  = image_paths[ref_idx]
        ref_mask_path = os.path.join(mask_dir, os.path.basename(ref_img_path))
        mask_ref = cv2.resize(
            cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE), (W, H),
            interpolation=cv2.INTER_NEAREST
        )
        mask_ref = (mask_ref > 0).astype(np.uint8) * 255

        if ref_idx not in ref_cache:
            vggt_raw_depth = cv2.resize(
                raw_depth_maps[ref_idx], (W, H),
                interpolation=cv2.INTER_NEAREST
            )

            K_ref = intrinsics[ref_idx].copy()
            K_ref[0, :] *= scale_x
            K_ref[1, :] *= scale_y

            scaffold_depth = extrapolate_3d_geometry(vggt_raw_depth, mask_ref, K_ref)

            # ── [改動] 傳入 raw_depth_lowres 啟動 Adaptive 雙次生成 ──
            inpainted_rgb_ref = run_diffusion_texture_generation(
                ref_img_path, mask_ref, scaffold_depth,
                raw_depth_lowres=raw_depth_maps[ref_idx]   # ← 新增
            )

            ref_cache[ref_idx] = (inpainted_rgb_ref, scaffold_depth)

        inpainted_rgb_ref, scaffold_depth = ref_cache[ref_idx]

        kernel_expand_ref = np.ones((25, 25), np.uint8)
        expanded_mask_ref = cv2.dilate(mask_ref, kernel_expand_ref, iterations=1)
        v_ref, u_ref = np.where(expanded_mask_ref > 0)

        Z_ref   = scaffold_depth[v_ref, u_ref]
        valid_z = ~np.isnan(Z_ref) & ~np.isinf(Z_ref) & (Z_ref > 0)
        u_ref, v_ref, Z_ref = u_ref[valid_z], v_ref[valid_z], Z_ref[valid_z]
        colors_ref = inpainted_rgb_ref[v_ref, u_ref]

        c2w_ref = np.linalg.inv(all_cam_to_world_mat[ref_idx])
        K_ref   = intrinsics[ref_idx].copy()
        K_ref[0, :] *= scale_x
        K_ref[1, :] *= scale_y

        X_cam = (u_ref - K_ref[0, 2]) * Z_ref / K_ref[0, 0]
        Y_cam = (v_ref - K_ref[1, 2]) * Z_ref / K_ref[1, 1]
        pts_cam      = np.column_stack((X_cam, Y_cam, Z_ref))
        pts_cam_homo = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
        pts_world    = (c2w_ref @ pts_cam_homo.T).T[:, :3]

        pts_world_homo = np.hstack((pts_world, np.ones((pts_world.shape[0], 1))))
        pts_tgt_cam    = (w2c_tgt @ pts_world_homo.T).T[:, :3]

        valid_z_tgt = pts_tgt_cam[:, 2] > 0.1
        pts_tgt_cam  = pts_tgt_cam[valid_z_tgt]
        final_colors = colors_ref[valid_z_tgt]

        pts_2d_tgt = (K_tgt @ pts_tgt_cam.T).T
        Z_tgt_proj = pts_tgt_cam[:, 2]
        u_tgt = (pts_2d_tgt[:, 0] / Z_tgt_proj).astype(np.int32)
        v_tgt = (pts_2d_tgt[:, 1] / Z_tgt_proj).astype(np.int32)

        valid_uv = (u_tgt >= 0) & (u_tgt < W) & (v_tgt >= 0) & (v_tgt < H)
        u_tgt, v_tgt, final_colors = u_tgt[valid_uv], v_tgt[valid_uv], final_colors[valid_uv]

        warp_canvas     = np.zeros_like(target_img)
        valid_warp_mask = np.zeros((H, W), dtype=np.uint8)

        for v, u, color in zip(v_tgt, u_tgt, final_colors):
            cv2.circle(warp_canvas,     (u, v), 3, color.tolist(), -1)
            cv2.circle(valid_warp_mask, (u, v), 3, 255,            -1)

        valid_warp_mask_smoothed = cv2.morphologyEx(valid_warp_mask, cv2.MORPH_OPEN, morph_kernel)
        valid_warp_mask_bool     = valid_warp_mask_smoothed > 0

        ring_mask = (expanded_mask_tgt > 0) & (mask_tgt == 0) & valid_warp_mask_bool
        best_dx, best_dy = 0, 0

        if np.any(ring_mask):
            y_idx, x_idx = np.where(ring_mask)
            y_min = max(0, y_idx.min()-15); y_max = min(H, y_idx.max()+15)
            x_min = max(0, x_idx.min()-15); x_max = min(W, x_idx.max()+15)

            tgt_crop  = target_img[y_min:y_max, x_min:x_max].astype(np.float32)
            warp_crop = warp_canvas[y_min:y_max, x_min:x_max].astype(np.float32)
            ring_crop = ring_mask[y_min:y_max, x_min:x_max]

            min_error = float('inf')
            for dy in range(-10, 11):
                for dx in range(-10, 11):
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    shifted_warp = cv2.warpAffine(warp_crop, M, (warp_crop.shape[1], warp_crop.shape[0]))
                    diff  = tgt_crop[ring_crop] - shifted_warp[ring_crop]
                    error = np.mean(np.square(diff))
                    if error < min_error:
                        min_error = error
                        best_dx, best_dy = dx, dy

        M_best         = np.float32([[1, 0, best_dx], [0, 1, best_dy]])
        shifted_canvas = cv2.warpAffine(warp_canvas,             M_best, (W, H))
        shifted_mask   = cv2.warpAffine(valid_warp_mask_smoothed, M_best, (W, H))

        current_eval_ring = (expanded_mask_tgt > 0) & (mask_tgt == 0) & (shifted_mask > 0)
        if np.any(current_eval_ring):
            diff      = (shifted_canvas[current_eval_ring].astype(np.float32)
                         - target_img[current_eval_ring].astype(np.float32))
            patch_mse = float(np.mean(np.square(diff)))
            boundary_mses.append(patch_mse)

        paste_mask = (shifted_mask > 0) & remaining_hole_mask
        paste_y, paste_x = np.where(paste_mask)
        final_canvas[paste_y, paste_x]    = shifted_canvas[paste_y, paste_x]
        remaining_hole_mask[paste_y, paste_x] = False

    red_mask_smoothed = cv2.morphologyEx(
        (remaining_hole_mask * 255).astype(np.uint8), cv2.MORPH_OPEN, morph_kernel
    )
    red_y, red_x = np.where(red_mask_smoothed > 0)
    final_canvas[red_y, red_x] = [0, 0, 255]
    red_area = len(red_y)

    final_boundary_mse = max(boundary_mses) if boundary_mses else 0.0

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(str(output_dir / f"inpainted_{target_idx}.png"), final_canvas)

    print(f"✅ V_{target_idx} 處理完成！")
    print(f"   - 剩餘紅色死角面積: {red_area} 像素")
    print(f"   - 最高邊界紋理撕裂誤差 (MSE): {final_boundary_mse:.2f}\n")

    return red_area, final_boundary_mse