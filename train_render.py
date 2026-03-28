import os
import torch
import random
import cv2
import math
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import save_image

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer.render import render
import lpips






import torch.nn.functional as F



def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    img1, img2: (C, H, W) 或 (B, C, H, W)，值域 [0, 1]
    回傳 scalar SSIM 值（越高越好）
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)  # → (1, C, H, W)
        img2 = img2.unsqueeze(0)

    B, C, H, W = img1.shape

    # 建 Gaussian kernel
    def gaussian_kernel(window_size, sigma=1.5):
        x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)  # (window, window)
        return kernel_2d

    kernel = gaussian_kernel(window_size).to(img1.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)
    kernel = kernel.expand(C, 1, window_size, window_size)  # (C, 1, W, W)

    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, kernel, padding=pad, groups=C) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean()


def ssim_loss(img1, img2):
    return 1.0 - ssim(img1, img2)





def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def get_expon_lr_func(lr_init, lr_final, max_steps):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0): return 0.0
        if step >= max_steps: return lr_final
        t = np.clip(step / max_steps, 0, 1)
        return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return helper

class DummyPipeline:
    def __init__(self):
        self.compute_cov3D_python = False
        self.convert_SHs_python = False
        self.debug = False

def train_and_render(args):
    print("🚀 啟動 3DGIC 幾何優化與渲染引擎 (嚴格 Zero Data Leakage 版)")
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = DummyPipeline()
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    
    # ==========================================
    # 🛡️ 嚴格資料分割：雙場景讀取機制
    # ==========================================
    # 1. 讀取 GT Test 相機 (嚴格從原始 GT COLMAP 提取，絕不參與幾何重建)
    print(f"📂 正在從 {args.test_colmap} 提取 Test 相機位姿...")
    dummy_gaussians = GaussianModel(sh_degree=3)
    gt_scene = Scene(args.test_colmap, dummy_gaussians, shuffle=False)
    all_gt_cams = gt_scene.getTrainCameras()
    all_gt_cams.sort(key=lambda x: x.image_name)
    test_cameras = all_gt_cams[:40] # 嚴格切出前 40 張 Test 視角
    
    # 釋放 Dummy 記憶體
    del dummy_gaussians
    del gt_scene

    # ==========================================
    # 🛡️ 嚴格資料分割：單一宇宙切分機制
    # ==========================================
    print(f"📂 正在從 {args.init_colmap} 載入幾何鷹架與相機位姿...")
    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(args.init_colmap, gaussians, shuffle=False)
    
    all_cams = scene.getTrainCameras()
    # 確保按照檔名排序 (前面是 40 張 GT，後面是 60 張 Inpainted)
    all_cams.sort(key=lambda x: x.image_name)
    
    # 💯 嚴格切分：前 40 張保留給 Test，後 60 張拿去 Train
    test_cameras = all_cams[:40]
    train_cameras = all_cams[40:]
    
    print(f"✅ 分割完成！Test 視角: {len(test_cameras)} 張, Train 視角: {len(train_cameras)} 張")


    # ==========================================
    # 🎯 記憶體劫持與訓練初始化
    # ==========================================
    purify_files = sorted([f for f in os.listdir(args.train_img_dir) if f.lower().endswith(('.png', '.jpg'))])
    assert len(purify_files) == len(train_cameras), f"修補圖數量 ({len(purify_files)}) 與 Train 相機數量 ({len(train_cameras)}) 不符！"
    
    for cam, pur_file in zip(train_cameras, purify_files):
        img_path = os.path.join(args.train_img_dir, pur_file)
        img_cv = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_cv = cv2.resize(img_cv, (cam.image_width, cam.image_height))
        cam.original_image = torch.tensor(img_cv, dtype=torch.float32).permute(2, 0, 1).cuda() / 255.0

# 🌍 動態計算場景空間範圍 (手動提取所有相機的中心點計算半徑)
    cam_centers = torch.stack([cam.camera_center for cam in train_cameras])
    extent = torch.norm(cam_centers - cam_centers.mean(dim=0), dim=-1).max().item() * 1.1
    print(f"🌍 計算出場景動態範圍 Extent: {extent:.2f}")
    with torch.no_grad():
        max_safe_scale = math.log(extent * 0.1) 
        scales = gaussians._scaling.detach()
        outlier_mask = scales > max_safe_scale
        if outlier_mask.any():
            scales[outlier_mask] = max_safe_scale
            gaussians._scaling = torch.nn.Parameter(scales.requires_grad_(True))

    l = [
        {'params': [gaussians._xyz], 'lr': 0.00016 * extent, "name": "xyz"},
        {'params': [gaussians._features_dc], 'lr': 0.0025, "name": "f_dc"},
        {'params': [gaussians._features_rest], 'lr': 0.0001 / 20.0, "name": "f_rest"},
        {'params': [gaussians._opacity], 'lr': 0.05, "name": "opacity"},
        {'params': [gaussians._scaling], 'lr': 0.005, "name": "scaling"},
        {'params': [gaussians._rotation], 'lr': 0.001, "name": "rotation"}
    ]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    gaussians.optimizer = optimizer
    xyz_scheduler_args = get_expon_lr_func(lr_init=0.00016 * extent, lr_final=0.0000016 * extent, max_steps=10000)

    lpips_train_fn = lpips.LPIPS(net='vgg').to("cuda").eval()
    for param in lpips_train_fn.parameters(): param.requires_grad = False

    # ==========================================
    # 🔥 核心優化迴圈
    # ==========================================
    # print("🔥 開始優化 3D 場景...")
    # for iteration in tqdm(range(1, 5001), desc="Training"):
    #     for param_group in optimizer.param_groups:
    #         if param_group["name"] == "xyz": param_group['lr'] = xyz_scheduler_args(iteration)

    #     viewpoint_cam = random.choice(train_cameras)

    print("🔥 開始優化 3D 場景...")
    camera_cycle = train_cameras.copy()   # ← 加這兩行
    random.shuffle(camera_cycle)          # ← 
    cycle_idx = 0                         # ←

    for iteration in tqdm(range(1, 5001), desc="Training"):
        for param_group in optimizer.param_groups:
            if param_group["name"] == "xyz": param_group['lr'] = xyz_scheduler_args(iteration)

        viewpoint_cam = camera_cycle[cycle_idx % len(camera_cycle)]  # ← 改這行
        cycle_idx += 1                                                 # ← 加這行
        if cycle_idx % len(camera_cycle) == 0:                        # ← 加這三行
            random.shuffle(camera_cycle)                               # ←


        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg_color)
        image = render_pkg["render"]
        
        # gt_image = viewpoint_cam.original_image
        # Ll1 = l1_loss(image, gt_image)
        # lpips_loss_val = lpips_train_fn(image.unsqueeze(0) * 2.0 - 1.0, gt_image.unsqueeze(0) * 2.0 - 1.0).mean()
        
        # loss = (1.0 - 0.2) * Ll1 + 0.2 * lpips_loss_val 
        # loss.backward()


        gt_image = viewpoint_cam.original_image
        Ll1 = l1_loss(image, gt_image)
        lpips_loss_val = lpips_train_fn(
            image.unsqueeze(0) * 2.0 - 1.0, 
            gt_image.unsqueeze(0) * 2.0 - 1.0
        ).mean()

        s_loss = ssim_loss(image, gt_image)

        loss = 0.7 * Ll1 + 0.1 * s_loss + 0.2 * lpips_loss_val
        loss.backward()        




        with torch.no_grad():
            if iteration < 3500 and gaussians.get_xyz.shape[0] < 800000:
                #3500
                gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(gaussians.max_radii2D[render_pkg["visibility_filter"]], render_pkg["radii"][render_pkg["visibility_filter"]])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], render_pkg["visibility_filter"])
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if iteration % 250 == 0: gaussians.active_sh_degree = min(gaussians.active_sh_degree + 1, gaussians.max_sh_degree)
            #500

            if iteration > 200 and iteration < 3500:
                if iteration % 100 == 0:
                    gaussians.densify_and_prune(max_grad=0.0002, min_opacity=0.005, extent=extent, max_screen_size=20)
                if iteration % 33 == 0:
                    prune_mask = (gaussians.get_opacity < 0.005).squeeze()
                    if prune_mask.any(): gaussians.prune_points(prune_mask)
                if iteration % 1000 == 0:
                    gaussians.reset_opacity()

    # ==========================================
    # 📸 嚴格 Test 渲染
    # ==========================================
    print("\n📸 開始使用保留的 40 個 GT 位姿進行 NVS 渲染...")
    with torch.no_grad():
        for cam in tqdm(test_cameras, desc="Rendering"):
            render_pkg = render(cam, gaussians, pipe, bg_color)
            save_image(render_pkg["render"], os.path.join(args.output_dir, f"{cam.image_name}.png"))
    print(f"✅ 渲染完成，嚴格測試影像已儲存至: {args.output_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--init_colmap", type=str, required=True, help=" VGGT 僅用 60 張圖重建的幾何目錄 (Train)")
    parser.add_argument("--train_img_dir", type=str, required=True, help=" 60 張 inpainted 修補圖目錄")
    parser.add_argument("--test_colmap", type=str, required=True, help=" 包含 100 張 GT 位姿的原始 COLMAP 目錄")
    parser.add_argument("--output_dir", type=str, required=True, help=" 渲染影像輸出目錄")
    args = parser.parse_args()
    train_and_render(args)