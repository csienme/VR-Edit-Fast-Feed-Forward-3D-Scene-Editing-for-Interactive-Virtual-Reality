import os
import torch
import random
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from math import exp
import torch.nn.functional as F

# 載入我們剛剛寫好的核心模組
from scene import Scene
from scene.gaussian_model import GaussianModel

# 載入你提供的渲染器
try:
    from gaussian_renderer.render import render
except ImportError:
    print("⚠️ 找不到 gaussian_renderer/render.py，請確認資料夾結構！")
    exit(1)

import torch.nn as nn

# ==========================================
# 幾何正則化模組 (移植自 3DGIS)
# ==========================================
# def cal_gradient(data):
#     """ 計算 2D 空間梯度 (Sobel 濾波器概念) """
#     kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
#     kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

#     kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
#     kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

#     weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
#     weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

#     grad_x = F.conv2d(data, weight_x, padding='same')
#     grad_y = F.conv2d(data, weight_y, padding='same')
#     gradient = torch.abs(grad_x) + torch.abs(grad_y)

#     return gradient

# def bilateral_smooth_loss(depth, image):
#     """ 邊緣感知的深度平滑約束：在顏色連續的區域強制深度平滑，消除浮空雜訊 """
#     # image: [3, H, W], depth: [1, H, W]
#     img_grad = cal_gradient(image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]
#     depth_grad = cal_gradient(depth.unsqueeze(0)).squeeze(0)  # [1, H, W]

#     # 當影像梯度大(有邊緣)時，exp(-img_grad) 趨近於 0，允許深度斷層
#     # 當影像平滑時，強制深度梯度(depth_grad)也必須極小化
#     smooth_loss = (depth_grad * torch.exp(-img_grad)).mean()

#     return smooth_loss




# # ==========================================
# # 極速 3D 空間正則化模組 (移植並改良自 3DGIS)
# # ==========================================
# def lightweight_knn_smooth_loss(xyz, features, scales, sample_size=4000, k=4):
#     """
#     極度輕量級的 3D 空間平滑約束。
#     隨機抽取少量高斯球，計算與其周圍 K 個鄰居的特徵與體積差異。
#     """
#     num_pts = xyz.shape[0]
#     if num_pts < sample_size:
#         return torch.tensor(0.0, device=xyz.device)
        
#     # 1. 隨機採樣 (極大降低計算量，保持 Lightweight)
#     indices = torch.randperm(num_pts, device=xyz.device)[:sample_size]
#     sample_xyz = xyz[indices]
#     sample_features = features[indices]
#     sample_scales = scales[indices]
    
#     # 2. 計算這 4000 個點之間的距離矩陣 (PyTorch 矩陣運算，極快)
#     dists = torch.cdist(sample_xyz, sample_xyz)
#     _, neighbor_indices = dists.topk(k + 1, largest=False)
    
#     # 排除自己 (距離為 0 的點)
#     neighbor_indices = neighbor_indices[:, 1:] # [sample_size, k]
    
#     # 3. 計算與鄰居的特徵 (顏色/紋理) 差異
#     neighbor_features = sample_features[neighbor_indices] 
#     diff_features = sample_features.unsqueeze(1) - neighbor_features
#     loss_features = torch.norm(diff_features, dim=-1).mean()

#     # 4. 計算與鄰居的體積 (Scale) 差異 (消除突兀的巨大浮空物)
#     neighbor_scales = sample_scales[neighbor_indices]
#     diff_scales = sample_scales.unsqueeze(1) - neighbor_scales
#     loss_scales = torch.norm(diff_scales, dim=-1).mean()
    
#     return loss_features + loss_scales * 0.5


def get_expon_lr_func(lr_init, lr_final, max_steps):
    """ 指數衰減學習率：讓高斯球在後期『冷卻定型』 """
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0): return 0.0
        if step >= max_steps: return lr_final
        t = np.clip(step / max_steps, 0, 1)
        return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return helper



# ==========================================
# 輔助模組：簡單的 L1 與 SSIM Loss
# ==========================================
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# ==========================================
# 虛擬 Pipeline 參數
# ==========================================
class DummyPipeline:
    def __init__(self):
        self.compute_cov3D_python = False
        self.convert_SHs_python = False
        self.debug = False

# ==========================================
# 主訓練迴圈 (加入 3DGS 靈魂：Densification)
# ==========================================
# ==========================================
# 主訓練迴圈 (加入 XYZ 冷卻系統與精準 Extent)
# ==========================================
def train_and_render(colmap_path, output_dir, iterations=7000):
    print(f"🚀 啟動 3DGS 快速訓練與渲染管線 (迭代次數: {iterations})")
    
    os.makedirs(output_dir, exist_ok=True)
    render_out_dir = os.path.join(output_dir, "renders")
    os.makedirs(render_out_dir, exist_ok=True)

    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(colmap_path, gaussians, shuffle=True)
    
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    pipe = DummyPipeline()

    train_cameras = scene.getTrainCameras()

    # 💥 關鍵修改 1：精準計算場景動態範圍 (Extent)
    # 🌍 動態計算場景空間範圍 (手動提取所有相機的中心點計算半徑)
    cam_centers = torch.stack([cam.camera_center for cam in train_cameras])
    extent = torch.norm(cam_centers - cam_centers.mean(dim=0), dim=-1).max().item() * 1.1
    print(f"🌍 計算出場景動態範圍 Extent: {extent:.2f}")
    print(f"🌍 計算出場景動態範圍 Extent: {extent:.2f}")

    gaussians.spatial_lr_scale = extent
    if not hasattr(gaussians, 'max_radii2D'):
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

    # 設定 Optimizer (Adam)
    l = [
        # 注意：xyz 的學習率會隨著 extent 動態縮放
        {'params': [gaussians._xyz], 'lr': 0.00016 * extent, "name": "xyz"},
        {'params': [gaussians._features_dc], 'lr': 0.0025, "name": "f_dc"},
        {'params': [gaussians._features_rest], 'lr': 0.0001 / 20.0, "name": "f_rest"},
        {'params': [gaussians._opacity], 'lr': 0.05, "name": "opacity"},
        {'params': [gaussians._scaling], 'lr': 0.005, "name": "scaling"},
        {'params': [gaussians._rotation], 'lr': 0.001, "name": "rotation"}
    ]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    gaussians.optimizer = optimizer
    
    # 💥 關鍵修改 2：初始化 XYZ 座標的冷卻排程器
    xyz_scheduler_args = get_expon_lr_func(lr_init=0.00016 * extent, lr_final=0.0000016 * extent, max_steps=10000)

    print("\n🔥 開始優化 3D 高斯場景...")
    progress_bar = tqdm(range(1, iterations + 1), desc="Training Progress")
    
    for iteration in progress_bar:
        # 💥 關鍵修改 3：每一步動態衰減 XYZ 學習率，強制微觀收斂
        lr = xyz_scheduler_args(iteration)
        for param_group in optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = lr

        viewpoint_cam = random.choice(train_cameras)
        
# 標準 2D 渲染與光度 Loss
# 取得標準渲染包
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg_color)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim(image, gt_image)
        
        # 💥 Phase 1 核心：完全捨棄 3D 平滑約束，將 SSIM 權重從 0.2 暴增至 0.4
        # 機制：SSIM 評估的是「局部結構的變異數與共變異數」。拉高其權重，
        # 會強迫優化器不能只滿足於「平均顏色對齊 (L1)」，必須連邊緣的對比度 (高頻信號) 也要完全咬合。
        loss = (1.0 - 0.4) * Ll1 + 0.4 * Lssim 
        
        loss.backward()




        # 啟動演化引擎 (Densification & Pruning)
        with torch.no_grad():
            if iteration < 3500:
                # 追蹤 2D 梯度，這是決定哪裡要增加細節的「雷達」
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 逐漸解鎖高頻 SH 細節 (每 1000 步解鎖一階)
            if iteration % 500 == 0:
                gaussians.active_sh_degree = min(gaussians.active_sh_degree + 1, gaussians.max_sh_degree)

            # 在第 500 步解鎖細胞分裂，每 100 步繁衍一次
            if iteration > 200 and iteration < 3500:
                if iteration % 50 == 0:
                    gaussians.densify_and_prune(max_grad=0.0002, min_opacity=0.005, extent=extent, max_screen_size=1)
                
                # 每 3000 步清除一次透明度過低的「幽靈高斯球」
                if iteration % 1000 == 0:
                    gaussians.reset_opacity()

        # 顯示即時損失與「高斯球總數」
        if iteration % 100 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Pts": gaussians.get_xyz.shape[0]})

    print("✅ 訓練完成！")

    # ==========================================
    # 生成 2D 渲染對比圖
    # ==========================================
    print("\n📸 正在從所有視角渲染最終影像...")
    with torch.no_grad():
        for i, cam in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering Views")):
            # 渲染高斯球
            render_pkg = render(cam, gaussians, pipe, bg_color)
            rendered_img = render_pkg["render"]
            
            # 將 Tensor 轉換為 numpy (C, H, W) -> (H, W, C) -> BGR (OpenCV 格式)
            rendered_img_np = rendered_img.detach().cpu().numpy()
            rendered_img_np = np.transpose(rendered_img_np, (1, 2, 0))
            rendered_img_bgr = cv2.cvtColor(np.clip(rendered_img_np * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # 將 Ground Truth 也轉出來，方便並排對比
            gt_img_np = cam.original_image.detach().cpu().numpy()
            gt_img_np = np.transpose(gt_img_np, (1, 2, 0))
            gt_img_bgr = cv2.cvtColor(np.clip(gt_img_np * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # 水平拼接: 左邊是 GT，右邊是 3DGS 渲染結果
            comparison_img = np.hstack((gt_img_bgr, rendered_img_bgr))
            
            # 儲存
            save_path = os.path.join(render_out_dir, f"render_vs_gt_{cam.image_name}.png")
            cv2.imwrite(save_path, comparison_img)

    print(f"\n🎉 渲染任務大功告成！所有影像已存至: {render_out_dir}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train and Render Custom 3DGS")
    parser.add_argument("--colmap_path", type=str, required=True, help="你的 COLMAP 輸出目錄 (包含 sparse 和 images)")
    parser.add_argument("--output_dir", type=str, default="./custom_3dgs_output", help="訓練結果與渲染圖片的輸出目錄")
    parser.add_argument("--iterations", type=int, default=5000, help="訓練迭代次數")
    
    args = parser.parse_args()
    
    train_and_render(args.colmap_path, args.output_dir, args.iterations)