import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import save_image

import lpips
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torchmetrics.image.fid import FrechetInceptionDistance

def load_image_tensor(path, is_mask=False):
    if is_mask:
        # Mask 讀取為單通道灰階
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

def evaluate_metrics(args):
    print("📊 啟動 NVS 學術指標計算引擎 (含 Masked 區域黑底評估 - 嚴格對齊 SOTA)")
    
    # --- 動態建立資料夾結構 ---
    run_name = os.path.basename(os.path.normpath(args.render_img_dir))
    base_out_dir = os.path.join(args.output_dir, run_name)
    compare_dir = os.path.join(base_out_dir, "compare_results")
    txt_log_path = os.path.join(base_out_dir, "metrics_log.txt")
    os.makedirs(compare_dir, exist_ok=True)

    # 1. 獲取並排序圖檔
    all_gt_files = sorted([f for f in os.listdir(args.gt_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    gt_files = all_gt_files[:40]
    render_files = sorted([f for f in os.listdir(args.render_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # 讀取 Mask 檔案
    all_mask_files = sorted([f for f in os.listdir(args.masked_label_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = all_mask_files[:40]

    print("\n" + "="*50)
    print("📋 [資料夾讀取狀態確認]")
    print(f"   ► GT 抽取數量    : {len(gt_files)} 張")
    print(f"   ► Render 產物數量: {len(render_files)} 張")
    print(f"   ► Mask 標籤數量  : {len(mask_files)} 張")
    print(f"   ► 儲存目的地     : {base_out_dir}")
    print("="*50 + "\n")

    assert len(gt_files) == 40, f"❌ GT 影像提取數量異常！實際為 {len(gt_files)} 張。"
    assert len(render_files) == 40, f"❌ Render 產物數量異常！實際為 {len(render_files)} 張。"
    assert len(mask_files) == 40, f"❌ Mask 影像提取數量異常！實際為 {len(mask_files)} 張。"

    # 2. 初始化評估模組 (全圖 vs 局部)
    lpips_fn = lpips.LPIPS(net='vgg').to("cuda").eval()
    for param in lpips_fn.parameters(): param.requires_grad = False
    
    fid_metric_global = FrechetInceptionDistance(feature=2048).to("cuda")
    fid_metric_masked = FrechetInceptionDistance(feature=2048).to("cuda")

    # 數據紀錄列表
    psnr_global, ssim_global, lpips_global = [], [], []
    psnr_masked, ssim_masked, lpips_masked = [], [], []

    print("🔍 逐張計算 全局 (Global) 與 局部 (Masked) 誤差...")
    with torch.no_grad():
        for gt_name, render_name, mask_name in tqdm(zip(gt_files, render_files, mask_files), total=40):
            # 讀取 Tensor
            gt_tensor = load_image_tensor(os.path.join(args.gt_img_dir, gt_name)).cuda()
            render_tensor = load_image_tensor(os.path.join(args.render_img_dir, render_name)).cuda()
            mask_tensor = load_image_tensor(os.path.join(args.masked_label_dir, mask_name), is_mask=True).cuda()
            
            if gt_tensor.shape != render_tensor.shape:
                render_tensor = torch.nn.functional.interpolate(render_tensor.unsqueeze(0), size=(gt_tensor.shape[1], gt_tensor.shape[2]), mode='bilinear').squeeze(0)
            if mask_tensor.shape[1:] != gt_tensor.shape[1:]:
                mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0), size=(gt_tensor.shape[1], gt_tensor.shape[2]), mode='nearest').squeeze(0)

            # --- 全局 (Global) 評估 ---
            gt_np = gt_tensor.permute(1, 2, 0).cpu().numpy()
            render_np = render_tensor.permute(1, 2, 0).cpu().numpy()
            psnr_global.append(compute_psnr(gt_np, render_np, data_range=1.0))
            ssim_global.append(compute_ssim(gt_np, render_np, data_range=1.0, channel_axis=2))
            
            pred_lpips_g = render_tensor.unsqueeze(0) * 2.0 - 1.0
            gt_lpips_g = gt_tensor.unsqueeze(0) * 2.0 - 1.0
            lpips_global.append(lpips_fn(pred_lpips_g, gt_lpips_g).item())
            
            fid_metric_global.update((gt_tensor.unsqueeze(0) * 255).byte(), real=True)
            fid_metric_global.update((render_tensor.unsqueeze(0) * 255).byte(), real=False)

            # --- 局部 (Masked) 評估 (嚴格對齊學術界算法) ---
            # 1. 取得絕對的 0/1 遮罩，並確保 1 代表破洞 (修補區)
            mask_binary = (mask_tensor > 0).float()
            if mask_binary.mean() > 0.5:
                mask_binary = 1.0 - mask_binary  # 面積過大代表反轉了，強制作反
                
            # 2. 論文核心邏輯：把背景全部乘上 0 變黑，保留全尺寸
            gt_masked = gt_tensor * mask_binary
            render_masked = render_tensor * mask_binary

            # 3. PSNR / SSIM (Masked)
            gt_masked_np = gt_masked.permute(1, 2, 0).cpu().numpy()
            render_masked_np = render_masked.permute(1, 2, 0).cpu().numpy()
            psnr_masked.append(compute_psnr(gt_masked_np, render_masked_np, data_range=1.0))
            ssim_masked.append(compute_ssim(gt_masked_np, render_masked_np, data_range=1.0, channel_axis=2))

            # 4. 論文的 m-LPIPS 計算：含有 90% 黑底的張量直接算 LPIPS
            pred_lpips_m = render_masked.unsqueeze(0) * 2.0 - 1.0
            gt_lpips_m = gt_masked.unsqueeze(0) * 2.0 - 1.0
            lpips_masked.append(lpips_fn(pred_lpips_m, gt_lpips_m).item())

            # 5. 論文的 m-FID 計算
            fid_metric_masked.update((gt_masked.unsqueeze(0) * 255).byte(), real=True)
            fid_metric_masked.update((render_masked.unsqueeze(0) * 255).byte(), real=False)

            # 儲存對比圖 (全圖拼接)
            save_image(torch.cat([gt_tensor, render_tensor], dim=2), os.path.join(compare_dir, f"cmp_{gt_name}"))

    # 計算最終 FID
    fid_score_global = fid_metric_global.compute().item()
    fid_score_masked = fid_metric_masked.compute().item()

    # --- 整理輸出字串 ---
    log_content = (
        "===========================================\n"
        f"📈 最終學術評估結果 (Test Set 40張) - 實驗: {run_name}\n"
        "===========================================\n"
        "【 全局指標 (Global) - 整張影像 】\n"
        f"   ► FID   : {fid_score_global:.4f}\n"
        f"   ► LPIPS : {np.mean(lpips_global):.4f}\n"
        f"   ► PSNR  : {np.mean(psnr_global):.4f}\n"
        f"   ► SSIM  : {np.mean(ssim_global):.4f}\n\n"
        "【 局部指標 (Masked) - 僅計算修補區域 】\n"
        f"   ► m-FID   : {fid_score_masked:.4f}\n"
        f"   ► m-LPIPS : {np.mean(lpips_masked):.4f}\n"
        f"   ► m-PSNR  : {np.mean(psnr_masked):.4f}\n"
        f"   ► m-SSIM  : {np.mean(ssim_masked):.4f}\n"
        "===========================================\n"
    )

    # 印出並寫入檔案
    print("\n" + log_content)
    with open(txt_log_path, "w") as f:
        f.write(log_content)
    
    print(f"📁 評估結果與對比圖已完整儲存至: {base_out_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gt_img_dir", type=str, required=True, help="包含 GT 原圖的目錄 (將自動取前 40 張)")
    parser.add_argument("--render_img_dir", type=str, required=True, help="渲染出的 40 張影像目錄")
    parser.add_argument("--masked_label_dir", type=str, required=True, help="包含 40 張黑白 Mask 遮罩圖的目錄")
    parser.add_argument("--output_dir", type=str, default="./metric_logs", help="根輸出目錄")
    args = parser.parse_args()
    evaluate_metrics(args)