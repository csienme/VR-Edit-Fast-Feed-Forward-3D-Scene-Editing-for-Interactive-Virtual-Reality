import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import lpips
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torchmetrics.image.fid import FrechetInceptionDistance

# ==========================================
# 核心工具：取得 Mask 的 Bounding Box (已替換為 A 版的 Contour 最大區塊邏輯)
# ==========================================
def get_bbox_from_mask(mask_tensor):
    """
    從 Mask 張量中提取最小包圍矩形 (Bounding Box)
    套用 cv2.findContours 邏輯，只鎖定「面積最大的 Mask 區塊」，避免受到零星雜訊干擾。
    """
    # 1. 將 PyTorch Tensor (1, H, W) 且值為 0~1 的格式，轉回 OpenCV 支援的 Numpy uint8 (0~255) 格式
    mask_np = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # 2. 使用 OpenCV 尋找所有輪廓 (Contours)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 防呆機制：如果 Mask 全黑（找不到任何輪廓），則回傳整張圖範圍避免程式崩潰
    if not contours:
        return 0, mask_np.shape[0], 0, mask_np.shape[1]

    # 3. 依照面積大小對輪廓進行排序，並取出面積最大的那一個 (即 contours[-1])
    contours = sorted(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contours[-1])

    # 4. 轉換為 array slicing 用的 rmin, rmax, cmin, cmax
    # row 對應 y (高), col 對應 x (寬)
    rmin, rmax = y, y + h
    cmin, cmax = x, x + w

    return rmin, rmax, cmin, cmax

def load_image_tensor(path, is_mask=False):
    if is_mask:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_float = img.astype(np.float32)
        # mask 像素值可能是 0/1（非 0/255），統一 normalize 到 0/1
        if img_float.max() <= 1.0:
            pass  # 已經是 0/1，不需要除
        else:
            img_float = img_float / 255.0
        return torch.tensor(img_float, dtype=torch.float32).unsqueeze(0)
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

def evaluate_metrics(args):
    print("📊 啟動 NVS 學術指標計算引擎 (已套用 Maximum Contour BBox 邏輯)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化 Metrics 實體
    lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    
    # torchmetrics 的 FID 預設接收 (N, 3, H, W) 的 uint8 張量 [0, 255]
    fid_global = FrechetInceptionDistance(feature=2048).to(device)
    fid_masked = FrechetInceptionDistance(feature=2048).to(device)

    psnr_global_list, ssim_global_list, lpips_global_list = [], [], []
    psnr_masked_list, ssim_masked_list, lpips_masked_list = [], [], []

    # 取得並排序 Render 檔名
    img_names = sorted([f for f in os.listdir(args.render_img_dir) if f.endswith(('.png', '.jpg'))])
    
    # 取得並排序 Mask 檔名 (解決 out_00001.png 命名不對稱問題)
    mask_names = sorted([f for f in os.listdir(args.mask_dir) if f.endswith(('.png', '.jpg'))])

    # 防呆：確保兩邊數量一致
    if len(img_names) != len(mask_names):
        print(f"⚠️ 警告: Render 數量 ({len(img_names)}) 與 Mask 數量 ({len(mask_names)}) 不一致！請檢查資料集。")
        # 取最小數量以防崩潰
        min_len = min(len(img_names), len(mask_names))
        img_names = img_names[:min_len]
        mask_names = mask_names[:min_len]

    # 使用 zip 進行一對一配對迴圈
    for img_name, mask_name in tqdm(zip(img_names, mask_names), total=len(img_names), desc="Evaluating"):
        pred_path = os.path.join(args.render_img_dir, img_name)
        mask_path = os.path.join(args.mask_dir, mask_name)
        
        # 處理 GT 檔名 (去除雙重副檔名)
        clean_name = img_name.replace(".png.png", ".png").replace(".jpg.png", ".jpg")
        gt_path = os.path.join(args.gt_img_dir, clean_name)

        if not os.path.exists(gt_path):
            print(f"\n⚠️ 找不到 GT: {gt_path}，跳過。")
            continue

        pred_tensor = load_image_tensor(pred_path)
        gt_tensor = load_image_tensor(gt_path)
        mask_tensor = load_image_tensor(mask_path, is_mask=True)
        
        # Bug fix: resize mask to match GT resolution (mask may be different res, e.g. 576 vs 567)
        _, gt_h, gt_w = gt_tensor.shape
        if mask_tensor.shape[1] != gt_h or mask_tensor.shape[2] != gt_w:
            mask_np_resized = cv2.resize(
                mask_tensor.squeeze(0).numpy(), (gt_w, gt_h),
                interpolation=cv2.INTER_NEAREST
            )
            mask_tensor = torch.tensor(mask_np_resized, dtype=torch.float32).unsqueeze(0)
            
        # ----------------------------------------------------
        # 1. 全局指標計算 (Global) - 整張影像
        # ----------------------------------------------------
        gt_np = gt_tensor.permute(1, 2, 0).numpy()
        pred_np = pred_tensor.permute(1, 2, 0).numpy()

        psnr_global_list.append(compute_psnr(gt_np, pred_np, data_range=1.0))
        ssim_global_list.append(compute_ssim(gt_np, pred_np, data_range=1.0, channel_axis=2))
        
        # LPIPS 需要 [-1, 1] 範圍
        lpips_in_pred = (pred_tensor * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_in_gt = (gt_tensor * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_global_list.append(lpips_vgg(lpips_in_pred, lpips_in_gt).item())

        # FID 需要 uint8 [0, 255]
        fid_global.update((gt_tensor * 255).to(torch.uint8).unsqueeze(0).to(device), real=True)
        fid_global.update((pred_tensor * 255).to(torch.uint8).unsqueeze(0).to(device), real=False)

        # ----------------------------------------------------
        # 2. 局部指標計算 (Masked BBox) - 依最大 Contour 裁切
        # ----------------------------------------------------
        # 取得 BBox 座標
        rmin, rmax, cmin, cmax = get_bbox_from_mask(mask_tensor)

        # 裁切影像 (Crop)
        gt_crop = gt_tensor[:, rmin:rmax, cmin:cmax]
        pred_crop = pred_tensor[:, rmin:rmax, cmin:cmax]

        gt_crop_np = gt_crop.permute(1, 2, 0).numpy()
        pred_crop_np = pred_crop.permute(1, 2, 0).numpy()

        # 計算 Crop 區域的 PSNR / SSIM
        psnr_masked_list.append(compute_psnr(gt_crop_np, pred_crop_np, data_range=1.0))
        
        # 若 Crop 區域過小 (小於 7x7)，SSIM 會報錯，需加上防呆
        win_size = min(7, gt_crop_np.shape[0], gt_crop_np.shape[1])
        win_size = win_size - 1 if win_size % 2 == 0 else win_size # 確保 win_size 是奇數
        
        if win_size >= 3:
            ssim_masked_list.append(compute_ssim(gt_crop_np, pred_crop_np, data_range=1.0, channel_axis=2, win_size=win_size))
        else:
            ssim_masked_list.append(0.0) # 區域過小無法計算結構相似度

        # 計算 Crop 區域的 LPIPS
        lpips_in_pred_crop = (pred_crop * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_in_gt_crop = (gt_crop * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_masked_list.append(lpips_vgg(lpips_in_pred_crop, lpips_in_gt_crop).item())

        # 計算 Crop 區域的 FID
        fid_masked.update((gt_crop * 255).to(torch.uint8).unsqueeze(0).to(device), real=True)
        fid_masked.update((pred_crop * 255).to(torch.uint8).unsqueeze(0).to(device), real=False)


    # --- 結算輸出 ---
    fid_score_global = fid_global.compute().item()
    fid_score_masked = fid_masked.compute().item()

    run_name = os.path.basename(os.path.normpath(args.render_img_dir))
    txt_log_path = os.path.join(args.output_dir, f"{run_name}_metrics.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    log_content = (
        "===========================================\n"
        f"📈 最終學術評估結果 - 實驗: {run_name}\n"
        "===========================================\n"
        "【 全局指標 (Global) - 整張影像 】\n"
        f"   ► FID   : {fid_score_global:.4f}\n"
        f"   ► LPIPS : {np.mean(lpips_global_list):.4f}\n"
        f"   ► PSNR  : {np.mean(psnr_global_list):.4f}\n"
        f"   ► SSIM  : {np.mean(ssim_global_list):.4f}\n\n"
        "【 局部指標 (BBox Masked) - 最大 Contour 裁切 】\n"
        f"   ► m-FID   : {fid_score_masked:.4f}\n"
        f"   ► m-LPIPS : {np.mean(lpips_masked_list):.4f}\n"
        f"   ► m-PSNR  : {np.mean(psnr_masked_list):.4f}\n"
        f"   ► m-SSIM  : {np.mean(ssim_masked_list):.4f}\n"
        "===========================================\n"
    )

    print("\n" + log_content)
    with open(txt_log_path, "w") as f:
        f.write(log_content)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--render_img_dir", type=str, required=True)
    parser.add_argument("--gt_img_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    evaluate_metrics(args)