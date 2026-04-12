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
# 核心工具：取得 Mask 的 Bounding Box
# ==========================================
def get_bbox_from_mask(mask_tensor):
    mask_np = mask_tensor.squeeze().cpu().numpy()
    rows = np.any(mask_np > 0.5, axis=1)
    cols = np.any(mask_np > 0.5, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, mask_np.shape[0], 0, mask_np.shape[1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax + 1, cmin, cmax + 1

def load_image_tensor(path, is_mask=False):
    if is_mask:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_float = img.astype(np.float32)
        # mask 像素值可能是 0/1（非 0/255），統一 normalize 到 0/1
        if img_float.max() <= 1.0:
            pass
        else:
            img_float = img_float / 255.0
        return torch.tensor(img_float, dtype=torch.float32).unsqueeze(0)
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

def evaluate_metrics(args):
    print("📊 啟動 NVS 學術指標計算引擎 (3DGIC Pixel-Level Mask Protocol)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    fid_global = FrechetInceptionDistance(feature=2048).to(device)
    fid_masked = FrechetInceptionDistance(feature=2048).to(device)

    psnr_global_list, ssim_global_list, lpips_global_list = [], [], []
    psnr_masked_list, ssim_masked_list, lpips_masked_list = [], [], []

    img_names  = sorted([f for f in os.listdir(args.render_img_dir) if f.endswith(('.png', '.jpg'))])
    mask_names = sorted([f for f in os.listdir(args.mask_dir)       if f.endswith(('.png', '.jpg'))])

    if len(img_names) != len(mask_names):
        print(f"⚠️ 警告: Render 數量 ({len(img_names)}) 與 Mask 數量 ({len(mask_names)}) 不一致！")
        min_len    = min(len(img_names), len(mask_names))
        img_names  = img_names[:min_len]
        mask_names = mask_names[:min_len]

    for img_name, mask_name in tqdm(zip(img_names, mask_names), total=len(img_names), desc="Evaluating"):
        pred_path  = os.path.join(args.render_img_dir, img_name)
        mask_path  = os.path.join(args.mask_dir, mask_name)
        clean_name = img_name.replace(".png.png", ".png").replace(".jpg.png", ".jpg")
        gt_path    = os.path.join(args.gt_img_dir, clean_name)

        if not os.path.exists(gt_path):
            print(f"\n⚠️ 找不到 GT: {gt_path}，跳過。")
            continue

        pred_tensor = load_image_tensor(pred_path)
        gt_tensor   = load_image_tensor(gt_path)
        mask_tensor = load_image_tensor(mask_path, is_mask=True)

        # resize mask to match GT resolution
        _, gt_h, gt_w = gt_tensor.shape
        if mask_tensor.shape[1] != gt_h or mask_tensor.shape[2] != gt_w:
            mask_np_resized = cv2.resize(
                mask_tensor.squeeze(0).numpy(), (gt_w, gt_h),
                interpolation=cv2.INTER_NEAREST
            )
            mask_tensor = torch.tensor(mask_np_resized, dtype=torch.float32).unsqueeze(0)

        # ── 1. Global ─────────────────────────────────────────────────
        gt_np   = gt_tensor.permute(1, 2, 0).numpy()
        pred_np = pred_tensor.permute(1, 2, 0).numpy()

        psnr_global_list.append(compute_psnr(gt_np, pred_np, data_range=1.0))
        ssim_global_list.append(compute_ssim(gt_np, pred_np, data_range=1.0, channel_axis=2))

        lpips_in_pred = (pred_tensor * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_in_gt   = (gt_tensor   * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_global_list.append(lpips_vgg(lpips_in_pred, lpips_in_gt).item())

        fid_global.update((gt_tensor   * 255).to(torch.uint8).unsqueeze(0).to(device), real=True)
        fid_global.update((pred_tensor * 255).to(torch.uint8).unsqueeze(0).to(device), real=False)

        # ── 2. Masked — 3DGIC Pixel-Level Protocol ────────────────────
        mask_bool = mask_tensor.squeeze(0).numpy() > 0.5   # (H, W) bool

        # m-PSNR: 只取 mask 內像素的 MSE
        gt_px   = gt_tensor.permute(1, 2, 0).numpy()[mask_bool]    # (N_px, 3)
        pred_px = pred_tensor.permute(1, 2, 0).numpy()[mask_bool]  # (N_px, 3)
        mse = np.mean((gt_px - pred_px) ** 2)
        psnr_masked_list.append(10 * np.log10(1.0 / mse) if mse > 0 else 100.0)

        # m-SSIM: BBox crop（SSIM 是基於局部統計窗口的空間指標，
        #         無法對不連續的散點集合計算，必須保留空間鄰域）
        rmin, rmax, cmin, cmax = get_bbox_from_mask(mask_tensor)
        gt_crop   = gt_tensor[:,   rmin:rmax, cmin:cmax]
        pred_crop = pred_tensor[:, rmin:rmax, cmin:cmax]
        gt_crop_np   = gt_crop.permute(1, 2, 0).numpy()
        pred_crop_np = pred_crop.permute(1, 2, 0).numpy()

        win_size = min(7, gt_crop_np.shape[0], gt_crop_np.shape[1])
        win_size = win_size - 1 if win_size % 2 == 0 else win_size
        if win_size >= 3:
            ssim_masked_list.append(
                compute_ssim(gt_crop_np, pred_crop_np, data_range=1.0,
                             channel_axis=2, win_size=win_size))
        else:
            ssim_masked_list.append(0.0)

        # m-LPIPS: mask 外像素歸零，只讓 mask 內紋理貢獻 feature distance
        mask_3ch    = mask_tensor.expand(3, -1, -1)
        pred_masked = pred_tensor * mask_3ch
        gt_masked   = gt_tensor   * mask_3ch
        lpips_in_pred_m = (pred_masked * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_in_gt_m   = (gt_masked   * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_masked_list.append(lpips_vgg(lpips_in_pred_m, lpips_in_gt_m).item())

        # m-FID: BBox crop（見 output 下方說明）
        fid_masked.update((gt_crop   * 255).to(torch.uint8).unsqueeze(0).to(device), real=True)
        fid_masked.update((pred_crop * 255).to(torch.uint8).unsqueeze(0).to(device), real=False)

    # ── 結算輸出 ──────────────────────────────────────────────────────
    fid_score_global = fid_global.compute().item()
    fid_score_masked = fid_masked.compute().item()

    run_name     = os.path.basename(os.path.normpath(args.render_img_dir))
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
        "【 局部指標 (Masked) - 3DGIC Protocol 】\n"
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
    parser.add_argument("--gt_img_dir",     type=str, required=True)
    parser.add_argument("--mask_dir",       type=str, required=True)
    parser.add_argument("--output_dir",     type=str, required=True)
    args = parser.parse_args()
    evaluate_metrics(args)