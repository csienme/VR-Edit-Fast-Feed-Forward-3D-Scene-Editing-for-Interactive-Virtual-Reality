"""
eval_metric_spinnerf_prtcl.py
==============================
Per-scene metric evaluation (SPIn-NeRF protocol).

Changes from original:
  - [NEW] Saves {scene}_metrics.json alongside {scene}_metrics.txt
          JSON is machine-readable and used by calculate_avg_metrics.py + grid_search.py

Output structure:
  {output_dir}/{exp_name}/{scene}_metrics.txt   ← human-readable (unchanged)
  {output_dir}/{exp_name}/{scene}_metrics.json  ← machine-readable (NEW)
"""

import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import lpips
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torchmetrics.image.fid import FrechetInceptionDistance


def get_bbox_from_mask(mask_tensor):
    mask_np = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, mask_np.shape[0], 0, mask_np.shape[1]
    contours = sorted(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contours[-1])
    H, W = mask_np.shape
    expand_r = int(h * 0.1)
    expand_c = int(w * 0.1)
    rmin = max(0, y - expand_r)
    rmax = min(H, y + h + expand_r)
    cmin = max(0, x - expand_c)
    cmax = min(W, x + w + expand_c)
    return rmin, rmax, cmin, cmax


def load_image_tensor(path, is_mask=False):
    if is_mask:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_float = img.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0
        return torch.tensor(img_float, dtype=torch.float32).unsqueeze(0)
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0


def evaluate_metrics(args):
    print(f"Evaluating: exp={args.exp_name}, scene={args.scene}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    fid_global = FrechetInceptionDistance(feature=2048).to(device)
    fid_masked = FrechetInceptionDistance(feature=2048).to(device)

    psnr_global_list, ssim_global_list, lpips_global_list = [], [], []
    psnr_masked_list, ssim_masked_list, lpips_masked_list = [], [], []

    render_names = sorted([f for f in os.listdir(args.render_img_dir) if f.endswith(('.png', '.jpg'))])
    gt_names     = sorted([f for f in os.listdir(args.gt_img_dir)     if f.endswith(('.png', '.jpg'))])
    mask_names   = sorted([f for f in os.listdir(args.mask_dir)       if f.endswith(('.png', '.jpg'))])

    # SPIn-NeRF dataset: GT = first 40 images (sorted order)
    gt_names = gt_names[:40]

    n = min(len(render_names), len(gt_names), len(mask_names))
    print(f"Render: {len(render_names)}, GT: {len(gt_names)}, Mask: {len(mask_names)} → Evaluating {n} pairs")

    if n == 0:
        print("ERROR: No pairs found.")
        return

    for i in tqdm(range(n), desc=f"[{args.exp_name}/{args.scene}]"):
        pred_path = os.path.join(args.render_img_dir, render_names[i])
        gt_path   = os.path.join(args.gt_img_dir,     gt_names[i])
        mask_path = os.path.join(args.mask_dir,        mask_names[i])

        pred_tensor = load_image_tensor(pred_path)
        gt_tensor   = load_image_tensor(gt_path)
        mask_tensor = load_image_tensor(mask_path, is_mask=True)

        _, gt_h, gt_w = gt_tensor.shape
        if pred_tensor.shape[1] != gt_h or pred_tensor.shape[2] != gt_w:
            pred_np = pred_tensor.permute(1, 2, 0).numpy()
            pred_np = cv2.resize(pred_np, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
            pred_tensor = torch.tensor(pred_np, dtype=torch.float32).permute(2, 0, 1)
        if mask_tensor.shape[1] != gt_h or mask_tensor.shape[2] != gt_w:
            mask_np = cv2.resize(mask_tensor.squeeze(0).numpy(), (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        gt_np   = gt_tensor.permute(1, 2, 0).numpy()
        pred_np = pred_tensor.permute(1, 2, 0).numpy()

        # Global
        psnr_global_list.append(compute_psnr(gt_np, pred_np, data_range=1.0))
        ssim_global_list.append(compute_ssim(gt_np, pred_np, data_range=1.0, channel_axis=2))
        lpips_in_pred = (pred_tensor * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_in_gt   = (gt_tensor   * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_global_list.append(lpips_vgg(lpips_in_pred, lpips_in_gt).item())
        fid_global.update((gt_tensor   * 255).to(torch.uint8).unsqueeze(0).to(device), real=True)
        fid_global.update((pred_tensor * 255).to(torch.uint8).unsqueeze(0).to(device), real=False)

        # Masked BBox (SPIn-NeRF protocol: 10% expansion)
        rmin, rmax, cmin, cmax = get_bbox_from_mask(mask_tensor)
        gt_crop   = gt_tensor[:,   rmin:rmax, cmin:cmax]
        pred_crop = pred_tensor[:, rmin:rmax, cmin:cmax]
        gt_crop_np   = gt_crop.permute(1, 2, 0).numpy()
        pred_crop_np = pred_crop.permute(1, 2, 0).numpy()

        psnr_masked_list.append(compute_psnr(gt_crop_np, pred_crop_np, data_range=1.0))
        win_size = min(7, gt_crop_np.shape[0], gt_crop_np.shape[1])
        win_size = win_size if win_size % 2 == 1 else win_size - 1
        if win_size >= 3:
            ssim_masked_list.append(compute_ssim(
                gt_crop_np, pred_crop_np, data_range=1.0, channel_axis=2, win_size=win_size))
        else:
            ssim_masked_list.append(0.0)
        lpips_in_pred_crop = (pred_crop * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_in_gt_crop   = (gt_crop   * 2.0 - 1.0).unsqueeze(0).to(device)
        lpips_masked_list.append(lpips_vgg(lpips_in_pred_crop, lpips_in_gt_crop).item())
        fid_masked.update((gt_crop   * 255).to(torch.uint8).unsqueeze(0).to(device), real=True)
        fid_masked.update((pred_crop * 255).to(torch.uint8).unsqueeze(0).to(device), real=False)

    fid_score_global = fid_global.compute().item()
    fid_score_masked = fid_masked.compute().item()

    # ── Save directory ──────────────────────────────────────────────────────────
    save_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── [NEW] Save .json for machine reading (grid_search.py reads this) ────────
    result_dict = {
        "exp_name": args.exp_name,
        "scene":    args.scene,
        "global": {
            "FID":   round(fid_score_global, 6),
            "LPIPS": round(float(np.mean(lpips_global_list)), 6),
            "PSNR":  round(float(np.mean(psnr_global_list)),  6),
            "SSIM":  round(float(np.mean(ssim_global_list)),  6),
        },
        "masked": {
            "FID":   round(fid_score_masked, 6),
            "LPIPS": round(float(np.mean(lpips_masked_list)), 6),
            "PSNR":  round(float(np.mean(psnr_masked_list)),  6),
            "SSIM":  round(float(np.mean(ssim_masked_list)),  6),
        },
    }
    json_path = os.path.join(save_dir, f"{args.scene}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"Saved json → {json_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--render_img_dir", type=str, required=True)
    parser.add_argument("--gt_img_dir",     type=str, required=True)
    parser.add_argument("--mask_dir",       type=str, required=True)
    parser.add_argument("--output_dir",     type=str, required=True,
                        help="Root output dir, e.g. metric_logs/")
    parser.add_argument("--exp_name",       type=str, required=True,
                        help="Experiment name, e.g. trial_0001")
    parser.add_argument("--scene",          type=str, required=True,
                        help="Scene name, e.g. 1")
    args = parser.parse_args()
    evaluate_metrics(args)