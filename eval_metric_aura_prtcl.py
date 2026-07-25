"""
eval_metric_aura_prtcl.py
=========================
Per-scene metric evaluation following the **AuraFusion360** protocol (360-USID).

Drop-in replacement for eval_metric_spinnerf_prtcl.py. Same CLI args, same JSON
schema (so calculate_avg_metrics.py / grid_search.py keep working), but the
metric *definitions* and the GT/render matching follow aura_eval.py.

Key differences from eval_metric_spinnerf_prtcl.py
--------------------------------------------------
1. Matching is by FILENAME STEM, not positional.
   In 360-USID the test views (e.g. 00188) sort AFTER all training views
   (00000-00185). The SPIn-NeRF script took gt_names[:40] + render[i]<->gt[i],
   which would pick the WRONG frames here. Stem matching also automatically
   selects the ~30 test renders out of the ~220 rendered NVS poses, and handles
   the .jpg(GT) / .png(render) extension mismatch.

2. "Object" (masked) metrics use AuraFusion's PIXEL-WISE mask convention
   (NOT SPIn-NeRF's 10%-expanded bounding-box crop):
       masked PSNR  : MSE summed over object pixels
       masked LPIPS : spatial LPIPS map, mask-weighted average
       masked SSIM  : full-image SSIM computed on mask-multiplied images
   These are the numbers comparable to AuraFusion Table 1 (PSNR / LPIPS).

3. "Global" metrics are full-image PSNR / SSIM / LPIPS.

4. LPIPS inputs are fed in [0, 1] (NOT scaled to [-1, 1]), exactly as
   aura_eval.py does, so the values line up with AuraFusion's reported numbers.
   (This is intentionally different from spinnerf_prtcl, which used [-1, 1].)

5. FID is AuraFusion's full-image test-set FID. There is no separate masked FID
   under this protocol, so the same FID value is written into both json blocks.
   NOTE: aura_eval.py uses pytorch_fid; here we use torchmetrics FID (the same
   backend your golden-baseline pipeline already uses) for env consistency and
   so a missing pytorch-fid never crashes the run. If you need AuraFusion's
   *exact* FID number for a head-to-head, swap the FID block for
   pytorch_fid.fid_score.calculate_fid_given_paths.

Self-contained: psnr / masked_psnr / ssim are replicated inline so this script
does NOT import a repo-specific utils/ package (FORGE has no masked_psnr).

Output:
  {output_dir}/{exp_name}/{scene}_metrics.json
"""

import os
import json
import math
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as tf
from tqdm import tqdm

import lpips
from torchmetrics.image.fid import FrechetInceptionDistance


# ----------------------------------------------------------------------------
# Metric primitives (replicated from 3DGS / AuraFusion utils)
# ----------------------------------------------------------------------------
def psnr(img1, img2):
    """Full-image PSNR. img: [B,3,H,W] in [0,1]. (3DGS utils/image_utils.py)"""
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def masked_psnr(img1, img2, mask):
    """
    PSNR over object pixels only.
    img : [B,3,H,W] in [0,1]; mask: [B,1,H,W] or [B,3,H,W] in {0,1}.
    NOTE: standard masked-PSNR form (MSE = sum(se*mask)/sum(mask)). If your
    AuraFusion repo's utils.image_utils.masked_psnr uses a different
    normalisation, send it and this one line can be matched exactly.
    """
    if mask.shape[1] == 1:
        mask = mask.repeat(1, img1.shape[1], 1, 1)
    se = (img1 - img2) ** 2
    mse = (se * mask).sum() / (mask.sum() + 1e-8)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _gaussian(window_size, sigma):
    g = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                      for x in range(window_size)])
    return g / g.sum()


def _create_window(window_size, channel):
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11):
    """Gaussian-window SSIM (3DGS utils/loss_utils.py). img: [B,3,H,W] in [0,1]."""
    channel = img1.size(-3)
    window = _create_window(window_size, channel).to(img1.device).type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _load_rgb(path, device):
    img = Image.open(path).convert("RGB")
    return tf.to_tensor(img).unsqueeze(0).to(device)          # [1,3,H,W] in [0,1]


def _load_mask(path, device):
    m = Image.open(path).convert("L")
    t = tf.to_tensor(m).unsqueeze(0).to(device)               # [1,1,H,W] in [0,1]
    t = (t > 0.5).float()
    return t


def _resize_to(t, h, w, mode):
    if t.shape[-2] == h and t.shape[-1] == w:
        return t
    if mode == "nearest":
        return F.interpolate(t, size=(h, w), mode="nearest")
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------
def evaluate_metrics(args):
    print(f"[AuraProtocol] exp={args.exp_name}, scene={args.scene}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_orig = lpips.LPIPS(net='vgg').to(device)
    lpips_spat = lpips.LPIPS(net='vgg', spatial=True).to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    EXT = ('.png', '.jpg', '.jpeg')
    gt_map   = {_stem(f): os.path.join(args.gt_img_dir, f)
                for f in os.listdir(args.gt_img_dir)     if f.lower().endswith(EXT)}
    rnd_map  = {_stem(f): os.path.join(args.render_img_dir, f)
                for f in os.listdir(args.render_img_dir) if f.lower().endswith(EXT)}
    mask_map = {_stem(f): os.path.join(args.mask_dir, f)
                for f in os.listdir(args.mask_dir)        if f.lower().endswith(EXT)}

    # stems are zero-padded -> lexical sort == numeric sort
    common = sorted(set(gt_map) & set(rnd_map) & set(mask_map))
    print(f"GT: {len(gt_map)}, Render: {len(rnd_map)}, Mask: {len(mask_map)} "
          f"-> matched by stem: {len(common)}")

    missing = sorted(set(gt_map) & set(mask_map) - set(rnd_map))
    if missing:
        print(f"WARNING: {len(missing)} test view(s) have GT+mask but NO render, "
              f"e.g. {missing[:5]}. Check render.py output naming / coverage.")
    if len(common) == 0:
        print("ERROR: no (gt, render, mask) triples matched by filename stem.\n"
              "       Renders are probably NOT named by source-image stem "
              "(expected e.g. 00188.png). Send render.py and I'll fix the matching.")
        return

    psnr_g, ssim_g, lpips_g = [], [], []      # global / full-image
    psnr_o, ssim_o, lpips_o = [], [], []      # object / masked

    for name in tqdm(common, desc=f"[{args.exp_name}/{args.scene}]"):
        gt   = _load_rgb(gt_map[name], device)        # [1,3,H,W]
        pred = _load_rgb(rnd_map[name], device)
        mask = _load_mask(mask_map[name], device)     # [1,1,H,W]

        _, _, H, W = gt.shape
        pred = _resize_to(pred, H, W, "bilinear").clamp(0, 1)
        mask = (_resize_to(mask, H, W, "nearest") > 0.5).float()

        # ---- Global (full image) -----------------------------------------
        psnr_g.append(psnr(pred, gt).item())
        ssim_g.append(ssim(pred, gt).item())
        lpips_g.append(lpips_orig(pred, gt).item())   # AuraFusion feeds [0,1]

        # ---- Object (pixel-wise mask) ------------------------------------
        psnr_o.append(masked_psnr(pred, gt, mask).item())
        lmap = lpips_spat(pred, gt)                    # [1,1,H,W], same HxW
        lpips_o.append((torch.sum(lmap * mask) / (torch.sum(mask) + 1e-8)).item())
        mask3 = mask.repeat(1, 3, 1, 1)
        ssim_o.append(ssim(pred * mask3, gt * mask3).item())

        # ---- FID (full image, whole matched test set) --------------------
        fid.update((gt   * 255).clamp(0, 255).to(torch.uint8), real=True)
        fid.update((pred * 255).clamp(0, 255).to(torch.uint8), real=False)

    fid_score = fid.compute().item()

    result = {
        "exp_name": args.exp_name,
        "scene":    args.scene,
        "protocol": "aurafusion360",
        "n_eval":   len(common),
        "global": {                                   # full-image
            "FID":   round(fid_score, 6),
            "LPIPS": round(float(np.mean(lpips_g)), 6),
            "PSNR":  round(float(np.mean(psnr_g)),  6),
            "SSIM":  round(float(np.mean(ssim_g)),  6),
        },
        "masked": {                                   # within object mask (Table-1 comparable)
            "FID":   round(fid_score, 6),             # same full-image FID (Aura has one)
            "LPIPS": round(float(np.mean(lpips_o)), 6),
            "PSNR":  round(float(np.mean(psnr_o)),  6),
            "SSIM":  round(float(np.mean(ssim_o)),  6),
        },
    }

    save_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{args.scene}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  [object] PSNR {result['masked']['PSNR']:.3f} | "
          f"LPIPS {result['masked']['LPIPS']:.4f} | SSIM {result['masked']['SSIM']:.4f}")
    print(f"  [global] PSNR {result['global']['PSNR']:.3f} | "
          f"LPIPS {result['global']['LPIPS']:.4f} | SSIM {result['global']['SSIM']:.4f} | "
          f"FID {fid_score:.3f}")
    print(f"Saved json -> {json_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--render_img_dir", type=str, required=True)
    parser.add_argument("--gt_img_dir",     type=str, required=True)
    parser.add_argument("--mask_dir",       type=str, required=True,
                        help="360-USID: test_object_masks/")
    parser.add_argument("--output_dir",     type=str, required=True,
                        help="Root output dir, e.g. metric_logs/")
    parser.add_argument("--exp_name",       type=str, required=True)
    parser.add_argument("--scene",          type=str, required=True)
    args = parser.parse_args()
    evaluate_metrics(args)