"""
calculate_avg_metrics.py
=========================
Reads per-scene JSON files produced by eval_metric_spinnerf_prtcl.py,
computes 10-scene averages, and saves avg_metrics.json.

Usage (standalone):
    python calculate_avg_metrics.py \
        --metric_dir metric_logs_test \
        --exp_name   trial_0001

Usage (programmatic — called by grid_search.py):
    from calculate_avg_metrics import compute_avg_for_exp
    avg = compute_avg_for_exp("metric_logs_test", "trial_0001")
    print(avg["masked"]["LPIPS"])   # → 0.2341

Output:
    {metric_dir}/{exp_name}/avg_metrics.json
"""

import os
import json
import argparse


SCENES = ["1", "2", "3", "4", "7", "9", "10", "12", "book", "trash"]


def compute_avg_for_exp(metric_dir: str, exp_name: str) -> dict:
    """
    Read per-scene JSON files, compute averages, save avg_metrics.json.

    Returns:
        dict with keys 'global' and 'masked', each containing
        FID / LPIPS / PSNR / SSIM averages.
        Returns None if fewer than 1 scene has valid JSON.
    """
    base_path = os.path.join(metric_dir, exp_name)

    global_acc = {"FID": [], "LPIPS": [], "PSNR": [], "SSIM": []}
    masked_acc = {"FID": [], "LPIPS": [], "PSNR": [], "SSIM": []}
    scenes_found = []

    for scene in SCENES:
        json_path = os.path.join(base_path, f"{scene}_metrics.json")
        if not os.path.exists(json_path):
            print(f"  ⚠️  Missing: {json_path}")
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        for key in global_acc:
            global_acc[key].append(data["global"][key])
            masked_acc[key].append(data["masked"][key])
        scenes_found.append(scene)

    if not scenes_found:
        print(f"❌ No valid JSON files found under {base_path}")
        return None

    n = len(scenes_found)
    avg = {
        "exp_name":     exp_name,
        "n_scenes":     n,
        "scenes_found": scenes_found,
        "global": {k: round(sum(v) / len(v), 6) for k, v in global_acc.items() if v},
        "masked": {k: round(sum(v) / len(v), 6) for k, v in masked_acc.items() if v},
    }

    # Save
    out_path = os.path.join(base_path, "avg_metrics.json")
    with open(out_path, "w") as f:
        json.dump(avg, f, indent=2)

    return avg


def _print_avg(avg: dict):
    print(f"\n{'='*45}")
    print(f"  Exp   : {avg['exp_name']}")
    print(f"  Scenes: {avg['n_scenes']} / 10  {avg['scenes_found']}")
    print(f"{'='*45}")
    print("  [ Global ]")
    for k, v in avg["global"].items():
        print(f"    {k:<8}: {v:.4f}")
    print("  [ Masked BBox (SPIn-NeRF protocol) ]")
    for k, v in avg["masked"].items():
        print(f"    m-{k:<6}: {v:.4f}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_dir", type=str, required=True,
                        help="Root metric log dir, e.g. metric_logs_test")
    parser.add_argument("--exp_name",   type=str, required=True,
                        help="Experiment name, e.g. trial_0001")
    args = parser.parse_args()

    avg = compute_avg_for_exp(args.metric_dir, args.exp_name)
    if avg:
        _print_avg(avg)
        saved = os.path.join(args.metric_dir, args.exp_name, "avg_metrics.json")
        print(f"Saved → {saved}")