"""
grid_search.py — FID-constrained Optuna search for the SPIn-NeRF inpainting pipeline.
=====================================================================================
目標：在「保住低 m-FID（你的 SOTA 優勢）」的前提下，壓低 m-LPIPS、提升 m-PSNR。

為什麼不是只最小化 m-LPIPS：
  前面實驗證明 FID 與 LPIPS 在本系統是對著拉的（Gaussian 越多→越貼 inpaint→FID↓ LPIPS↑）。
  只追 LPIPS 會把 FID 優勢賣掉。因此改成「FID 當約束、LPIPS/PSNR 當目標」的純量化目標。

Objective (minimise):
    J = m_LPIPS  -  ALPHA_PSNR * m_PSNR  +  LAMBDA_FID * max(0, m_FID - FID_BUDGET)

  - m-LPIPS：主目標
  - m-PSNR：次目標（ALPHA_PSNR 控制每 dB 的價值）
  - m-FID ：超過 FID_BUDGET 才被罰 → 把 FID 鎖在優勢區間，不會為了 LPIPS 把 FID 賣掉

兩個 search mode：
  --mode train_only   只搜 training 參數（沿用既有 inpainting，Steps 4-6）。~7 min/trial。先跑這個。
  --mode full         固定 train（用 phase1 best），只搜 inpaint 參數（Steps 1-6）。較慢。

重要：--n_trials 現在代表「總試驗數」。重跑會接續到該總數，不會重頭加總，
方便放著跑一週、中斷再續（SQLite 持久化）。

Output:
    grid_search_results/{study_name}/
    ├── best_config.yaml     ← 目前最佳參數 → 直接餵 run_spinnerf.sh
    ├── best_result.json     ← 最佳 J + 全部 masked/global 指標
    └── all_trials.json      ← 每個 trial（事後分析用）
    └── optuna_{study_name}.db
"""

import os
import sys
import json
import copy
import math
import shutil
import subprocess
import argparse

import yaml
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from calculate_avg_metrics import compute_avg_for_exp
except Exception:  # 讓 import 失敗時 smoke-test 還能跑（實機上一定 import 得到）
    compute_avg_for_exp = None


# ============================================================================
# ★★★  目標權重 — 想調整三個指標的取捨，只改這四個常數  ★★★
# ============================================================================
FID_BUDGET   = 150.0    # masked FID 上限（≈ 你 best 的 146.84）；avg m-FID 超過才開始罰
LAMBDA_FID   = 0.005    # FID 每超標 +1 的懲罰（+20 ≈ 0.10，等同一次明顯的 LPIPS 退步）
ALPHA_PSNR   = 0.015    # 每 +1 dB m-PSNR 抵 0.015 m-LPIPS（PSNR 為次要目標）
FAIL_PENALTY = 1.0e6    # 失敗 / 缺指標 / NaN 的 trial 回傳值


def _scalarize(m_lpips, m_psnr, m_fid):
    """把三個 masked 指標壓成單一要最小化的 J。"""
    vals = (m_lpips, m_psnr, m_fid)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        return FAIL_PENALTY
    fid_pen = LAMBDA_FID * max(0.0, float(m_fid) - FID_BUDGET)
    return float(m_lpips) - ALPHA_PSNR * float(m_psnr) + fid_pen


# ============================================================================
# Search space — 依前面所有實驗的證據設計
#   * dead_weight 只在「低」區間（高會傷 FID，fullsup 實驗已證）
#   * dw_dead_scale 偏向保留 w_dyn（保護 FID）
#   * lpips_mask_prob：之前從未乾淨測過、理論上唯一能對 masked LPIPS 有效的旋鈕 → 納入
#   * max_grad/densify_until：FID↔LPIPS 取捨主旋鈕；範圍涵蓋「更少 Gaussian」(LPIPS 較好) 那側
# ============================================================================
TRAIN_PARAMS = {
    "loss_lpips_w":    (0.05, 0.45),
    "loss_l1_w":       (0.40, 0.80),
    "dw_alpha":        (2.0, 12.0),                        # log
    "dw_dead_scale":   [0.25, 0.5, 0.75, 1.0],
    "dead_weight":     [0.2, 0.3, 0.4, 0.5],              # 封頂 0.5（1.0 已知炸 FID）
    "lpips_mask_prob": [0.0, 0.25, 0.5, 0.75],
    "densify_until":   [6000, 8000, 10000, 12000, 15000],
    "max_grad":        (0.0002, 0.002),                   # log，兩側都開大＝Gaussian 數量控制更激進
}

INPAINT_PARAMS = {
    "phot_z_thresh":   (1.5, 5.0),
    "src_dilation_px": [3, 7, 11, 15, 19, 23],
    "tgt_dilation_px": [1, 3, 5, 7, 9, 11],
}


def _suggest_params(trial: optuna.Trial, mode: str) -> dict:
    p = {}
    if mode == "train_only":
        p["loss_lpips_w"]    = trial.suggest_float("loss_lpips_w", *TRAIN_PARAMS["loss_lpips_w"])
        p["loss_l1_w"]       = trial.suggest_float("loss_l1_w",   *TRAIN_PARAMS["loss_l1_w"])
        p["loss_ssim_w"]     = round(max(0.05, 1.0 - p["loss_lpips_w"] - p["loss_l1_w"]), 4)
        p["dw_alpha"]        = trial.suggest_float("dw_alpha", *TRAIN_PARAMS["dw_alpha"], log=True)
        p["dw_dead_scale"]   = trial.suggest_categorical("dw_dead_scale", TRAIN_PARAMS["dw_dead_scale"])
        p["dead_weight"]     = trial.suggest_categorical("dead_weight",   TRAIN_PARAMS["dead_weight"])
        p["lpips_mask_prob"] = trial.suggest_categorical("lpips_mask_prob", TRAIN_PARAMS["lpips_mask_prob"])
        p["densify_until"]   = trial.suggest_categorical("densify_until", TRAIN_PARAMS["densify_until"])
        p["max_grad"]        = trial.suggest_float("max_grad", *TRAIN_PARAMS["max_grad"], log=True)
    elif mode == "full":
        # train 參數由 base_config（= phase1 best_config.yaml）固定，這裡只搜 inpaint
        p["phot_z_thresh"]   = trial.suggest_float("phot_z_thresh", *INPAINT_PARAMS["phot_z_thresh"])
        p["src_dilation_px"] = trial.suggest_categorical("src_dilation_px", INPAINT_PARAMS["src_dilation_px"])
        p["tgt_dilation_px"] = trial.suggest_categorical("tgt_dilation_px", INPAINT_PARAMS["tgt_dilation_px"])
    return p


# ============================================================================
# Helpers
# ============================================================================
def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _build_trial_config(base_config: dict, trial_params: dict, trial_name: str) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("experiment", {})["name"] = trial_name

    train_keys   = set(TRAIN_PARAMS.keys()) | {"loss_ssim_w"}
    inpaint_keys = set(INPAINT_PARAMS.keys())

    for key, val in trial_params.items():
        if key in train_keys:
            cfg.setdefault("train", {})[key] = val
        elif key in inpaint_keys:
            cfg.setdefault("eval_iggt", {})[key] = val
    return cfg


def _run_trial(config_path: str, exp_name: str, mode: str, timeout_sec: int):
    env = os.environ.copy()
    env["CONFIG"]   = config_path
    env["EXP_NAME"] = exp_name
    env["MODE"]     = mode
    result = subprocess.run(["bash", "grid_spinnerf.sh"], env=env, timeout=timeout_sec)
    if result.returncode != 0:
        raise RuntimeError(f"grid_spinnerf.sh failed (returncode={result.returncode})")


# ============================================================================
# Objective
# ============================================================================
def make_objective(base_config, mode, metric_dir, results_dir, timeout_sec):

    best_path       = os.path.join(results_dir, "best_result.json")
    all_trials_path = os.path.join(results_dir, "all_trials.json")

    def objective(trial: optuna.Trial) -> float:
        trial_name  = f"trial_{trial.number:04d}"
        cfg_path    = os.path.join(results_dir, "configs", f"{trial_name}.yaml")

        trial_params = _suggest_params(trial, mode)
        trial_cfg    = _build_trial_config(base_config, trial_params, trial_name)
        _save_yaml(trial_cfg, cfg_path)

        print(f"\n{'='*60}\n  Trial {trial.number:04d}  [{mode}]\n"
              f"  {json.dumps(trial_params, ensure_ascii=False)}\n{'='*60}")

        # ── 跑 pipeline（任何失敗都回傳 penalty，不讓 study 崩） ──
        try:
            _run_trial(cfg_path, trial_name, mode, timeout_sec)
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  Trial timed out (> {timeout_sec}s) → penalty")
            return FAIL_PENALTY
        except Exception as e:
            print(f"  ❌ Pipeline failed: {e} → penalty")
            return FAIL_PENALTY

        # ── 讀指標 ──
        if compute_avg_for_exp is None:
            print("  ❌ compute_avg_for_exp unavailable → penalty")
            return FAIL_PENALTY
        try:
            avg = compute_avg_for_exp(metric_dir, trial_name)
        except Exception as e:
            print(f"  ❌ metric read error: {e} → penalty")
            return FAIL_PENALTY
        if not avg or "masked" not in avg:
            print(f"  ❌ no metrics for {trial_name} → penalty")
            return FAIL_PENALTY

        m_lpips = avg["masked"].get("LPIPS", float("nan"))
        m_psnr  = avg["masked"].get("PSNR",  float("nan"))
        m_fid   = avg["masked"].get("FID",   float("nan"))
        J = _scalarize(m_lpips, m_psnr, m_fid)

        fid_flag = "✅" if m_fid <= FID_BUDGET else f"⚠️(>{FID_BUDGET:.0f})"
        print(f"  → m-LPIPS={m_lpips:.4f}  m-PSNR={m_psnr:.4f}  m-FID={m_fid:.2f} {fid_flag}  |  J={J:.4f}")

        # ── 存進 trial（供事後 Pareto 分析） ──
        trial.set_user_attr("m_lpips", float(m_lpips) if math.isfinite(m_lpips) else None)
        trial.set_user_attr("m_psnr",  float(m_psnr)  if math.isfinite(m_psnr)  else None)
        trial.set_user_attr("m_fid",   float(m_fid)   if math.isfinite(m_fid)   else None)
        trial.set_user_attr("config_path", cfg_path)

        # ── 累積 all_trials.json（best-effort log） ──
        record = {
            "trial": trial.number, "trial_name": trial_name, "J": J,
            "params": trial_params, "n_scenes": avg.get("n_scenes"),
            "masked": avg.get("masked"), "global": avg.get("global"),
            "config_path": cfg_path,
        }
        try:
            allt = []
            if os.path.exists(all_trials_path):
                with open(all_trials_path) as f:
                    allt = json.load(f)
            allt.append(record)
            with open(all_trials_path, "w") as f:
                json.dump(allt, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  (warn) all_trials.json write failed: {e}")

        # ── 更新 best（用 J；檔案式，斷線也安全） ──
        is_best = True
        if os.path.exists(best_path):
            try:
                with open(best_path) as f:
                    is_best = J < json.load(f).get("J", float("inf"))
            except Exception:
                is_best = True
        if is_best:
            print(f"  🏆 New best!  J={J:.4f}  (m-LPIPS={m_lpips:.4f}, m-PSNR={m_psnr:.4f}, m-FID={m_fid:.2f})")
            with open(best_path, "w") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            shutil.copy(cfg_path, os.path.join(results_dir, "best_config.yaml"))

        return J

    return objective


# ============================================================================
# Pareto report
# ============================================================================
def _print_summary(study):
    rows = []
    for t in study.get_trials(deepcopy=False):
        if t.state != TrialState.COMPLETE:
            continue
        ua = t.user_attrs
        if ua.get("m_lpips") is None:
            continue
        rows.append((t.number, ua["m_lpips"], ua["m_psnr"], ua["m_fid"], t.value))
    if not rows:
        print("  (沒有成功的 trial)")
        return

    print(f"\n  ── Top-5 by J ──")
    for n, l, p, f, j in sorted(rows, key=lambda r: r[4])[:5]:
        print(f"    #{n:04d}  J={j:.4f}  m-LPIPS={l:.4f}  m-PSNR={p:.4f}  m-FID={f:.2f}")

    # Pareto front: minimise LPIPS, maximise PSNR, minimise FID
    pareto = []
    for (n, l, p, f, j) in rows:
        dominated = False
        for (_, l2, p2, f2, _) in rows:
            if (l2 <= l and p2 >= p and f2 <= f) and (l2 < l or p2 > p or f2 < f):
                dominated = True
                break
        if not dominated:
            pareto.append((n, l, p, f, j))
    print(f"\n  ── Pareto front ({len(pareto)} 個非支配解；想換取捨可從這裡挑) ──")
    for n, l, p, f, j in sorted(pareto, key=lambda r: r[1]):
        print(f"    #{n:04d}  m-LPIPS={l:.4f}  m-PSNR={p:.4f}  m-FID={f:.2f}  (J={j:.4f})")


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", required=True, help="Base YAML，e.g. configs/exp_baseline.yaml")
    ap.add_argument("--mode", default="train_only", choices=["train_only", "full"])
    ap.add_argument("--n_trials", type=int, default=300, help="總試驗數（重跑會接續到此總數）")
    ap.add_argument("--metric_dir", default="metric_logs_test")
    ap.add_argument("--study_name", default="grid_search")
    ap.add_argument("--timeout_sec", type=int, default=14400, help="單 trial 逾時（秒），預設 4h")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results_dir = os.path.join("grid_search_results", args.study_name)
    os.makedirs(os.path.join(results_dir, "configs"), exist_ok=True)

    db_path = os.path.join(results_dir, f"optuna_{args.study_name}.db")
    storage = f"sqlite:///{db_path}"
    base_config = _load_yaml(args.base_config)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=TPESampler(seed=args.seed, n_startup_trials=20, multivariate=True, group=True),
        storage=storage,
        load_if_exists=True,
    )

    done      = len(study.get_trials(deepcopy=False))
    remaining = max(0, args.n_trials - done)

    print(f"\n{'='*60}")
    print(f"  Grid Search : {args.study_name}")
    print(f"  Mode        : {args.mode}")
    print(f"  Objective   : min J = LPIPS - {ALPHA_PSNR}*PSNR + {LAMBDA_FID}*max(0, FID-{FID_BUDGET})")
    print(f"  Trials      : {done} done / {args.n_trials} target → {remaining} to run")
    print(f"  Output      : {results_dir}/")
    print(f"{'='*60}\n")

    if remaining > 0:
        objective = make_objective(base_config, args.mode, args.metric_dir, results_dir, args.timeout_sec)
        study.optimize(objective, n_trials=remaining, show_progress_bar=False, catch=(Exception,))
    else:
        print("  已達目標試驗數，無需再跑。")

    print(f"\n{'='*60}")
    try:
        best = study.best_trial
        print(f"  🏆 BEST Trial #{best.number}   J={best.value:.4f}")
        ua = best.user_attrs
        if ua.get("m_lpips") is not None:
            print(f"     m-LPIPS={ua['m_lpips']:.4f}  m-PSNR={ua['m_psnr']:.4f}  m-FID={ua['m_fid']:.2f}")
        print(f"     Params : {json.dumps(best.params, ensure_ascii=False)}")
        print(f"     Config : {results_dir}/best_config.yaml")
    except Exception as e:
        print(f"  (尚無最佳 trial: {e})")
    _print_summary(study)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()