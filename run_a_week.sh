#!/bin/bash
# =============================================================
# run_a_week.sh — 放著跑：一次性 inpaint(若無) → Phase 1 → Phase 2
# =============================================================
# 用法：
#   nohup bash run_a_week.sh > /dev/null 2>&1 &
#   tail -f grid_run_*.log          # 看進度
#
# 全程可重入：中斷後再執行同一行，會自動接續（inpaint 跳過、phase 續跑到目標 trial 數）。
# =============================================================

set -u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt


ANCHOR="configs/exp_baseline.yaml"
LOG="grid_run_$(date +%Y%m%d_%H%M%S).log"

# 想跑久一點就把 PHASE1_TRIALS 調大（TPE 約 150-250 trial 就收斂，再多是微調，無害）
PHASE1_TRIALS=300
PHASE2_TRIALS=25

{
echo "######## START $(date) ########"

# ---- optuna 在不在 fastvggt env ----
python -c "import optuna" 2>/dev/null || pip install optuna --quiet

# ---- 0) 一次性 inpainting (Steps 1-3)，只在缺的時候跑 ----
echo "===== [0] one-time inpainting check ====="
NEED=0
for S in 1 2 3 4 7 9 10 12 book trash; do
    [ -d "eval_results_custom/spinnerf-dataset/${S}/inpainted" ] || NEED=1
    [ -d "eval_results_custom/spinnerf-dataset/${S}/colmap" ]    || NEED=1
done
if [ "${NEED}" = "1" ]; then
    echo "inpainting/colmap 缺 → 跑 Steps 1-3 (SD，會花一段時間)..."
    CONFIG="${ANCHOR}" function=1 bash run_spinnerf.sh
    CONFIG="${ANCHOR}" function=2 bash run_spinnerf.sh
    CONFIG="${ANCHOR}" function=3 bash run_spinnerf.sh
else
    echo "inpainting/colmap 已存在 → 跳過。"
fi

# ---- 1) Phase 1: train_only（主力，快）----
echo "===== [1] Phase 1: train_only (target ${PHASE1_TRIALS} trials) ====="
python grid_search.py \
    --base_config "${ANCHOR}" \
    --mode        train_only \
    --n_trials    "${PHASE1_TRIALS}" \
    --study_name  phase1_train \
    --timeout_sec 14400

# ---- 2) Phase 2: full（固定 phase1 best，只搜 inpaint 的 phot_z_thresh）----
P1_BEST="grid_search_results/phase1_train/best_config.yaml"
if [ -f "${P1_BEST}" ]; then
    echo "===== [2] Phase 2: full on phase1 best (target ${PHASE2_TRIALS} trials) ====="
    python grid_search.py \
        --base_config "${P1_BEST}" \
        --mode        full \
        --n_trials    "${PHASE2_TRIALS}" \
        --study_name  phase2_full \
        --timeout_sec 21600
else
    echo "找不到 phase1 best_config，跳過 Phase 2。"
fi

echo "######## ALL DONE $(date) ########"
echo "最終最佳設定："
echo "  Phase 1: grid_search_results/phase1_train/best_config.yaml"
echo "  Phase 2: grid_search_results/phase2_full/best_config.yaml"

} >> "${LOG}" 2>&1