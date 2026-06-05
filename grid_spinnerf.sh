#!/bin/bash
# =============================================================
# grid_spinnerf.sh — Grid search pipeline runner
# =============================================================
# 專門給 grid_search.py 呼叫，不要直接執行。
# 固定跑全部 10 個 scene，每個 scene 跑完 metrics 後立即清除中間產物。
#
# 必要環境變數（由 grid_search.py 注入）：
#   CONFIG    — trial YAML 路徑
#   EXP_NAME  — trial 名稱（作為 metric 的 exp_name）
#   MODE      — train_only | full
#
# ★ 修正：Step 1 不再 hardcode --inpaint_method lama，改用 YAML 裡的 inpaint_method（sd）。
# =============================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt

set -e

CONFIG_FILE="${CONFIG:?需要設定 CONFIG 環境變數}"
EXP_NAME="${EXP_NAME:?需要設定 EXP_NAME 環境變數}"
MODE="${MODE:?需要設定 MODE 環境變數 (train_only 或 full)}"

DATASET_ROOT="../spinnerf-dataset"
DATASET_NAME="$(basename "${DATASET_ROOT}")"
OUTPUT_ROOT="./eval_results_custom"
METRIC_ROOT="./metric_logs_test"

# 想讓 trial 跑更快、訊號更乾淨，可在這裡拿掉 3、10（pose/init 壞、對參數不反應）。
# 預設保留全 10 場景（= 你論文要報的平均）。
SCENES=("1" "2" "3" "4" "7" "9" "10" "12" "book" "trash")

echo "========================================================"
echo "  Trial  : ${EXP_NAME}"
echo "  Mode   : ${MODE}"
echo "  Config : ${CONFIG_FILE}"
echo "========================================================"

for SCENE in "${SCENES[@]}"; do

    RGB_DIR="${DATASET_ROOT}/${SCENE}/images_4"
    MASK_DIR="${RGB_DIR}/label"
    BASE_OUT="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}"
    INPAINTED_DIR="${BASE_OUT}/inpainted"
    DEADMASK_DIR="${BASE_OUT}/deadmasks"
    COLMAP_DIR="${BASE_OUT}/colmap"
    NVS_COLMAP_DIR="${BASE_OUT}/nvs_colmap"
    RENDER_DIR="${BASE_OUT}/renders"
    MERGED_DIR="./merged_data/${SCENE}"

    echo ""
    echo "[${SCENE}] ────────────────────────────────────────"
    mkdir -p "${BASE_OUT}"

    # ── full mode only: Steps 1-3 ─────────────────────────────
    if [[ "${MODE}" == "full" ]]; then

        echo "[${SCENE}] Step 1: Inpainting..."
        python eval/eval_iggt.py \
            --config         "${CONFIG_FILE}" \
            --data_path      "${RGB_DIR}" \
            --mask_path      "${MASK_DIR}" \
            --exp_name       "${SCENE}" \
            --output_root    "${BASE_OUT}"
        # ↑ 不再傳 --inpaint_method；改由 YAML 的 inpaint_method (sd) 決定

        echo "[${SCENE}] Step 2: Merging GT + inpainted..."
        mkdir -p "${MERGED_DIR}"
        ls "${RGB_DIR}"/*.png | sort | head -n 40 | xargs -I {} cp {} "${MERGED_DIR}/"
        cp "${INPAINTED_DIR}"/*.png "${MERGED_DIR}/"

        echo "[${SCENE}] Step 3: NVS COLMAP..."
        python eval/eval_custom_colmap_masked.py \
            --data_path   "${MERGED_DIR}" \
            --output_path "${NVS_COLMAP_DIR}"
        rm -rf "${MERGED_DIR}"
    fi

    # ── Steps 4-6 (both modes) ────────────────────────────────
    echo "[${SCENE}] Step 4: 3DGS training..."
    mkdir -p "${RENDER_DIR}"
    python train.py \
        --config          "${CONFIG_FILE}" \
        --colmap_dir      "${COLMAP_DIR}" \
        --train_img_dir   "${COLMAP_DIR}/images" \
        --deadmask_dir    "${DEADMASK_DIR}" \
        --output_gaussian "${RENDER_DIR}/gaussians.pth"

    echo "[${SCENE}] Step 5: Rendering..."
    python render.py \
        --nvs_pose          "${NVS_COLMAP_DIR}" \
        --gaussian_path     "${RENDER_DIR}/gaussians.pth" \
        --render_output_dir "${RENDER_DIR}"

    echo "[${SCENE}] Step 6: Metrics..."
    python eval_metric_spinnerf_prtcl.py \
        --gt_img_dir     "${RGB_DIR}" \
        --render_img_dir "${RENDER_DIR}" \
        --mask_dir       "${RGB_DIR}/test_label" \
        --output_dir     "${METRIC_ROOT}" \
        --scene          "${SCENE}" \
        --exp_name       "${EXP_NAME}"

    # ── Cleanup ───────────────────────────────────────────────
    echo "[${SCENE}] Cleanup..."
    rm -rf "${RENDER_DIR}"
    if [[ "${MODE}" == "full" ]]; then
        rm -rf "${INPAINTED_DIR}" "${DEADMASK_DIR}" "${COLMAP_DIR}" "${NVS_COLMAP_DIR}"
    fi
    echo "[${SCENE}] ✅ Done."
done

echo ""
echo "========================================================"
echo "  🎉 Trial ${EXP_NAME} complete."
echo "========================================================"