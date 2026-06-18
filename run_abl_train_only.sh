#!/bin/bash
# 重用既有 inpainted/colmap/nvs_colmap，只跑 Step 4(train)→5(render)→6(metric)
# 用法：CONFIG=configs/abl_only_wbase.yaml bash run_abl_train_only.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt
set -e

DATASET_ROOT="../spinnerf-dataset"
DATASET_NAME="$(basename "${DATASET_ROOT}")"
OUTPUT_ROOT="./eval_results_custom"
METRIC_ROOT="./metric_logs"
CONFIG_FILE="${CONFIG:?需要設定 CONFIG 環境變數}"

METRIC_EXP_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_FILE}')).get('experiment',{}).get('name','spinnerf_eval'))")
echo "  Metric exp_name: ${METRIC_EXP_NAME}"

SCENES=("1" "2" "3" "4" "7" "9" "10" "12" "book" "trash")

for SCENE in "${SCENES[@]}"; do
    RGB_DIR="${DATASET_ROOT}/${SCENE}/images_4"
    BASE_OUT="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}"
    COLMAP_DIR="${BASE_OUT}/colmap"
    NVS_COLMAP_DIR="${BASE_OUT}/nvs_colmap"
    DEADMASK_DIR="${BASE_OUT}/deadmasks"
    RENDER_DIR="${BASE_OUT}/renders"

    echo ""
    echo "=== [${SCENE}] 重用 inpaint/colmap，只跑 train→render→metric ==="

    # 防呆：確認重用的產物都在
    if [ ! -d "${COLMAP_DIR}/sparse" ] || [ ! -d "${NVS_COLMAP_DIR}" ]; then
        echo "  ❌ ${SCENE}: colmap 或 nvs_colmap 不存在，無法重用。請先用 golden_467 全跑一次。跳過。"
        continue
    fi

    mkdir -p "${RENDER_DIR}"

    echo "[${SCENE}] Step 4: 3DGS training..."
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
        --exp_name       "${METRIC_EXP_NAME}"

    echo "[${SCENE}] ✅ Done."
done

echo ""
echo "🎉 ${METRIC_EXP_NAME} 全部完成"