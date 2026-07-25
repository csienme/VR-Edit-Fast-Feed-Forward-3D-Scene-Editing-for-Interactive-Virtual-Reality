#!/bin/bash
# w/o attention bias ablation：vanilla VGGT，獨立 OUTPUT_ROOT，全跑 Step 1-6
# 前提：vggt/models/vggt.py 已換成 vanilla 版（不注入 attention bias）
#       eval/inpaint_module.py 已換回 FULL 版（pull harvest 正常）
# 用法：CONFIG=configs/abl_no_bias.yaml bash run_abl_no_bias.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt
set -e

DATASET_ROOT="../spinnerf-dataset"
DATASET_NAME="$(basename "${DATASET_ROOT}")"
OUTPUT_ROOT="./eval_results_abl_nobias"     # ← 獨立目錄，不碰 Full
METRIC_ROOT="./metric_logs"
CONFIG_FILE="${CONFIG:?需要設定 CONFIG 環境變數}"

METRIC_EXP_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_FILE}')).get('experiment',{}).get('name','spinnerf_eval'))")
echo "  Metric exp_name: ${METRIC_EXP_NAME}"
echo "  OUTPUT_ROOT: ${OUTPUT_ROOT}（獨立，不碰 golden_467）"

SCENES=("1" "2" "3" "4" "7" "9" "10" "12" "book" "trash")

for SCENE in "${SCENES[@]}"; do
    RGB_DIR="${DATASET_ROOT}/${SCENE}/images_4"
    MASK_DIR="${RGB_DIR}/label"
    BASE_OUT="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}"
    INPAINTED_DIR="${BASE_OUT}/inpainted"
    DEADMASK_DIR="${BASE_OUT}/deadmasks"
    COLMAP_DIR="${BASE_OUT}/colmap"
    NVS_COLMAP_DIR="${BASE_OUT}/nvs_colmap"
    RENDER_DIR="${BASE_OUT}/renders"
    MERGED_DIR="./merged_data_abl_nobias/${SCENE}"

    echo ""
    echo "=== [${SCENE}] w/o attention bias | OUT=${BASE_OUT} ==="
    mkdir -p "${BASE_OUT}"

    echo "[${SCENE}] Step 1: vanilla VGGT inpainting + COLMAP..."
    python eval/abl_bias_eval_iggt.py \
        --config "${CONFIG_FILE}" \
        --data_path "${RGB_DIR}" \
        --mask_path "${MASK_DIR}" \
        --exp_name "${SCENE}" \
        --output_root "${BASE_OUT}"

    echo "[${SCENE}] Step 2: Merging 40 GT + 60 inpainted..."
    mkdir -p "${MERGED_DIR}"
    ls "${RGB_DIR}"/*.png | sort | head -n 40 | xargs -I {} cp {} "${MERGED_DIR}/"
    cp "${INPAINTED_DIR}"/*.png "${MERGED_DIR}/"

    echo "[${SCENE}] Step 3: NVS COLMAP..."
    python eval/eval_custom_colmap_masked.py \
        --data_path "${MERGED_DIR}" \
        --output_path "${NVS_COLMAP_DIR}"
    rm -rf "${MERGED_DIR}"

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
        --exp_name       "${METRIC_EXP_NAME}"

    echo "[${SCENE}] ✅ Done."
done

echo ""
echo "🎉 ${METRIC_EXP_NAME} 全部完成（產物在 ${OUTPUT_ROOT}）"