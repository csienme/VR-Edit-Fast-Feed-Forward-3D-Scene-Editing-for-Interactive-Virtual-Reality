#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt

set -e

SCENE="${1:?需要 scene name}"
RGB_DIR="${2:?需要 RGB 圖片目錄（絕對路徑或相對路徑）}"
MASK_DIR="${3:?需要 mask 目錄（白=要 inpaint 的物體區）}"

# 例：../data/Other-360/pinecone/images -> Other-360 / pinecone
DATASET_NAME=$(basename "$(dirname "$(dirname "${RGB_DIR}")")")
SCENE_NAME=$(basename "$(dirname "${RGB_DIR}")")

BASE_OUT="eval_results_custom/${DATASET_NAME}/${SCENE}"
INPAINTED_DIR="${BASE_OUT}/inpainted"
DEADMASK_DIR="${BASE_OUT}/deadmasks"
COLMAP_DIR="${BASE_OUT}/colmap"
RENDER_DIR="${BASE_OUT}/renders"

echo "============================================================"
echo "🚀 場景: ${SCENE}"
echo "   Dataset: ${DATASET_NAME}"
echo "   RGB:     ${RGB_DIR}"
echo "   Mask:    ${MASK_DIR}"
echo "   Output:  ${BASE_OUT}"
echo "============================================================"

mkdir -p "${BASE_OUT}"

# ── Step 1: VGGT inpainting ─────────────────────────────────
echo ""
echo "[${SCENE}] Step 1: VGGT inpainting all frames..."
python eval/eval_custom_360.py \
    --data_path "${RGB_DIR}" \
    --mask_path "${MASK_DIR}" \
    --enable_gen_3d_prop \
    --generate "all frame" \
    --exp_name "${SCENE}" \
    --inpaint_method sd \
    --output_root "${BASE_OUT}"

# ── Step 2: 建立單一 COLMAP（從 inpainted 圖）────────────────
echo ""
echo "[${SCENE}] Step 2: Building single COLMAP from inpainted frames..."
python eval/eval_custom_colmap_masked.py \
    --data_path "${INPAINTED_DIR}" \
    --output_path "${COLMAP_DIR}"

# ── Step 3: 準備 train_render 要讀的圖 ───────────────────────
echo ""
echo "[${SCENE}] Step 3: Copying inpainted images into colmap/images..."
mkdir -p "${COLMAP_DIR}/images"
cp "${INPAINTED_DIR}"/inpainted_*.png "${COLMAP_DIR}/images/"

# ── Step 4: 3DGS training + render ──────────────────────────
echo ""
echo "[${SCENE}] Step 4: 3DGS training + rendering..."
if [ ! -d "${DEADMASK_DIR}" ]; then
    echo "  ⚠️  DEADMASK_DIR 不存在：${DEADMASK_DIR}，將以 uniform loss 繼續"
    DEADMASK_ARG=""
else
    n_dm=$(ls "${DEADMASK_DIR}"/*.png 2>/dev/null | wc -l)
    echo "  💀 Dead masks found: ${n_dm} files in ${DEADMASK_DIR}"
    DEADMASK_ARG="--deadmask_dir ${DEADMASK_DIR}"
fi

python train_render_360.py \
    --colmap_dir    "${COLMAP_DIR}" \
    --nvs_pose      "${COLMAP_DIR}" \
    --train_img_dir "${COLMAP_DIR}/images" \
    --deadmask_dir  "${DEADMASK_DIR}" \
    --output_dir    "${RENDER_DIR}" \
    --total_iters   20000 \
    --dead_weight   0.3 \
    --patch_size    256

echo ""
echo "============================================================"
echo "✅ 場景 ${SCENE} 完成！"
echo "   Inpainted:  ${INPAINTED_DIR}/"
echo "   Deadmasks:  ${DEADMASK_DIR}/"
echo "   COLMAP:     ${COLMAP_DIR}/"
echo "   Renders:    ${RENDER_DIR}/"
echo "============================================================"