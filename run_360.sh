#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt

set -e

SCENE="${1:?需要 scene name}"
RGB_DIR="${2:?需要 RGB 圖片目錄（絕對路徑或相對路徑）}"
MASK_DIR="${3:?需要 mask 目錄（白=要 inpaint 的物體區）}"

# 可選：指定只跑某一階段
# 用法：
#   bash run_360.sh kitchen ../data/Other-360/kitchen/images ../data/Other-360/kitchen/object_mask
#   function="4" bash run_360.sh kitchen ../data/Other-360/kitchen/images ../data/Other-360/kitchen/object_mask
#
# 預設值：all（代表 1~4 全部執行）
RUN_STAGE="${function:-all}"

should_run () {
    local target="$1"
    [[ "${RUN_STAGE}" == "all" || "${RUN_STAGE}" == "${target}" ]]
}

# 輸出結構:
#   eval_results_custom/{DATASET_NAME}/{SCENE}/
#   ├── inpainted/       ← VGGT inpainting 結果
#   ├── deadmasks/       ← dead zone masks
#   ├── colmap/          ← COLMAP sparse reconstruction + images/
#   └── renders/         ← 3DGS 渲染結果

DATASET_NAME=$(basename "$(dirname "$(dirname "${RGB_DIR}")")")

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
echo "   Stage:   ${RUN_STAGE}"
echo "   Structure:"
echo "     inpainted/ → ${INPAINTED_DIR}"
echo "     deadmasks/ → ${DEADMASK_DIR}"
echo "     colmap/    → ${COLMAP_DIR}"
echo "     renders/   → ${RENDER_DIR}"
echo "============================================================"

mkdir -p "${BASE_OUT}"

# ── Step 1: VGGT inpainting ─────────────────────────────────
if should_run 1; then
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
fi

# ── Step 2: 建立 COLMAP（從 inpainted 圖）────────────────────
if should_run 2; then
    echo ""
    echo "[${SCENE}] Step 2: Building COLMAP from inpainted frames..."
    python eval/eval_custom_colmap_masked.py \
        --data_path "${INPAINTED_DIR}" \
        --output_path "${COLMAP_DIR}"
fi

# ── Step 3: 把 inpainted 圖複製進 colmap/images/ ─────────────
if should_run 3; then
    echo ""
    echo "[${SCENE}] Step 3: Copying inpainted images into colmap/images/..."
    mkdir -p "${COLMAP_DIR}/images"
    cp "${INPAINTED_DIR}"/inpainted_*.png "${COLMAP_DIR}/images/"
fi

# ── Step 4: 3DGS training + render ──────────────────────────
if should_run 4; then
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

    mkdir -p "${RENDER_DIR}"

    python train_render_360.py \
        --colmap_dir    "${COLMAP_DIR}" \
        --nvs_pose      "${COLMAP_DIR}" \
        --train_img_dir "${COLMAP_DIR}/images" \
        ${DEADMASK_ARG} \
        --output_dir "${RENDER_DIR}" \
        --total_iters 20000 \
        --dead_weight 0.3 \
        --patch_size 256 \
        --dw_alpha 7.0 \
        --dw_warmup 1500
fi
echo ""
echo "============================================================"
echo "✅ 場景 ${SCENE} 完成！"
echo "   Inpainted:  ${INPAINTED_DIR}/"
echo "   Deadmasks:  ${DEADMASK_DIR}/"
echo "   COLMAP:     ${COLMAP_DIR}/"
echo "   Renders:    ${RENDER_DIR}/"
echo "============================================================"