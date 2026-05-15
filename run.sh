#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt

set -e

SCENE="${1:?需要 scene name}"
RGB_DIR="${2:?需要 RGB 圖片目錄（絕對路徑或相對路徑）}"
MASK_DIR="${3:?需要 mask 目錄（白=要 inpaint 的物體區）}"

# 可選：指定只跑某一階段
# 用法：
#   bash run.sh kitchen ../data/Other-360/kitchen/images ../data/Other-360/kitchen/object_masks
#   function="4" bash run_360.sh kitchen ...
#
# 預設值：all（1 和 4 全部執行）
# Step 2 和 Step 3 已移除：COLMAP 現在由 Step 1 直接產生
RUN_STAGE="${function:-all}"

should_run () {
    local target="$1"
    [[ "${RUN_STAGE}" == "all" || "${RUN_STAGE}" == "${target}" ]]
}

# 輸出結構:
#   eval_results_custom/{DATASET_NAME}/{SCENE}/
#   ├── inpainted/       ← VGGT inpainting 結果
#   ├── deadmasks/       ← dead zone masks
#   ├── colmap/          ← COLMAP sparse + images/（由 Step 1 直接產生）
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
echo "     colmap/    → ${COLMAP_DIR}  (由 Step 1 直接產生，無需 Step 2/3)"
echo "     renders/   → ${RENDER_DIR}"
echo "============================================================"

mkdir -p "${BASE_OUT}"

# ── Step 1: VGGT inpainting + COLMAP export ──────────────────
# eval_iggt.py 在 inpainting 完成後自動產生：
#   - colmap/sparse/（cameras.bin, images.bin, points3D.bin）
#   - colmap/images/（inpainted_*.png 的 symlink/copy）
# Step 2（eval_custom_colmap_masked.py）和 Step 3（cp）已不再需要
if should_run 1; then
    echo ""
    echo "[${SCENE}] Step 1: VGGT inpainting + COLMAP export..."
    python eval/eval_iggt.py \
        --data_path "${RGB_DIR}" \
        --mask_path "${MASK_DIR}" \
        --enable_gen_3d_prop \
        --generate "all frame" \
        --exp_name "${SCENE}" \
        --inpaint_method sd \
        --output_root "${BASE_OUT}"
fi

# ── Step 4: 3DGS training + render ──────────────────────────
if should_run 4; then
    echo ""
    echo "[${SCENE}] Step 4: 3DGS training + rendering..."

    if [ ! -d "${DEADMASK_DIR}" ]; then
        echo "  ⚠️  DEADMASK_DIR 不存在：${DEADMASK_DIR}，將以 uniform loss 繼續"
    else
        n_dm=$(ls "${DEADMASK_DIR}"/*.png 2>/dev/null | wc -l)
        echo "  💀 Dead masks found: ${n_dm} files in ${DEADMASK_DIR}"
    fi

    mkdir -p "${RENDER_DIR}"

    python train.py \
        --colmap_dir      "${COLMAP_DIR}" \
        --train_img_dir   "${COLMAP_DIR}/images" \
        --deadmask_dir    "${DEADMASK_DIR}" \
        --output_gaussian "${RENDER_DIR}/gaussians.pth" \
        --total_iters     20000 \
        --dead_weight     0.3 \
        --patch_size      256

    # Bug fix: gaussian_path 指向 train_.py 的輸出（RENDER_DIR），不是 COLMAP_DIR
    python render.py \
        --nvs_pose          "${COLMAP_DIR}" \
        --gaussian_path     "${RENDER_DIR}/gaussians.pth" \
        --render_output_dir "${RENDER_DIR}/renders"
fi

echo ""
echo "============================================================"
echo "✅ 場景 ${SCENE} 完成！"
echo "   Inpainted:  ${INPAINTED_DIR}/"
echo "   Deadmasks:  ${DEADMASK_DIR}/"
echo "   COLMAP:     ${COLMAP_DIR}/"
echo "   Renders:    ${RENDER_DIR}/"
echo "============================================================"