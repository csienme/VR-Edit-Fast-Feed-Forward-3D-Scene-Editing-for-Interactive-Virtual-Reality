#!/bin/bash
# conda 環境啟動

# ============================================================
# SPInNeRF 全場景批次測試腳本
# 場景: 1 2 3 4 7 9 10 12 book trash
#
# 流程：
#   1) VGGT inpainting（跳過前 40 張，處理後 60 張）
#   2) 合併前 40 張 GT + 後 60 張 inpainted
#   3) 用合併圖建 nvs_pose COLMAP
#   4) 用純 inpainted 圖建 point cloud COLMAP
#   5) 將 inpainted 圖移到 purify_scene/images
#   6) 3DGS training + rendering
#   7) 評估 metric
# ============================================================

set -e

DATASET_ROOT="../spinnerf-dataset"
DATASET_NAME="$(basename "${DATASET_ROOT}")"
OUTPUT_ROOT="./eval_results_custom"
MERGED_ROOT="./merged_data"
PURIFY_HYBRID_ROOT="./purify_hybrid"
PURIFY_ROOT="./purify"
RENDER_ROOT="./renders"
METRIC_ROOT="./metric_logs_test"

if [ -n "$SCENE" ]; then
    SCENES=("$SCENE")
else
    SCENES=("1" "2" "3" "4" "7" "9" "10" "12" "book" "trash")
fi

for SCENE in "${SCENES[@]}"; do
    RGB_DIR="${DATASET_ROOT}/${SCENE}/images_4"
    MASK_DIR="${RGB_DIR}/label"

    INPAINTED_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}/inpainted"
    DEADMASK_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}/deadmasks"
    MERGED_DIR="${MERGED_ROOT}/${SCENE}_images"
    PURIFY_HYBRID_DIR="${PURIFY_HYBRID_ROOT}_${SCENE}"
    PURIFY_DIR="${PURIFY_ROOT}_${SCENE}"
    RENDER_DIR="${RENDER_ROOT}_${SCENE}"

    echo ""
    echo "============================================================"
    echo "🚀 開始處理場景: ${SCENE}"
    echo "   RGB:  ${RGB_DIR}"
    echo "   Mask: ${MASK_DIR}"
    echo "============================================================"

    # ── Step 1: VGGT inpainting（只取後 60 張）─────────────────
    echo ""
    echo "[${SCENE}] Step 1: VGGT inpainting..."
    python eval/eval_custom_360.py \
        --data_path "${RGB_DIR}" \
        --mask_path "${MASK_DIR}" \
        --enable_gen_3d_prop \
        --generate "all frame" \
        --inpaint_method sd \
        --n_skip 40

    # ── Step 2: 合併前 40 張 GT + 後 60 張 inpainted ─────────────
    echo ""
    echo "[${SCENE}] Step 2: Merging GT + inpainted images..."
    mkdir -p "${MERGED_DIR}"

    ls "${RGB_DIR}"/*.png | head -n 40 | xargs -I {} cp {} "${MERGED_DIR}"

    cp "${INPAINTED_DIR}"/*.png "${MERGED_DIR}"

    # ── Step 3: 用 merged 圖建立 COLMAP（nvs_pose 用）─────────────
    echo ""
    echo "[${SCENE}] Step 3: Building nvs_pose COLMAP (100 frames)..."
    python eval/eval_custom_colmap_masked.py \
        --data_path "${MERGED_DIR}" \
        --output_path "${PURIFY_HYBRID_DIR}" \
        

    # ── Step 4: 用純 inpainted 圖建立 COLMAP ─────────────────────
    echo ""
    echo "[${SCENE}] Step 4: Building point cloud COLMAP (60 inpainted frames)..."
    python eval/eval_custom_colmap_masked.py \
        --data_path "${INPAINTED_DIR}" \
        --output_path "${PURIFY_DIR}"

    # ── Step 5: 把 inpainted 圖搬到 purify_scene/images ─────────
    echo ""
    echo "[${SCENE}] Step 5: Moving inpainted images into purify dir..."
    mkdir -p "${PURIFY_DIR}/images"
    mv "${INPAINTED_DIR}"/*.png "${PURIFY_DIR}/images"

    # ── Step 6: 3DGS 訓練與渲染 ──────────────────────────────────
    echo ""
    echo "[${SCENE}] Step 6: 3DGS training and rendering..."
    python train_render_360.py \
        --colmap_dir    "${PURIFY_DIR}" \
        --nvs_pose      "${PURIFY_HYBRID_DIR}" \
        --train_img_dir "${PURIFY_DIR}/images" \
        --deadmask_dir  "${DEADMASK_DIR}" \
        --output_dir    "${RENDER_DIR}" \
        --total_iters   20000 \
        --dead_weight   0.3 \
        --patch_size    256

    # ── Step 7: 評估指標 ────────────────────────────────────────
    # echo ""
    # echo "[${SCENE}] Step 7: Evaluating metrics..."
    # python eval_metric_spinnerf_prtcl.py \
    #     --gt_img_dir     "${RGB_DIR}" \
    #     --render_img_dir "${RENDER_DIR}" \
    #     --mask_dir       "${RGB_DIR}/test_label" \
    #     --output_dir     "${METRIC_ROOT}"

    # echo ""
    # echo "✅ 場景 ${SCENE} 完成！"
done

echo ""
echo "============================================================"
echo "🎉 所有場景處理完畢！"
echo "============================================================"