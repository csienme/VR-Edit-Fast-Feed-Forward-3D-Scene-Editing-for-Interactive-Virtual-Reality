#!/bin/bash
# ============================================================
# SPInNeRF 全場景批次測試腳本
# 場景: 1 2 3 4 7 9 10 12
# ============================================================

set -e  # 任何指令失敗就停止

if [ -n "$SCENE" ]; then
    SCENES=("$SCENE")
else
    SCENES=("1" "2" "3" "4" "7" "9" "10" "12" "book" "test")
fi

for SCENE in "${SCENES[@]}"; do
    echo ""
    echo "============================================================"
    echo "🚀 開始處理場景: ${SCENE}"
    echo "============================================================"

    # ── Step 1: VGGT inpainting（只取後60張）──────────────────────
    echo "[${SCENE}] Step 1: VGGT inpainting..."
    python eval/eval_custom.py \
        --data_path spinnerf-dataset/${SCENE}/images_4 \
        --mask_path spinnerf-dataset/${SCENE}/images_4/label \
        --enable_gen_3d_prop \
        --generate "all frame" \
        --exp_name "${SCENE}" \
        --n_skip 40

    # ── Step 2: 合併前40張GT + 後60張inpainted ────────────────────
    echo "[${SCENE}] Step 2: Merging GT + inpainted images..."
    mkdir -p merged_data/${SCENE}_images

    ls spinnerf-dataset/${SCENE}/images_4/*.png | head -n 40 | xargs -I {} cp {} merged_data/${SCENE}_images

    cp eval_results_custom/custom_dataset_spinnerf-dataset/${SCENE}/images_4/${SCENE}/*.png \
       merged_data/${SCENE}_images

    # ── Step 3: 用 eval_custom_colmap 對合併圖建立 COLMAP（nvs_pose 用）
    echo "[${SCENE}] Step 3: Building nvs_pose COLMAP (100 frames)..."
    python eval/eval_custom_colmap.py \
        --data_path merged_data/${SCENE}_images \
        --output_path purify_hybrid_${SCENE}

    # ── Step 4: 用 eval_custom_colmap 對純60張 inpainted 建立 COLMAP ──
    echo "[${SCENE}] Step 4: Building point cloud COLMAP (60 inpainted frames)..."
    python eval/eval_custom_colmap.py \
        --data_path eval_results_custom/custom_dataset_spinnerf-dataset/${SCENE}/images_4/${SCENE} \
        --output_path purify_${SCENE}

    # ── Step 5: 把 inpainted 圖移到 purify_scene 的 images/ 底下 ──
    echo "[${SCENE}] Step 5: Moving inpainted images into purify dir..."
    mkdir -p purify_${SCENE}/images

    mv eval_results_custom/custom_dataset_spinnerf-dataset/${SCENE}/images_4/${SCENE}/*.png \
       purify_${SCENE}/images

    # ── Step 6: 3DGS 訓練與渲染 ────────────────────────────────────
    echo "[${SCENE}] Step 6: 3DGS training and rendering..."
    python train_render.py \
        --colmap_dir    purify_${SCENE} \
        --nvs_pose      purify_hybrid_${SCENE} \
        --train_img_dir purify_${SCENE}/images \
        --output_dir    ./renders_${SCENE}

    # ── Step 7: 評估指標 ────────────────────────────────────────────
    echo "[${SCENE}] Step 7: Evaluating metrics..."
    python eval_metric.py \
        --gt_img_dir       spinnerf-dataset/${SCENE}/images_4 \
        --render_img_dir   renders_${SCENE} \
        --mask_dir spinnerf-dataset/${SCENE}/images_4/test_label \
        --output_dir       ./metric_logs

    echo "✅ 場景 ${SCENE} 完成！"
done

echo ""
echo "============================================================"
echo "🎉 所有場景處理完畢！"
echo "============================================================"