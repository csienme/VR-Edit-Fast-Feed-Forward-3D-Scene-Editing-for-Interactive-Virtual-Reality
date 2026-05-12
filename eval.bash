set -e  # 任何指令失敗就停止

if [ -n "$SCENE" ]; then
    SCENES=("$SCENE")
else
    SCENES=("1" "2" "3" "4" "7" "9" "10" "12" "book" "trash")
fi

for SCENE in "${SCENES[@]}"; do

    # ── Step 7: 評估指標 ────────────────────────────────────────────
    echo "[${SCENE}] Step 7: Evaluating metrics..."
    python eval_metric_spinnerf_prtcl.py \
        --gt_img_dir       ../spinnerf-dataset/${SCENE}/images_4 \
        --render_img_dir   render_output/${SCENE} \
        --mask_dir ../spinnerf-dataset/${SCENE}/images_4/test_label \
        --output_dir metric_logs\
        --exp_name spinnerf \
        --scene ${SCENE}

    echo "✅ 場景 ${SCENE} 完成！"
done