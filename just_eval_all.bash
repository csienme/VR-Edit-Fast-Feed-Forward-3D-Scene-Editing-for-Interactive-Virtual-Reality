SCENES=("1" "2" "3" "4" "7" "9" "10" "12" "book" "trash")

for SCENE in "${SCENES[@]}"; do

    echo "[${SCENE}] Step 7: Evaluating metrics..."
    python eval_metric_3dgic_prtcl.py \
        --gt_img_dir       spinnerf-dataset/${SCENE}/images_4 \
        --render_img_dir   renders_${SCENE} \
        --mask_dir spinnerf-dataset/${SCENE}/images_4/test_label \
        --output_dir       ./metric_logs

    echo "✅ 場景 ${SCENE} 完成！"
done