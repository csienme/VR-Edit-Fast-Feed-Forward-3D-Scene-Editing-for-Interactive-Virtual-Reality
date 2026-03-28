
python eval_metric.py \
  --gt_img_dir spinnerf-dataset/trash/images_4 \
  --render_img_dir ./renders_40 \
  --masked_label_dir spinnerf-dataset/trash/images_4/test_label \
  --output_dir ./metric_logs
#gt image 會自動只取前面40張