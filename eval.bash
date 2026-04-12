
python eval_metric.py \
  --gt_img_dir spinnerf-dataset/book/images_4 \
  --render_img_dir renders_book \
  --masked_label_dir spinnerf-dataset/book/images_4/test_label \
  --output_dir ./metric_logs
#gt image 會自動只取前面40張