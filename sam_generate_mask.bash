conda activate sam2env
cd ~/paul/FastVGGT

python generate_masks_sam2.py \
    --img_dir my_scene4 \
    --out_dir my_scene4_masks \
    --frame_idx 0 \
    --points "660,500;760,470;620,580" \
    --preview

#points 拿去給gpt分析 只要給第一張frame分析就好