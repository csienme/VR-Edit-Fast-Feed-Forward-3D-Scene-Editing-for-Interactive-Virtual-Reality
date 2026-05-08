#!/bin/bash
# conda 環境啟動
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt
# ============================================================
# 單一新場景 inpaint + render（無 GT test view，render 在 train poses）
#
# 用法：
#   bash run_custom_scene.sh <scene_name> <rgb_dir> <mask_dir>
#
# 例：
#   bash run_custom_scene.sh bear_statue ./mydata/bear/rgb ./mydata/bear/mask
#
# 對應 SPInNeRF 場景的執行方式不變（仍用原本的 batch script）。
# ============================================================

set -e

SCENE="${1:?需要 scene name}"
RGB_DIR="${2:?需要 RGB 圖片目錄（絕對路徑或相對路徑）}"
MASK_DIR="${3:?需要 mask 目錄（白=要 inpaint 的物體區）}"

echo "============================================================"
echo "🚀 場景: ${SCENE}"
echo "   RGB:  ${RGB_DIR}"
echo "   Mask: ${MASK_DIR}"
echo "============================================================"

# eval_custom.py 的輸出路徑公式（line 247）：
#   output_path / f"custom_dataset_{data_path}" / exp_name
# 即直接把 data_path 字串接在 custom_dataset_ 後面，包含 ../bear/input 這樣的路徑
# 因此 INPAINTED_DIR 直接用 RGB_DIR 的字面值組合
INPAINTED_DIR="eval_results_custom/custom_dataset_${RGB_DIR}/${SCENE}"
DEADMASK_DIR="eval_results_custom/custom_dataset_${RGB_DIR}/deadmasks"
# ── Step 1: VGGT inpainting（n_skip=0 = 全部 frame 都 inpaint）─────
echo ""
echo "[${SCENE}] Step 1: VGGT inpainting all frames..."
python eval/eval_custom_360.py \
    --data_path  "${RGB_DIR}" \
    --mask_path  "${MASK_DIR}" \
    --enable_gen_3d_prop \
    --generate   "all frame" \
    --exp_name   "${SCENE}" \
    --n_skip     0

# ── Step 2: 建立單一 COLMAP（從 inpainted 圖）────────────────────
echo ""
echo "[${SCENE}] Step 2: Building single COLMAP from inpainted frames..."
python eval/eval_custom_colmap.py \
    --data_path   "${INPAINTED_DIR}" \
    --output_path "purify_${SCENE}"

# ── Step 3: 把 inpainted 圖搬到 purify 底下供 train_render 讀取 ──
echo ""
echo "[${SCENE}] Step 3: Moving inpainted images into purify dir..."
mkdir -p "purify_${SCENE}/images"
cp "${INPAINTED_DIR}"/inpainted_*.png "purify_${SCENE}/images/"

# ── Step 4: 3DGS 訓練 + render 在 train poses（即原 input 視角）──
# 關鍵：--colmap_dir 和 --nvs_pose 指向同一個目錄
# 由於檔名是 inpainted_*.png，train_render.py 會自動 fallback
# 進入 single-COLMAP mode，render 全部 camera 即等於 input frame poses
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
    --colmap_dir    "purify_${SCENE}" \
    --nvs_pose      "purify_${SCENE}" \
    --train_img_dir "purify_${SCENE}/images" \
    --deadmask_dir  "${DEADMASK_DIR}" \
    --output_dir    "./renders_${SCENE}" \
    --total_iters   20000 \
    --dead_weight   0.3 \
    --patch_size    256

echo ""
echo "============================================================"
echo "✅ 場景 ${SCENE} 完成！"
echo "   Inpainted:  ${INPAINTED_DIR}/"
echo "   Renders:    ./renders_${SCENE}/"
echo "============================================================"