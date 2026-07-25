#!/bin/bash
# ============================================================
# 360-USID 全場景批次測試腳本（由 run_spinnerf.sh 移植）
#
# 與 SPIn-NeRF 版的關鍵差異（因 360-USID 資料結構不同而必改）：
#   - train/test 不再靠排序切：images/ 全是 training，test_images/ 是 GT novel views
#   - 副檔名：training RGB/mask=.jpg、test GT=.jpg、test mask=.png
#   - Step 1 對 images/ 內「全部」frame 做 VGGT inpaint（不再 skip 前 40）
#   - Step 2 合併 inpainted(.png) + test_images(.jpg→轉.png) 供 NVS pose 估計
#     （沿用 spinnerf.bash 邏輯：合併集全為去物件影像，SfM 才乾淨）
#   - Step 6 改用 AuraFusion 協定：eval_metric_aura_prtcl.py + test_object_masks/
#   - reference/、unseen_masks/、出廠 sparse/0/ 全程不使用（依你的要求重估 pose）
#
# 輸出結構（與 SPIn-NeRF 版對齊）：
#   eval_results_custom/{DATASET_NAME}/{SCENE}/
#   ├── inpainted/   ← VGGT inpainting（全部 training frames）
#   ├── deadmasks/   ← dead zone masks
#   ├── colmap/      ← training COLMAP（inpainted，Step 1 產生）
#   ├── nvs_colmap/  ← NVS COLMAP（inpainted + test 合併，Step 3 產生）
#   └── renders/     ← 3DGS 渲染（涵蓋 test views）
#   暫存：./merged_data/{SCENE}/（Step 2 產生，all 模式跑完自動刪）
#
# 用法：
#   全部場景 / 全部 stage：           bash run_360USID.sh
#   單一場景：               SCENE="carton" bash run_360USID.sh
#   單一 stage（1~6）：      function="1"   bash run_360USID.sh
#   指定 config：            CONFIG=configs/xxx.yaml bash run_360USID.sh
# ============================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt

set -e

# ── Stage control ─────────────────────────────────────────────
RUN_STAGE="${function:-all}"
should_run () {
    local target="$1"
    [[ "${RUN_STAGE}" == "all" || "${RUN_STAGE}" == "${target}" ]]
}

# ── Path config ───────────────────────────────────────────────
DATASET_ROOT="../data/360-USID"
DATASET_NAME="$(basename "${DATASET_ROOT}")"
OUTPUT_ROOT="./eval_results_custom"
METRIC_ROOT="./metric_logs"
CONFIG_FILE="${CONFIG:-configs/usid_360.yaml}"
# VGGT depth confidence 門檻。注意：它不在 eval_iggt 的 YAML defaults 裡，yaml 改不到，
# 只能從這裡以 CLI 傳入。預設沿用 3.0；360 場景多半要調低（依 conf 分布診斷結果）。
#   用法：CONF_THRESH=1.0 SCENE=carton function=1 bash run_360USID.sh
CONF_THRESH="${CONF_THRESH:-1.0}"
# COLMAP 種子點上限（conf 過濾後隨機降採樣到此數）。種子太稀疏、背景重建差時可調高，
# 例如 MAX_POINTS=500000。
MAX_POINTS="${MAX_POINTS:-100000}"

# 從 yaml 讀 experiment.name 當作 metric 的 exp_name（讀不到就退回 aura_eval）
METRIC_EXP_NAME=$(python3 -c "import yaml;print(yaml.safe_load(open('${CONFIG_FILE}')).get('experiment',{}).get('name','aura_eval'))" 2>/dev/null || echo "aura_eval")
echo "  Config:          ${CONFIG_FILE}"
echo "  Metric exp_name: ${METRIC_EXP_NAME}"

# ── Scene selection ───────────────────────────────────────────
if [ -n "$SCENE" ]; then
    SCENES=("$SCENE")
else
    SCENES=("carton" "cone" "cookie" "newcone" "plant" "skateboard" "sunflower")
fi

# ── Main loop ─────────────────────────────────────────────────
for SCENE in "${SCENES[@]}"; do

    RGB_DIR="${DATASET_ROOT}/${SCENE}/images"                   # training RGB (含物件)
    MASK_DIR="${DATASET_ROOT}/${SCENE}/object_masks"            # training object masks
    TEST_IMG_DIR="${DATASET_ROOT}/${SCENE}/test_images"         # GT test views (無物件)
    TEST_MASK_DIR="${DATASET_ROOT}/${SCENE}/test_object_masks"  # 評估用 object mask

    BASE_OUT="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}"
    INPAINTED_DIR="${BASE_OUT}/inpainted"
    DEADMASK_DIR="${BASE_OUT}/deadmasks"
    COLMAP_DIR="${BASE_OUT}/colmap"
    NVS_COLMAP_DIR="${BASE_OUT}/nvs_colmap"
    RENDER_DIR="${BASE_OUT}/renders"
    MERGED_DIR="./merged_data/${SCENE}"

    echo ""
    echo "============================================================"
    echo "🚀 場景: ${SCENE}  (Dataset: ${DATASET_NAME}, Stage: ${RUN_STAGE})"
    echo "   train RGB:  ${RGB_DIR}"
    echo "   train mask: ${MASK_DIR}"
    echo "   test  GT:   ${TEST_IMG_DIR}"
    echo "   test  mask: ${TEST_MASK_DIR}"
    echo "   output:     ${BASE_OUT}"
    echo "============================================================"
    mkdir -p "${BASE_OUT}"

    # ── Step 1: VGGT inpainting（全部 training frames）+ training COLMAP ──
    #   注意：images/ 內 186 張全部含物件、全部有對應 object_masks → 全部都要 inpaint。
    #   （SPIn-NeRF 是前 40 張無 mask 才被略過；這裡沒有那種情況）
    if should_run 1; then
        echo ""
        echo "[${SCENE}] Step 1: VGGT inpainting (ALL training frames) + training COLMAP..."
        python eval/eval_iggt.py \
            --config "${CONFIG_FILE}" \
            --data_path "${RGB_DIR}" \
            --mask_path "${MASK_DIR}" \
            --exp_name "${SCENE}" \
            --output_root "${BASE_OUT}" \
            --depth_conf_thresh "${CONF_THRESH}" \
            --colmap_max_points "${MAX_POINTS}"

        # Sanity gate：eval_iggt 即使「有效點 0」也會 return 且 exit 0，set -e 攔不到，
        # 會印出假的「✅完成」。這裡實際讀 COLMAP，空的就中止，避免後段建立在垃圾上。
        python3 - "${COLMAP_DIR}/sparse" <<'PY' || { echo "❌ Step 1 的 COLMAP 是空的（0 點 / 無註冊影像）。多半是 depth_conf_thresh 過嚴，或 VGGT 在大量 frame 下退化。停在這裡，不要往後跑 Step 2~6。"; exit 1; }
import sys
try:
    import pycolmap
except ImportError:
    print("  (pycolmap 不可用，略過 COLMAP 空值檢查)"); sys.exit(0)
try:
    rec = pycolmap.Reconstruction(sys.argv[1])
except Exception as e:
    print(f"  讀取 COLMAP 失敗：{e}"); sys.exit(1)
npts = len(rec.points3D)
nreg = sum(1 for im in rec.images.values() if im.registered)
print(f"  ✅ COLMAP check：{npts:,} points / {nreg} registered images")
sys.exit(0 if (npts > 0 and nreg > 0) else 1)
PY
    fi

    # ── Step 2: 合併 inpainted + test_images → temp（供 NVS pose 估計）──
    #   inpainted/ 為 .png（已去物件）；test_images/ 為 .jpg（GT，無物件）→ 轉 .png 放入，
    #   讓合併集維持「全 PNG」，避免後段對 *.png 做 glob 時漏掉 test。
    #   train stem (00000~) 與 test stem (高位 index) 不重疊，cp 不會互蓋。
    if should_run 2; then
        echo ""
        echo "[${SCENE}] Step 2: Merging inpainted + test_images (→ all PNG)..."
        rm -rf "${MERGED_DIR}"
        mkdir -p "${MERGED_DIR}"
        cp "${INPAINTED_DIR}"/*.png "${MERGED_DIR}/"
        python3 - "${TEST_IMG_DIR}" "${MERGED_DIR}" <<'PY'
import sys, glob, os
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.jpeg")))
for f in files:
    bn = os.path.splitext(os.path.basename(f))[0]
    Image.open(f).convert("RGB").save(os.path.join(dst, bn + ".png"))
print(f"  converted {len(files)} test frames .jpg -> .png")
PY
        echo "  ✅ merged: $(ls "${MERGED_DIR}"/*.png 2>/dev/null | wc -l) frames in ${MERGED_DIR}"
    fi

    # ── Step 3: 對合併集估計 COLMAP → nvs_colmap（含 test 的完整視角 pose）──
    if should_run 3; then
        echo ""
        echo "[${SCENE}] Step 3: Building NVS COLMAP (inpainted + test merged)..."
        python eval/eval_custom_colmap_masked.py \
            --data_path "${MERGED_DIR}" \
            --output_path "${NVS_COLMAP_DIR}"
    fi

    # ── Step 4: 3DGS 訓練（training COLMAP = colmap/，全部 inpainted）──
    if should_run 4; then
        echo ""
        echo "[${SCENE}] Step 4: 3DGS training..."
        if [ ! -d "${DEADMASK_DIR}" ]; then
            echo "  ⚠️  DEADMASK_DIR 不存在：${DEADMASK_DIR}"
        else
            echo "  💀 Dead masks: $(ls "${DEADMASK_DIR}"/*.png 2>/dev/null | wc -l) files"
        fi
        mkdir -p "${RENDER_DIR}"
        python train.py \
            --config          "${CONFIG_FILE}" \
            --colmap_dir      "${COLMAP_DIR}" \
            --train_img_dir   "${COLMAP_DIR}/images" \
            --deadmask_dir    "${DEADMASK_DIR}" \
            --output_gaussian "${RENDER_DIR}/gaussians.pth"
    fi

    # ── Step 5: 3DGS 渲染（NVS test poses from Step 3）──
    if should_run 5; then
        echo ""
        echo "[${SCENE}] Step 5: Rendering from NVS poses (${NVS_COLMAP_DIR})..."
        python render.py \
            --nvs_pose          "${NVS_COLMAP_DIR}" \
            --gaussian_path     "${RENDER_DIR}/gaussians.pth" \
            --render_output_dir "${RENDER_DIR}"
    fi

    # ── Step 6: 評估指標（AuraFusion 協定）──
    #   GT = test_images/、mask = test_object_masks/
    #   render/ 內含 train+test 全部 frame；aura_prtcl 以「檔名 stem」自動挑出 test 對應，
    #   並處理 .jpg(GT)/.png(render) 副檔名差異。
    if should_run 6; then
        echo ""
        echo "[${SCENE}] Step 6: Evaluating metrics (AuraFusion protocol)..."
        python eval_metric_aura_prtcl.py \
            --gt_img_dir     "${TEST_IMG_DIR}" \
            --render_img_dir "${RENDER_DIR}" \
            --mask_dir       "${TEST_MASK_DIR}" \
            --output_dir     "${METRIC_ROOT}" \
            --scene          "${SCENE}" \
            --exp_name       "${METRIC_EXP_NAME}"
    fi

    # ── Cleanup（僅 all 模式）──
    if [[ "${RUN_STAGE}" == "all" ]] && [ -d "${MERGED_DIR}" ]; then
        echo ""
        echo "[${SCENE}] Cleanup: removing temp merged dir..."
        rm -rf "${MERGED_DIR}"
    fi

    echo ""
    echo "✅ 場景 ${SCENE} 完成！renders → ${RENDER_DIR}/"
done

if [[ "${RUN_STAGE}" == "all" ]] && [ -d "./merged_data" ]; then
    echo ""
    echo "🧹 Final cleanup: removing ./merged_data..."
    rm -rf "./merged_data"
fi

echo ""
echo "============================================================"
echo "🎉 所有場景處理完畢！metric json → ${METRIC_ROOT}/${METRIC_EXP_NAME}/"
echo "============================================================"