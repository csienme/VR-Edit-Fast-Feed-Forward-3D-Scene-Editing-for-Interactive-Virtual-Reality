#!/bin/bash
# ============================================================
# SPIn-NeRF 全場景批次測試腳本
#
# 輸出結構（與 run.sh 對齊）：
#   eval_results_custom/{DATASET_NAME}/{SCENE}/
#   ├── inpainted/      ← VGGT inpainting 結果（60 frames）
#   ├── deadmasks/      ← dead zone masks（60 frames）
#   ├── colmap/         ← training COLMAP（60 inpainted，由 Step 1 直接產生）
#   ├── nvs_colmap/     ← NVS test COLMAP（100 merged frames，SPIn-NeRF 專屬）
#   └── renders/        ← 3DGS 渲染結果
#
# 暫存（pipeline 結束後自動刪除）：
#   ./merged_data/{SCENE}/  ← 40 GT + 60 inpainted（Step 2 用，Step 3 後即刪）
#
# Stage 控制（未指定則跑全部 1~6）：
#   function="1" bash run_spinnerf.sh
#   function="3" bash run_spinnerf.sh
#
# 指定單一場景：
#   SCENE="1" bash run_spinnerf.sh
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
DATASET_ROOT="../spinnerf-dataset"
DATASET_NAME="$(basename "${DATASET_ROOT}")"
OUTPUT_ROOT="./eval_results_custom"
METRIC_ROOT="./metric_logs_test"

# ── Scene selection ───────────────────────────────────────────
if [ -n "$SCENE" ]; then
    SCENES=("$SCENE")
else
    SCENES=("1" "2" "3" "4" "7" "9" "10" "12" "book" "trash")
fi

# ── Main loop ─────────────────────────────────────────────────
for SCENE in "${SCENES[@]}"; do

    RGB_DIR="${DATASET_ROOT}/${SCENE}/images_4"
    MASK_DIR="${RGB_DIR}/label"

    BASE_OUT="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}"

    INPAINTED_DIR="${BASE_OUT}/inpainted"
    DEADMASK_DIR="${BASE_OUT}/deadmasks"
    COLMAP_DIR="${BASE_OUT}/colmap"          # training COLMAP（60 inpainted，Step 1 產生）
    NVS_COLMAP_DIR="${BASE_OUT}/nvs_colmap"  # NVS COLMAP（100 merged frames，Step 3 產生）
    RENDER_DIR="${BASE_OUT}/renders"

    # 暫存：pipeline 結束後自動刪除
    MERGED_DIR="./merged_data/${SCENE}"

    echo ""
    echo "============================================================"
    echo "🚀 開始處理場景: ${SCENE}"
    echo "   Dataset: ${DATASET_NAME}"
    echo "   RGB:     ${RGB_DIR}"
    echo "   Mask:    ${MASK_DIR}"
    echo "   Output:  ${BASE_OUT}"
    echo "   Stage:   ${RUN_STAGE}"
    echo "   Structure:"
    echo "     inpainted/  → ${INPAINTED_DIR}"
    echo "     deadmasks/  → ${DEADMASK_DIR}"
    echo "     colmap/     → ${COLMAP_DIR}  (60 inpainted，Step 1)"
    echo "     nvs_colmap/ → ${NVS_COLMAP_DIR}  (100 merged，Step 3)"
    echo "     renders/    → ${RENDER_DIR}"
    echo "============================================================"

    mkdir -p "${BASE_OUT}"

    # ── Step 1: VGGT inpainting（後 60 張）+ training COLMAP ─────
    # eval_iggt.py 跳過前 40 張 GT，對後 60 張做 inpainting，
    # 並直接產生：
    #   - colmap/sparse/ + colmap/images/（training COLMAP）
    #   - inpainted/（inpainted PNG）
    #   - deadmasks/（dead zone masks）
    if should_run 1; then
        echo ""
        echo "[${SCENE}] Step 1: VGGT inpainting (skip first 40 GT) + training COLMAP..."
        python eval/eval_iggt.py \
            --data_path "${RGB_DIR}" \
            --mask_path "${MASK_DIR}" \
            --enable_gen_3d_prop \
            --generate "all frame" \
            --exp_name "${SCENE}" \
            --inpaint_method sd \
            --n_skip 40 \
            --output_root "${BASE_OUT}"
    fi

    # ── Step 2: 合併前 40 張 GT + 後 60 張 inpainted → temp ──────
    # 這份 100 張的合集是 Step 3 COLMAP 用的輸入（含完整相機軌跡）
    if should_run 2; then
        echo ""
        echo "[${SCENE}] Step 2: Merging 40 GT + 60 inpainted frames..."
        mkdir -p "${MERGED_DIR}"
        # 取前 40 張 GT（依序排列）
        ls "${RGB_DIR}"/*.png | sort | head -n 40 | xargs -I {} cp {} "${MERGED_DIR}/"
        # 加入 Step 1 產生的 60 張 inpainted
        cp "${INPAINTED_DIR}"/*.png "${MERGED_DIR}/"
        echo "  ✅ merged: $(ls "${MERGED_DIR}" | wc -l) frames in ${MERGED_DIR}"
    fi

    # ── Step 3: COLMAP on 100 merged frames → nvs_colmap ─────────
    # SPIn-NeRF 專屬：估計完整 100 視角（含 GT test views）的 camera pose
    # 供 render.py 用作 nvs_pose，確保能渲染到前 40 張 GT 的視角
    if should_run 3; then
        echo ""
        echo "[${SCENE}] Step 3: Building NVS COLMAP (100 merged frames)..."
        python eval/eval_custom_colmap_masked.py \
            --data_path "${MERGED_DIR}" \
            --output_path "${NVS_COLMAP_DIR}"
    fi

    # ── Step 4: 3DGS 訓練 ─────────────────────────────────────────
    # 訓練資料來自 colmap/（Step 1 產生，60 inpainted frames）
    # 輸出 gaussians.pth 供 Step 5 render 使用
    if should_run 4; then
        echo ""
        echo "[${SCENE}] Step 4: 3DGS training..."

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
    fi

    # ── Step 5: 3DGS 渲染（NVS test poses from Step 3）────────────
    # nvs_pose 使用 nvs_colmap/（100 frames COLMAP），
    # 確保渲染涵蓋 SPIn-NeRF 的 40 張 GT test views
    if should_run 5; then
        echo ""
        echo "[${SCENE}] Step 5: Rendering from NVS poses (${NVS_COLMAP_DIR})..."
        python render.py \
            --nvs_pose          "${NVS_COLMAP_DIR}" \
            --gaussian_path     "${RENDER_DIR}/gaussians.pth" \
            --render_output_dir "${RENDER_DIR}"
    fi

    # ── Step 6: 評估指標 ──────────────────────────────────────────
    if should_run 6; then
        echo ""
        echo "[${SCENE}] Step 6: Evaluating metrics..."
        python eval_metric_spinnerf_prtcl.py \
            --gt_img_dir     "${RGB_DIR}" \
            --render_img_dir "${RENDER_DIR}" \
            --mask_dir       "${RGB_DIR}/test_label" \
            --output_dir     "${METRIC_ROOT}" \
            --scene "${SCENE}"\
            --exp_name "simplify_pipe"

    fi

    # ── Cleanup: 刪除 merged 暫存（只在全跑 all 模式下自動清除）────
    # 單 stage 模式跑時保留，避免誤刪後續 stage 還需要的資料
    if [[ "${RUN_STAGE}" == "all" ]] && [ -d "${MERGED_DIR}" ]; then
        echo ""
        echo "[${SCENE}] Cleanup: removing temp merged dir..."
        rm -rf "${MERGED_DIR}"
    fi

    echo ""
    echo "✅ 場景 ${SCENE} 完成！"
    echo "   Inpainted:  ${INPAINTED_DIR}/"
    echo "   Deadmasks:  ${DEADMASK_DIR}/"
    echo "   COLMAP:     ${COLMAP_DIR}/"
    echo "   NVS COLMAP: ${NVS_COLMAP_DIR}/"
    echo "   Renders:    ${RENDER_DIR}/"

done

# 全跑完後若 merged_data/ 還存在（理論上不該有，防呆用），一併清除
if [[ "${RUN_STAGE}" == "all" ]] && [ -d "./merged_data" ]; then
    echo ""
    echo "🧹 Final cleanup: removing ./merged_data..."
    rm -rf "./merged_data"
fi

echo ""
echo "============================================================"
echo "🎉 所有場景處理完畢！"
echo "============================================================"