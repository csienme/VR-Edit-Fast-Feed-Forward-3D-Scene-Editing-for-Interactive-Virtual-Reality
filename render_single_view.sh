#!/bin/bash
# ============================================================
# render_single_view.sh
# 用途：把「已訓練好的 gaussian.pth」渲染到一個「新的 NVS 視角」。
#
# 輸入：
#   1) 目標視角的 RGB（單張檔，或一個資料夾＝批次多張）
#   2) 訓練好的 gaussian.pth
#   3) 當初訓練這顆 gaussian 的那批影像資料夾
#        （即 train.py --train_img_dir 指到的那個，通常是 .../colmap/images）
#
# 輸出：
#   OUT_DIR/<stem>.png  ← 對應目標視角 pose 渲染出的 RGB
#
# 原理：
#   單張影像無法獨立估 pose。把「訓練影像 + 目標視角」一起餵 VGGT，
#   目標視角的 pose 就會落在與訓練影像同一個座標系裡（合併純粹為了對齊座標系），
#   再用 gaussian 渲染那個 pose。render.py 會把合併集所有 pose 都渲染，
#   最後依「檔名 stem」把目標那張挑出來。
#
# ⚠️ 座標系一致性（重要，會影響這張 render 對不對）：
#   這支是「重新跑一次 VGGT 估 pose」。它假設這次 VGGT 的座標系，與當初訓練
#   gaussian 用的 training COLMAP 座標系一致（VGGT 以 frame[0] 為錨、scale 一致）。
#   forward-facing 大致成立；360 大視角變化時可能有尺度/旋轉飄移，而「飄移的座標系」
#   本身就是 test view render 變糟的可能成因之一。若這張 render 整體位移/歪斜，
#   穩健解是把目標相機「對齊進既有的 training COLMAP」（用兩個座標系共有的訓練相機做
#   Procrustes/相似變換，再把目標 pose 映射過去）——需要這版本再跟我說。
#
# 用法：
#   bash render_single_view.sh <test_view.jpg|dir> <gaussian.pth> <train_img_dir> [out_dir]
#
# 範例：
#   bash render_single_view.sh \
#       ../data/360-USID/carton/test_images/00188.jpg \
#       eval_results_custom/360-USID/carton/renders/gaussians.pth \
#       eval_results_custom/360-USID/carton/colmap/images \
#       ./single_render_out
# ============================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastvggt
set -e

# ── Args ──────────────────────────────────────────────────────
TEST_VIEW="$1"      # 要渲染的目標視角 RGB（單張檔或資料夾）
GAUSSIAN="$2"       # 訓練好的 gaussian.pth
TRAIN_DIR="$3"      # 當初訓練這顆 gaussian 的那批影像資料夾
OUT_DIR="${4:-./single_render_out}"

usage () {
    echo "用法: bash render_single_view.sh <test_view.jpg|dir> <gaussian.pth> <train_img_dir> [out_dir]"
    exit 1
}
if [ -z "$TEST_VIEW" ] || [ -z "$GAUSSIAN" ] || [ -z "$TRAIN_DIR" ]; then usage; fi
[ -e "$TEST_VIEW" ] || { echo "❌ 找不到目標視角: $TEST_VIEW"; exit 1; }
[ -f "$GAUSSIAN"  ] || { echo "❌ 找不到 gaussian.pth: $GAUSSIAN"; exit 1; }
[ -d "$TRAIN_DIR" ] || { echo "❌ 找不到 train_img_dir: $TRAIN_DIR"; exit 1; }

# ── Work dirs（暫存，可刪）─────────────────────────────────────
WORK="${OUT_DIR}/_work"
MERGED_DIR="${WORK}/merged"
NVS_COLMAP_DIR="${WORK}/nvs_colmap"
RENDER_DIR="${WORK}/renders"
rm -rf "${WORK}"
mkdir -p "${OUT_DIR}" "${MERGED_DIR}" "${RENDER_DIR}"

echo "============================================================"
echo "🎯 單視角渲染"
echo "   target view : ${TEST_VIEW}"
echo "   gaussian    : ${GAUSSIAN}"
echo "   train imgs  : ${TRAIN_DIR}"
echo "   output      : ${OUT_DIR}"
echo "============================================================"

# ── Step 1: 合併「訓練影像 + 目標視角」→ 全 PNG（只為對齊座標系）──
echo ""
echo "[1/3] 合併訓練影像 + 目標視角 → ${MERGED_DIR}"

# (a) 訓練影像：png 直接複製、其餘轉 png（保持與當初訓練輸入一致）
python3 - "${TRAIN_DIR}" "${MERGED_DIR}" <<'PY'
import sys, glob, os, shutil
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
exts = ("*.png","*.jpg","*.jpeg","*.JPG","*.JPEG","*.PNG")
files = sorted(set(sum([glob.glob(os.path.join(src, e)) for e in exts], [])))
for f in files:
    bn = os.path.splitext(os.path.basename(f))[0]
    out = os.path.join(dst, bn + ".png")
    if f.lower().endswith(".png"):
        shutil.copy2(f, out)
    else:
        Image.open(f).convert("RGB").save(out)
print(f"  訓練影像 {len(files)} 張")
PY
N_TRAIN=$(ls "${MERGED_DIR}"/*.png 2>/dev/null | wc -l)

# (b) 目標視角：轉 png 併入，並回傳其 stem（render 會用這個名字輸出）
#     若目標尺寸與訓練影像不同（VGGT 要求同尺寸，否則 np.stack 會炸），
#     自動 resize 成訓練影像尺寸（從已複製進 merged 的訓練圖偵測參考尺寸）。
TARGET_STEMS=$(python3 - "${TEST_VIEW}" "${MERGED_DIR}" <<'PY'
import sys, glob, os
from PIL import Image
tv, dst = sys.argv[1], sys.argv[2]

# 參考尺寸：取自已複製進 merged 的訓練影像
ref = sorted(glob.glob(os.path.join(dst, "*.png")))
ref_size = None
if ref:
    with Image.open(ref[0]) as im:
        ref_size = im.size  # (W, H)

if os.path.isdir(tv):
    exts = ("*.png","*.jpg","*.jpeg","*.JPG","*.JPEG","*.PNG")
    files = sorted(set(sum([glob.glob(os.path.join(tv, e)) for e in exts], [])))
else:
    files = [tv]

stems = []
for f in files:
    bn = os.path.splitext(os.path.basename(f))[0]
    out = os.path.join(dst, bn + ".png")
    im = Image.open(f).convert("RGB")
    if ref_size is not None and im.size != ref_size:
        print(f"  resize 目標 {bn}: {im.size} -> {ref_size}", file=sys.stderr)
        im = im.resize(ref_size, Image.BILINEAR)
    im.save(out)
    stems.append(bn)
print(" ".join(stems))
PY
)
if [ -z "${TARGET_STEMS// /}" ]; then echo "❌ 目標視角沒有可用影像"; exit 1; fi
echo "  目標視角 stem: ${TARGET_STEMS}"
echo "  合併後共 $(ls "${MERGED_DIR}"/*.png | wc -l) 張（訓練 ${N_TRAIN} + 目標）"

# 防呆：目標 stem 不可與訓練影像撞名（撞名會互相覆蓋）
for stem in ${TARGET_STEMS}; do
    if ls "${TRAIN_DIR}" | grep -qE "^${stem}\.(png|jpg|jpeg)$" 2>/dev/null; then
        echo "  ⚠️  目標 stem '${stem}' 與訓練影像同名，可能互蓋。建議改名目標檔。"
    fi
done

# ── Step 2: VGGT 估 pose（合併集）→ nvs_colmap ────────────────
echo ""
echo "[2/3] VGGT 估 pose（合併集）→ ${NVS_COLMAP_DIR}"
python eval/eval_custom_colmap_masked.py \
    --data_path   "${MERGED_DIR}" \
    --output_path "${NVS_COLMAP_DIR}"

# ── Step 3: 用 gaussian 渲染所有 pose，再挑出目標視角 ─────────
echo ""
echo "[3/3] 渲染 → ${RENDER_DIR}，再擷取目標視角"
python render.py \
    --nvs_pose          "${NVS_COLMAP_DIR}" \
    --gaussian_path     "${GAUSSIAN}" \
    --render_output_dir "${RENDER_DIR}"

# 依 stem 把目標視角的 render 複製到 OUT_DIR
echo ""
n_ok=0
for stem in ${TARGET_STEMS}; do
    hit=$(find "${RENDER_DIR}" -type f \( -name "${stem}.png" -o -name "${stem}.jpg" -o -name "${stem}.jpeg" \) | head -1)
    if [ -n "${hit}" ]; then
        cp "${hit}" "${OUT_DIR}/${stem}.png"
        echo "  ✅ ${OUT_DIR}/${stem}.png"
        n_ok=$((n_ok + 1))
    else
        echo "  ⚠️  找不到 '${stem}' 的 render；render 輸出命名可能不是來源 stem，檢查 ${RENDER_DIR}/"
    fi
done

echo ""
echo "============================================================"
echo "🎉 完成：${n_ok} 張單視角 render 在 ${OUT_DIR}/"
echo "   中間暫存在 ${WORK}/（合併影像 / nvs_colmap / 全部 render），可自行刪除。"
echo "============================================================"