#!/usr/bin/env python3
"""
generate_masks_sam2.py — 用 SAM2 video propagation 為自建場景生成 object masks
================================================================================
流程：把整個 scene 的圖片序列當成「影片」，你只需在其中一張圖上點幾個點
標出要移除的物件，SAM2 會自動把 mask 傳播到所有其他視角（跨視角一致）。

用法一（互動模式，有螢幕的機器）:
    python generate_masks_sam2.py \
        --img_dir my_scene \
        --out_dir my_scene/object_masks

    → 彈出提示視窗：左鍵點物件上的點(前景)，右鍵點背景(排除)，
      可點多個點，按 Enter 確認 → 自動傳播到全部 frame

用法二（CLI 模式，SSH 無螢幕時直接給座標）:
    python generate_masks_sam2.py \
        --img_dir my_scene \
        --out_dir my_scene/object_masks \
        --points "504,378;550,400" \
        --neg_points "100,100"

    座標格式 "x,y;x,y"（像素座標，原圖 1008x756 尺度）

其他參數:
    --frame_idx 0        在第幾張圖上標點（預設第 0 張；物件被遮擋時換一張清楚的）
    --ckpt ./checkpoints/sam2.1_hiera_large.pt
    --preview            傳播完成後輸出 overlay 預覽圖到 out_dir/../mask_preview/

輸出:
    out_dir/<原圖同名>.png   二值 mask（0/255，白=物件）
    與原圖同 stem、同排序 → pipeline 依 sorted 順序配對不會錯位。
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_pts(s):
    if not s:
        return []
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        x, y = tok.split(",")
        out.append((float(x), float(y)))
    return out


def interactive_pick(img_bgr, title="左鍵=物件(前景) 右鍵=背景(排除) Enter=完成 u=undo"):
    """matplotlib 互動點選。回傳 (pos_pts, neg_pts)。"""
    import matplotlib
    import matplotlib.pyplot as plt

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pos, neg = [], []
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img_rgb)
    ax.set_title(title)

    def redraw():
        ax.clear()
        ax.imshow(img_rgb)
        ax.set_title(title)
        if pos:
            p = np.array(pos)
            ax.scatter(p[:, 0], p[:, 1], c="lime", marker="*", s=200, edgecolors="k")
        if neg:
            n = np.array(neg)
            ax.scatter(n[:, 0], n[:, 1], c="red", marker="x", s=150)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax or ev.xdata is None:
            return
        if ev.button == 1:
            pos.append((ev.xdata, ev.ydata))
        elif ev.button == 3:
            neg.append((ev.xdata, ev.ydata))
        redraw()

    def on_key(ev):
        if ev.key == "enter":
            plt.close(fig)
        elif ev.key == "u":
            if neg and (not pos or True):
                pass
            if pos:
                pos.pop()
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="scene 圖片資料夾（.jpg/.png）")
    ap.add_argument("--out_dir", required=True, help="mask 輸出資料夾")
    ap.add_argument("--frame_idx", type=int, default=0, help="在第幾張圖上標點")
    ap.add_argument("--points", type=str, default="", help='前景點 "x,y;x,y"')
    ap.add_argument("--neg_points", type=str, default="", help='背景點 "x,y;x,y"')
    ap.add_argument("--ckpt", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--preview", action="store_true", help="輸出 overlay 預覽圖")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 收集圖片（sorted，與 pipeline 讀取順序一致）─────────────
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    img_paths = sorted(p for e in exts for p in img_dir.glob(e))
    # 去重（大小寫 glob 可能重複）
    seen, uniq = set(), []
    for p in img_paths:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    img_paths = uniq
    assert img_paths, f"❌ {img_dir} 裡找不到圖片"
    print(f"📁 找到 {len(img_paths)} 張圖，第一張: {img_paths[0].name}")

    # ── 2. SAM2 的 video predictor 需要「數字命名的 JPEG 序列」資料夾 ──
    #     建 temp dir，把圖轉成 00000.jpg, 00001.jpg ...（PNG 會轉存成 JPEG）
    tmp = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    print(f"🔧 建立 frame 序列 → {tmp}")
    for i, p in enumerate(img_paths):
        dst = tmp / f"{i:05d}.jpg"
        if p.suffix.lower() in (".jpg", ".jpeg"):
            os.symlink(p.resolve(), dst)
        else:
            img = cv2.imread(str(p))
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # ── 3. 取得標點 ─────────────────────────────────────────────
    ref_img = cv2.imread(str(img_paths[args.frame_idx]))
    H, W = ref_img.shape[:2]
    print(f"🖼️  標點參考圖: {img_paths[args.frame_idx].name} ({W}x{H})")

    pos = parse_pts(args.points)
    neg = parse_pts(args.neg_points)
    if not pos:
        print("🖱️  進入互動標點（左鍵=物件 右鍵=背景 Enter=完成 u=復原）...")
        pos, neg = interactive_pick(ref_img)
    assert pos, "❌ 至少需要一個前景點（--points 或互動點選）"
    print(f"   前景點: {pos}")
    print(f"   背景點: {neg}")

    # ── 4. 載入 SAM2 video predictor ─────────────────────────────
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 載入 SAM2 ({args.ckpt}) on {device} ...")
    predictor = build_sam2_video_predictor(args.model_cfg, args.ckpt, device=device)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(tmp))

        pts = np.array(pos + neg, dtype=np.float32)
        lbl = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=args.frame_idx, obj_id=1,
            points=pts, labels=lbl,
        )

        # ── 5. 雙向傳播（先往後，再從標點 frame 往前）────────────
        masks = {}
        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            m = (logits[0] > 0.0).squeeze().cpu().numpy().astype(np.uint8) * 255
            masks[fidx] = m
        for fidx, obj_ids, logits in predictor.propagate_in_video(
            state, reverse=True
        ):
            m = (logits[0] > 0.0).squeeze().cpu().numpy().astype(np.uint8) * 255
            masks[fidx] = m

    # ── 6. 存檔（與原圖同 stem 的 .png，sorted 順序一致）──────────
    prev_dir = out_dir.parent / "mask_preview"
    if args.preview:
        prev_dir.mkdir(parents=True, exist_ok=True)

    n_empty = 0
    for i, p in enumerate(img_paths):
        m = masks.get(i)
        if m is None:
            m = np.zeros((H, W), np.uint8)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out_path = out_dir / (p.stem + ".png")
        cv2.imwrite(str(out_path), m)
        if m.max() == 0:
            n_empty += 1
        if args.preview:
            img = cv2.imread(str(p))
            overlay = img.copy()
            overlay[m > 127] = (0.4 * overlay[m > 127] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(str(prev_dir / (p.stem + "_ov.jpg")), overlay)

    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n✅ 完成！{len(img_paths)} 張 mask → {out_dir}")
    if n_empty:
        print(f"⚠️  有 {n_empty} 張 mask 全黑（物件可能在該視角不可見或追蹤丟失）")
        print("   → 若物件其實可見，換一張更清楚的圖標點：--frame_idx N 重跑")
    if args.preview:
        print(f"🖼️  預覽 overlay → {prev_dir}（紅色=mask，逐張檢查有沒有跟丟）")


if __name__ == "__main__":
    main()