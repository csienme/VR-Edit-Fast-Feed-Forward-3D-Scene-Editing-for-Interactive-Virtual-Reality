#!/usr/bin/env python3
"""
generate_masks_sam2_multi.py — SAM2 多物件 mask 生成（合併輸出單張 mask）
================================================================================
與單物件版差異：--points 改為「每個物件一組」，用 "|" 分隔物件。
每個物件獨立追蹤（獨立 obj_id），最後將所有物件的 mask OR 合併成一張輸出。

用法（兩個物件：寶特瓶 + 錢包）:
    python generate_masks_sam2_multi.py \
        --img_dir my_scene2 \
        --out_dir my_scene2_masks \
        --frame_idx 0 \
        --points "930,900;950,1300|1230,1050" \
        --neg_points "700,1100" \
        --preview

    --points 格式:  "x,y;x,y|x,y;x,y"
                    "|" 分隔不同物件，";" 分隔同一物件的多個前景點
    --neg_points 格式同上（可選）：
                    - 若含 "|" 且段數與物件數相同 → 各物件用各自的背景點
                    - 否則 → 所有背景點套用到每個物件

輸出:
    out_dir/<原圖同stem>.png   二值 mask（0/255，白=所有物件的聯集）
    --preview 時輸出 overlay 到 out_dir/../mask_preview/
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_pts(s):
    out = []
    for tok in (s or "").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        x, y = tok.split(",")
        out.append((float(x), float(y)))
    return out


def parse_multi(s):
    """'x,y;x,y|x,y' → [[(x,y),(x,y)], [(x,y)]]"""
    if not s:
        return []
    return [parse_pts(seg) for seg in s.split("|")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_idx", type=int, default=0)
    ap.add_argument("--points", type=str, required=True,
                    help='每物件一組前景點，物件間用 "|"：如 "930,900;950,1300|1230,1050"')
    ap.add_argument("--neg_points", type=str, default="",
                    help='背景點；含 "|" 且段數=物件數則各自套用，否則共用')
    ap.add_argument("--ckpt", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    img_paths = sorted(p for e in exts for p in img_dir.glob(e))
    seen, uniq = set(), []
    for p in img_paths:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    img_paths = uniq
    assert img_paths, f"❌ {img_dir} 裡找不到圖片"
    print(f"📁 找到 {len(img_paths)} 張圖，第一張: {img_paths[0].name}")

    # ── 解析多物件點 ────────────────────────────────────────────
    obj_pos = parse_multi(args.points)
    n_obj = len(obj_pos)
    assert n_obj >= 1 and all(len(p) > 0 for p in obj_pos), \
        "❌ 每個物件至少需要一個前景點"

    neg_segs = parse_multi(args.neg_points) if args.neg_points else []
    if len(neg_segs) == n_obj:
        obj_neg = neg_segs                      # 各物件各自的背景點
    else:
        shared = parse_pts(args.neg_points)     # 共用背景點
        obj_neg = [shared for _ in range(n_obj)]

    print(f"🎯 物件數: {n_obj}")
    for i, (pp, nn) in enumerate(zip(obj_pos, obj_neg), 1):
        print(f"   物件{i}: 前景 {pp} | 背景 {nn}")

    # ── frame 序列 ──────────────────────────────────────────────
    # tmp = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    # print(f"🔧 建立 frame 序列 → {tmp}")
    # for i, p in enumerate(img_paths):
    #     dst = tmp / f"{i:05d}.jpg"
    #     if p.suffix.lower() in (".jpg", ".jpeg"):
    #         os.symlink(p.resolve(), dst)
    #     else:
    #         cv2.imwrite(str(dst), cv2.imread(str(p)),
    #                     [cv2.IMWRITE_JPEG_QUALITY, 100])
    # ── frame 序列 ──────────────────────────────────────────────
    tmp = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    print(f"🔧 建立 frame 序列 → {tmp} (Baking EXIF orientation...)")
    for i, p in enumerate(img_paths):
        dst = tmp / f"{i:05d}.jpg"
        
        # 【修改處】全部強制使用 cv2 讀寫，捨棄 os.symlink
        # 這能把手機照片的 EXIF 旋轉資訊直接「燒錄」進實體像素中，
        # 確保後續 SAM2 取圖的維度與 OpenCV 抓取的 (H, W) 完全一致。
        img = cv2.imread(str(p))
        if img is None:
            print(f"❌ 警告：無法讀取圖片 {p}")
            continue
            
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    ref_img = cv2.imread(str(img_paths[args.frame_idx]))
    H, W = ref_img.shape[:2]
    print(f"🖼️  標點參考圖: {img_paths[args.frame_idx].name} ({W}x{H})")

    # ── SAM2 ────────────────────────────────────────────────────
    from sam2.build_sam import build_sam2_video_predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 載入 SAM2 ({args.ckpt}) on {device} ...")
    predictor = build_sam2_video_predictor(args.model_cfg, args.ckpt, device=device)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(tmp))

        # 每個物件用獨立 obj_id 加點
        for oid in range(n_obj):
            pts = np.array(obj_pos[oid] + obj_neg[oid], dtype=np.float32)
            lbl = np.array([1] * len(obj_pos[oid]) + [0] * len(obj_neg[oid]),
                           dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=args.frame_idx,
                obj_id=oid + 1, points=pts, labels=lbl,
            )

        # 雙向傳播；每 frame 把所有 obj 的 mask OR 起來
        masks = {}

        def merge(fidx, obj_ids, logits):
            m_union = np.zeros((logits.shape[-2], logits.shape[-1]), dtype=bool)
            for k in range(len(obj_ids)):
                m_union |= (logits[k] > 0.0).squeeze().cpu().numpy()
            masks[fidx] = m_union.astype(np.uint8) * 255

        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            merge(fidx, obj_ids, logits)
        for fidx, obj_ids, logits in predictor.propagate_in_video(state, reverse=True):
            merge(fidx, obj_ids, logits)

    # ── 存檔 ────────────────────────────────────────────────────
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
        cv2.imwrite(str(out_dir / (p.stem + ".png")), m)
        if m.max() == 0:
            n_empty += 1
        if args.preview:
            img = cv2.imread(str(p))
            ov = img.copy()
            ov[m > 127] = (0.4 * ov[m > 127] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(str(prev_dir / (p.stem + "_ov.jpg")), ov)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n✅ 完成！{len(img_paths)} 張合併 mask（{n_obj} 物件聯集）→ {out_dir}")
    if n_empty:
        print(f"⚠️  {n_empty} 張 mask 全黑（追蹤丟失？換 --frame_idx 重跑）")
    if args.preview:
        print(f"🖼️  預覽 → {prev_dir}（紅=mask，逐張檢查兩個物件都有蓋到）")


if __name__ == "__main__":
    main()