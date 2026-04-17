"""
apply_mask_to_train.py

把 SPInNeRF trash 場景後60張有物體的圖片，用對應的 mask 蓋黑後輸出。

mask 值是 0/1（不是 0/255），mask=1 的區域是物體，蓋成黑色。
"""

import os
import cv2
import numpy as np
from pathlib import Path

IMG_DIR  = Path("spinnerf-dataset/1/images_4")
MASK_DIR = Path("spinnerf-dataset/1/images_4/label")
OUT_DIR  = Path("masked_train_1")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 取所有 png/jpg，按字母排序，跳過 label 子目錄
img_files = sorted([
    f for f in IMG_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')
])

mask_files = sorted([
    f for f in MASK_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')
])

# 前40張是GT（無物體），後60張是有物體的 train frames
train_imgs  = img_files[40:]   # 後60張
train_masks = mask_files       # label/ 下剛好對應這60張

assert len(train_imgs) == 60,  f"預期60張 train image，實際 {len(train_imgs)} 張"
assert len(train_masks) == 60, f"預期60張 mask，實際 {len(train_masks)} 張"

print(f"Image dir : {IMG_DIR}")
print(f"Mask dir  : {MASK_DIR}")
print(f"Output dir: {OUT_DIR}")
print(f"Processing {len(train_imgs)} images...\n")

for i, (img_path, mask_path) in enumerate(zip(train_imgs, train_masks)):
    img  = cv2.imread(str(img_path))           # BGR, uint8
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 0/1

    # mask 可能是 0/1，也可能是 0/255，統一轉成 bool
    if mask.max() <= 1:
        mask_bool = mask.astype(bool)
    else:
        mask_bool = mask > 127

    # resize mask 到 img 尺寸（以防解析度不同）
    if mask.shape[:2] != img.shape[:2]:
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8), (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    # mask=1 的區域（物體）蓋成黑色
    masked_img = img.copy()
    masked_img[mask_bool] = 0

    out_name = f"inpainted_{i}.png"
    out_path = OUT_DIR / out_name
    cv2.imwrite(str(out_path), masked_img)

    print(f"[{i:02d}] {img_path.name} + {mask_path.name} → {out_name}")

print(f"\n✅ 完成！輸出到 {OUT_DIR}")