import cv2, numpy as np, os, torch

mask_dir = "spinnerf-dataset/3/images_4/test_label"  # 換成你的場景
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png','.jpg'))])

for fname in mask_files[:3]:
    m = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
    print(f"{fname}: shape={m.shape}, min={m.min()}, max={m.max()}, "
          f"nonzero={np.count_nonzero(m)}, unique={np.unique(m)}")