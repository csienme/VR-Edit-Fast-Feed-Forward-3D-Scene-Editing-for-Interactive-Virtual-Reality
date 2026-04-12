import struct
import cv2

# 讀 cameras.bin 的內參
def read_cameras_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            # SIMPLE_RADIAL: f, cx, cy, k
            params = struct.unpack("<4d", f.read(32))
            print(f"  cameras.bin → width={width}, height={height}")
            print(f"  fx=fy={params[0]:.2f}, cx={params[1]:.2f}, cy={params[2]:.2f}")

read_cameras_bin("spinnerf-dataset/trash/sparse/0/cameras.bin")

# 實際 GT 圖片尺寸
img = cv2.imread("spinnerf-dataset/trash/images_4/20220811_093524.png")
print(f"  GT image (images_4) → width={img.shape[1]}, height={img.shape[0]}")

img2 = cv2.imread("spinnerf-dataset/trash/images/20220811_093524.jpg")  # 試試看有沒有全解析度
if img2 is not None:
    print(f"  Full-res image → width={img2.shape[1]}, height={img2.shape[0]}")