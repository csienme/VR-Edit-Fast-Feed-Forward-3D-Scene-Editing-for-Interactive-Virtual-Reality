import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

try:
    from simple_lama_inpainting import SimpleLama
except ImportError:
    print("⚠️ 尚未安裝 simple-lama-inpainting，請執行 pip install simple-lama-inpainting")
    exit(1)

def run_ablation_independent_inpaint(args):
    print("="*60)
    print("🔬 [Ablation Study] 啟動單視角獨立修補 (Independent 2D Inpainting Baseline)")
    print("="*60)

    # 1. 建立輸出資料夾
    data_dir = Path(args.data_path)
    mask_dir = Path(args.mask_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 獲取所有圖片並排序
    valid_exts = {'.png', '.jpg', '.jpeg'}
    image_files = sorted([f for f in os.listdir(data_dir) if Path(f).suffix.lower() in valid_exts])
    
    if not image_files:
        print(f"❌ 在 {data_dir} 中找不到圖片！")
        return

    print(f"🖼️ 找到 {len(image_files)} 張圖片準備進行獨立修補")

    # 3. 載入 LaMa 模型
    print("⏳ 正在載入 LaMa 模型至 GPU...")
    lama_model = SimpleLama()
    print("✅ LaMa 模型載入完成！\n")

    # 4. 為了公平比較，使用與主實驗相同的膨脹參數
    kernel = np.ones((15, 15), np.uint8)

    # 5. 逐幀獨立修補
    for img_name in tqdm(image_files, desc="Independent Inpainting"):
        img_path = data_dir / img_name
        mask_path = mask_dir / img_name
        
        # 檢查 Mask 是否存在
        if not mask_path.exists():
            print(f"⚠️ 找不到對應的 Mask: {mask_path}，跳過此圖。")
            continue

        # 讀取影像與 Mask
        img_pil = Image.open(str(img_path)).convert('RGB')
        mask_cv = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 尺寸對齊防呆
        W, H = img_pil.size
        if mask_cv.shape[:2] != (H, W):
            mask_cv = cv2.resize(mask_cv, (W, H), interpolation=cv2.INTER_NEAREST)

        # 執行 Mask 膨脹 (消滅邊緣陰影與滲漏)
        mask_dilated = cv2.dilate(mask_cv, kernel, iterations=1)
        mask_pil = Image.fromarray(mask_dilated).convert('L')

        # LaMa 獨立推論
        result_pil = lama_model(img_pil, mask_pil)
        
        # 轉回 OpenCV 格式並強制尺寸對齊 (對抗 LaMa Padding)
        inpainted_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        if inpainted_bgr.shape[:2] != (H, W):
            inpainted_bgr = cv2.resize(inpainted_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        # 存檔
        out_file_path = output_dir / img_name
        cv2.imwrite(str(out_file_path), inpainted_bgr)

    print("\n" + "="*60)
    print(f"🎉 Ablation Baseline 處理完成！所有獨立修補影像已存至: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study: Independent LaMa Inpainting")
    parser.add_argument("--data_path", type=str, required=True, help="原始圖片資料夾路徑")
    parser.add_argument("--mask_path", type=str, required=True, help="Mask 資料夾路徑")
    parser.add_argument("--output_dir", type=str, default="ablation_independent_results", help="獨立修補後的輸出資料夾")
    
    args = parser.parse_args()
    run_ablation_independent_inpaint(args)