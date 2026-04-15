from PIL import Image
import numpy as np
import os

def process_mask(input_path, output_dir):
    try:
        # 讀取圖像
        img = Image.open(input_path).convert("L")
        img_array = np.array(img)

        # 二值化
        binary_mask = np.where(img_array > 0, 255, 0).astype(np.uint8)

        # 轉回圖片
        output_img = Image.fromarray(binary_mask)

        # 建立輸出資料夾
        os.makedirs(output_dir, exist_ok=True)

        # 取得原檔名並組合輸出路徑
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        # 儲存
        output_img.save(output_path)
        print(f"處理完成，結果已保存到: {output_path}")

    except Exception as e:
        print(f"處理檔案時發生錯誤: {e}")

if __name__ == "__main__":
    input_file = "spinnerf-dataset/trash/images_4/label/20220811_093730.png"
    output_dir = "masks"

    process_mask(input_file, output_dir)
    