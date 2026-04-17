"""
test_inpaint_2d.py

純 2D Inpainting 測試：不使用任何 3D 幾何條件，
直接讓 SD Inpaint 補黑色遮罩區域，用來排除 ControlNet 深度引導的問題。

用法：
  python test_inpaint_2d.py --img_path <圖片路徑> [--output_path result.png]

黑色區域（RGB 全為 0）會被自動偵測為 mask。
"""

import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler


# ──────────────────────────────────────────────
# 載入模型（與 generative_inpaint_module.py 相同的風格）
# ──────────────────────────────────────────────
print("⏳ 正在載入 SD Inpainting 模型至 GPU...")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
print("✅ 模型載入完成！")


# ──────────────────────────────────────────────
# 主函式
# ──────────────────────────────────────────────

def run_test(img_path: str, output_path: str):
    # 1. 讀圖
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f"❌ 找不到圖片: {img_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W   = img_rgb.shape[:2]
    print(f"  圖片尺寸: {W}×{H}")

    # 2. 偵測黑色遮罩（RGB 三通道都 < 10）
    black_mask = np.all(img_rgb < 10, axis=2).astype(np.uint8) * 255
    n_black    = np.count_nonzero(black_mask)
    assert n_black > 0, "❌ 圖片中沒有偵測到黑色區域，請確認 mask 是否正確塗黑"
    print(f"  偵測到黑色遮罩像素: {n_black} ({n_black / (H*W) * 100:.1f}%)")

    # 3. 膨脹 + 模糊 mask（與 generative_inpaint_module.py 相同）
    kernel        = np.ones((9, 9), np.uint8)
    mask_dilated  = cv2.dilate(black_mask, kernel, iterations=1)
    mask_blurred  = cv2.GaussianBlur(mask_dilated, (15, 15), 0)

    # 4. 將黑色區域塗成中性灰（消除 VAE 記憶，與 module 相同）
    img_filled        = img_rgb.copy()
    img_filled[black_mask > 0] = [127, 127, 127]

    # 5. 轉 PIL
    img_pil  = Image.fromarray(img_filled)
    mask_pil = Image.fromarray(mask_blurred).convert("L")

    # 6. SD Inpaint（純 2D，無 ControlNet 深度條件）
    prompt = (
        "realistic background, seamless surface texture, "
        "consistent lighting, photorealistic"
    )
    negative_prompt = (
        "objects, blurry, artifacts, unnatural, floating, "
        "distorted, low quality, cardboard box, bag"
    )

    print("  🎨 開始 2D Inpainting（無深度條件）...")
    # SD 輸入尺寸必須是 8 的倍數，先 resize 再輸出時還原
    sd_w = (W // 8) * 8
    sd_h = (H // 8) * 8
    result = pipe(
        prompt              = prompt,
        negative_prompt     = negative_prompt,
        image               = img_pil.resize((sd_w, sd_h), Image.LANCZOS),
        mask_image          = mask_pil.resize((sd_w, sd_h), Image.NEAREST),
        height              = sd_h,
        width               = sd_w,
        num_inference_steps = 25,
        guidance_scale      = 7.5,
    ).images[0]
    # 還原原始解析度
    result = result.resize((W, H), Image.LANCZOS)

    # 7. 輸出
    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"✅ 結果儲存至: {output_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path",    type=str, required=True,
                        help="含黑色遮罩的輸入圖片路徑")
    parser.add_argument("--output_path", type=str, default="inpaint_2d_result.png",
                        help="輸出圖片路徑（預設: inpaint_2d_result.png）")
    args = parser.parse_args()
    run_test(args.img_path, args.output_path)