"""
dead_zone_inpainter.py
=======================
Swappable dead-zone inpainting strategies for the pull-harvest pipeline.

Strategy Pattern:
  - DeadZoneInpainter (abstract): 共同介面
  - CV2Inpainter:  Telea 演算法, 快但糊
  - LamaInpainter: LaMa, 好品質 + 確定性 (推薦)
  - SDInpainter:   Stable Diffusion + ControlNet depth, 最好品質但慢且不確定

使用方式:
  from eval.dead_zone_inpainter import build_inpainter
  inpainter = build_inpainter("lama")          # or "cv2", "sd"
  ref_cache["_inpainter"] = inpainter

擴充新策略:
  繼承 DeadZoneInpainter, 實作 inpaint() 方法,
  在 build_inpainter() 加 method 對應即可.
"""
import cv2
import numpy as np
from abc import ABC, abstractmethod


# ============================================================================
# 抽象介面
# ============================================================================
class DeadZoneInpainter(ABC):
    """Dead-zone 填補策略的共同介面。"""
    name = "base"

    @abstractmethod
    def inpaint(self, canvas: np.ndarray, dead_mask: np.ndarray,
                context: dict = None) -> np.ndarray:
        """
        canvas:    (H, W, 3) BGR uint8, 已經 pull harvest 完, mask 內絕大部分有真實 RGB
        dead_mask: (H, W) bool, True = 真死角 (沒任何 source 覆蓋, 需要 hallucinate)
        context:   optional dict, 給需要額外資訊的策略用 (e.g. SD 用 depth)
                   常見鍵: target_idx, depth, scene_d_range
        Returns:   (H, W, 3) BGR uint8, dead_mask 區域已被填好
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} ({self.name})>"


# ============================================================================
# 策略 1: OpenCV Telea (預設, 快但糊)
# ============================================================================
class CV2Inpainter(DeadZoneInpainter):
    """
    OpenCV Telea inpainting. 把鄰近 pixel 的 RGB 往內擴散.
    優點: 快 (<100ms), 內建, 確定性
    缺點: 高頻紋理會糊掉; 附近有暗色就拖暗成黑影
    """
    name = "cv2"

    def __init__(self, radius: int = 10, dilate_px: int = 3):
        self.radius = radius
        self.dilate_px = dilate_px

    # def inpaint(self, canvas, dead_mask, context=None):
    #     if not dead_mask.any():
    #         return canvas
    #     dead_u8 = dead_mask.astype(np.uint8) * 255
    #     if self.dilate_px > 0:
    #         kernel = np.ones((self.dilate_px, self.dilate_px), np.uint8)
    #         dead_u8 = cv2.dilate(dead_u8, kernel, iterations=1)
    #     return cv2.inpaint(canvas, dead_u8, self.radius, cv2.INPAINT_TELEA)
    def inpaint(self, canvas, dead_mask, context=None):
        if not dead_mask.any():
            return canvas
            
        dead_u8 = dead_mask.astype(np.uint8) * 255
        
        # --- 【升級 1: Mask 實體化 (Mask Consolidation)】 ---
        # 使用形態學閉運算，消滅內部的次像素黑點，同時不吃掉外圍真實像素
        # kernel 設為 5 或 7，足以吃掉大部分 3D 投影產生的浮點數縫隙
        close_kernel_size = 5 
        close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        dead_u8 = cv2.morphologyEx(dead_u8, cv2.MORPH_CLOSE, close_kernel)
        
        # --- 【升級 2: 處理極端 Forward-facing 場景的 AuraFusion 模式 (可選)】 ---
        # 如果你發現某些場景真的破得太誇張，可以直接啟用輪廓填充
        # contours, _ = cv2.findContours(dead_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(dead_u8, contours, -1, 255, thickness=cv2.FILLED)
        # --------------------------------------------------------

        # --- 保留微小的 Dilation (用來掩蓋拉扯邊緣的接縫) ---
        # 這裡的 dilate 只需要非常小 (例如 1-2 px)，單純用來柔和 Source 與 Inpaint 區域的交界
        if self.dilate_px > 0:
            kernel = np.ones((self.dilate_px, self.dilate_px), np.uint8)
            dead_u8 = cv2.dilate(dead_u8, kernel, iterations=1)
            
        # 最後交給生成器 (LaMa 或 CV2)
        return cv2.inpaint(canvas, dead_u8, self.radius, cv2.INPAINT_TELEA)


# ============================================================================
# 策略 2: LaMa (推薦)
# ============================================================================
class LamaInpainter(DeadZoneInpainter):
    """
    LaMa: Resolution-robust Large Mask Inpainting (WACV 2022).
    用 fast Fourier convolutions 處理大面積遮罩, 對複雜紋理表現好.

    優點: 比 cv2 好非常多 (特別是樹林、草地等高頻紋理)
          確定性 (沒有 random seed 不一致)
          中等速度 (~1-2s/view)
    缺點: 需安裝額外 package, 第一次用會下載 ~200MB checkpoint

    安裝:
      pip install simple-lama-inpainting
    """
    name = "lama"

    def __init__(self, dilate_px: int = 5, device: str = "cuda"):
        self.dilate_px = dilate_px
        self.device = device
        self._model = None  # lazy

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError:
            raise ImportError(
                "LamaInpainter 需要 simple-lama-inpainting:\n"
                "  pip install simple-lama-inpainting\n"
                "或從 source: https://github.com/advimman/lama"
            )
        print("  ⏳ 載入 LaMa (big-lama checkpoint, 第一次會下載 ~200MB) ...")
        self._model = SimpleLama(device=self.device)
        print("  ✅ LaMa 載入完成")
        return self._model

    def inpaint(self, canvas, dead_mask, context=None):
        if not dead_mask.any():
            return canvas
        from PIL import Image
        model = self._load()

        dead_u8 = dead_mask.astype(np.uint8) * 255
        if self.dilate_px > 0:
            kernel = np.ones((self.dilate_px, self.dilate_px), np.uint8)
            dead_u8 = cv2.dilate(dead_u8, kernel, iterations=1)

        # LaMa 吃 PIL RGB; canvas 是 BGR
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        mask_pil = Image.fromarray(dead_u8)
        result_pil = model(img_pil, mask_pil)
        result_rgb = np.array(result_pil)
        h_orig, w_orig = canvas.shape[:2]
        if result_rgb.shape[:2] != (h_orig, w_orig):
            result_rgb = cv2.resize(result_rgb, (w_orig, h_orig),
                                    interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    


# ============================================================================
# 策略 3: Stable Diffusion + ControlNet depth (最好品質, 慢)
# ============================================================================
class SDInpainter(DeadZoneInpainter):
    """
    Stable Diffusion inpainting + 可選 ControlNet depth guidance.

    優點: 最好的紋理生成品質, 可用 depth 引導幾何
    缺點: 慢 (~5-10s/view), 非確定性 (但給定 seed 可重現)
          有 hallucination 風險 (可能生成不存在的物體)

    Tips:
      - 用強 negative prompt 避免生成物體
      - guidance_scale 用低 (3-6), 讓 SD 偏向 passive blending 而非積極創造
      - context['depth'] 提供 VGGT depth → ControlNet 強約束幾何
    """
    name = "sd"

    def __init__(
        self,
        prompt: str =(
        "an RGB image of a seamless empty background, "
        "seamless surrounding textures, "
        "continuous surface, clean and uncluttered, "
        "empty scenery, highly detailed, photorealistic"
        ),
        negative_prompt: str = "object, statue, animal, person, blurry, distorted, dark spot",
        dilate_px: int = 10,
        num_steps: int = 25,
        guidance_scale: float = 5.0,
        use_controlnet: bool = True,
        controlnet_scale: float = 0.6,
        seed: int = 42,
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.dilate_px = dilate_px
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.use_controlnet = use_controlnet
        self.controlnet_scale = controlnet_scale
        self.seed = seed
        self._pipe = None  # lazy

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch

        if self.use_controlnet:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
                UniPCMultistepScheduler,
            )
            print("  ⏳ 載入 SD-ControlNet inpainting (第一次會下載 ~5GB) ...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                torch_dtype=torch.float16, use_safetensors=True,
            )
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True, variant="fp16",
            ).to("cuda")
        else:
            from diffusers import (
                StableDiffusionInpaintPipeline,
                UniPCMultistepScheduler,
            )
            print("  ⏳ 載入 SD inpainting (第一次會下載 ~4GB) ...")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
                use_safetensors=True, variant="fp16",
            ).to("cuda")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        self._pipe = pipe
        print("  ✅ SD 載入完成")
        return pipe

    def inpaint(self, canvas, dead_mask, context=None):
        if not dead_mask.any():
            return canvas
        import torch
        from PIL import Image

        pipe = self._load()
        H, W = canvas.shape[:2]

        # Mask dilate (給 SD 邊界 margin)
        dead_u8 = dead_mask.astype(np.uint8) * 255
        if self.dilate_px > 0:
            kernel = np.ones((self.dilate_px, self.dilate_px), np.uint8)
            dead_u8 = cv2.dilate(dead_u8, kernel, iterations=1)

        # BGR → RGB → PIL
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        mask_pil = Image.fromarray(dead_u8).convert("L")

        # SD 需要 8 的倍數
        sd_w, sd_h = (W // 8) * 8, (H // 8) * 8
        img_sd = img_pil.resize((sd_w, sd_h), Image.LANCZOS)
        mask_sd = mask_pil.resize((sd_w, sd_h), Image.NEAREST)

        # 不同 target 用不同 seed (避免每個 view 都生成同樣的紋路 → 反而不自然)
        # 但同一個 view 重跑會給相同結果
        target_idx = (context or {}).get("target_idx", 0)
        gen = torch.Generator(device="cuda").manual_seed(self.seed + target_idx)

        kwargs = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image": img_sd,
            "mask_image": mask_sd,
            "height": sd_h, "width": sd_w,
            "num_inference_steps": self.num_steps,
            "guidance_scale": self.guidance_scale,
            "generator": gen,
        }

        # ControlNet depth (如果有)
        if self.use_controlnet and context and "depth" in context:
            depth = context["depth"]
            valid_d = depth[(depth > 0) & ~np.isnan(depth)]
            if len(valid_d) > 0:
                mn, mx = valid_d.min(), valid_d.max()
                d_norm = np.clip(255 * (mx - depth) / (mx - mn + 1e-5), 0, 255)
                d_norm = d_norm.astype(np.uint8)
                d_rgb = cv2.cvtColor(d_norm, cv2.COLOR_GRAY2RGB)
                ctrl_pil = Image.fromarray(d_rgb).resize(
                    (sd_w, sd_h), Image.LANCZOS
                )
                kwargs["control_image"] = ctrl_pil
                kwargs["controlnet_conditioning_scale"] = self.controlnet_scale

        out_pil = pipe(**kwargs).images[0].resize((W, H), Image.LANCZOS)
        return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)


# ============================================================================
# Factory
# ============================================================================
def build_inpainter(method: str = "cv2", **kwargs) -> DeadZoneInpainter:
    """
    根據 method 字串建立對應的 inpainter.

    method ∈ {"cv2", "lama", "sd"}
    kwargs: 傳給對應 inpainter 的 __init__

    範例:
        build_inpainter("lama")
        build_inpainter("sd", num_steps=30, guidance_scale=4.0)
        build_inpainter("cv2", radius=15)
    """
    method = method.lower().strip()
    if method == "cv2" or method == "telea":
        return CV2Inpainter(**kwargs)
    elif method == "lama":
        return LamaInpainter(**kwargs)
    elif method == "sd" or method == "diffusion":
        return SDInpainter(**kwargs)
    else:
        raise ValueError(
            f"Unknown dead-zone inpaint method: '{method}'. "
            f"Supported: cv2, lama, sd"
        )