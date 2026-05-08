"""
train_render.py  —  Mask-Weighted Patch-LPIPS 版

核心改進（v4，直接針對 m-LPIPS 優化）：

  [Loss]
  1. Mask-weighted L1：inpainted region 上權 --fg_weight 倍（預設 8x）
     → 訓練梯度集中在被 m-LPIPS 評估的 bbox region
  2. Patch-based LPIPS：在 mask centroid 附近採樣 --patch_size × --patch_size 的 crop
     → 降低 full-image LPIPS 梯度 noise，同時聚焦 inpainted region
  3. Loss 配方：0.70 × L1_masked + 0.15 × SSIM + 0.15 × LPIPS_patch

  [Schedule]
  4. total_iters = 15000（LR 已修正後 loss 能正常下降，拉長有效益）
  5. xyz LR：0.00016×lr_extent → 0.0000016×lr_extent（lr_extent = max(extent,1.0)）
  6. Densification 窗口：500 ~ 10000
  7. reset_opacity：iter 3000、iter 7500（兩次，對應 15000 iters 均勻分佈）
  8. SH ramp-up：每 1000 iters

  [Debug]
  - 每 1000 iters 印出 l1_raw / l1_masked / ssim / lpips_patch / n_gaussians
  - Mask 載入摘要：coverage 統計（mean/min/max % over all train views）
  - 訓練前三次 patch 採樣座標，確認有 hit 到 mask region
  - 訓練結束後 opacity 分布（mean / median / >0.5 比例）

用法：
  python train_render.py \\
      --colmap_dir    purify_3              \\
      --nvs_pose      purify_hybrid_3       \\
      --train_img_dir purify_3/images       \\
      --mask_dir      purify_3/masks        \\   # 可選；有才啟動 mask weighting + patch LPIPS
      --output_dir    ./renders_3

  其他可調參數：
      --total_iters 15000   # 訓練步數（預設 15000）
      --fg_weight   8.0     # mask region 上權倍數（預設 8.0，可試 5/8/10）
      --patch_size  256     # LPIPS patch 邊長（預設 256）
"""

import os, struct, math, random, glob
import cv2, numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import save_image

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer.render import render
import lpips as lpips_module   # 避免與 local 變數名衝突


# ═══════════════════════════════════════════════════════════
#  COLMAP Binary Reader
# ═══════════════════════════════════════════════════════════

def _read_cameras_bin(path):
    MODEL_NPARAMS = {0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12}
    MODEL_NAMES   = {0:"SIMPLE_PINHOLE",1:"PINHOLE",2:"SIMPLE_RADIAL",
                     3:"RADIAL",4:"OPENCV",5:"OPENCV_FISHEYE",6:"FULL_OPENCV",
                     7:"FOV",8:"SIMPLE_RADIAL_FISHEYE",9:"RADIAL_FISHEYE",
                     10:"THIN_PRISM_FISHEYE"}
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid = struct.unpack("<I", f.read(4))[0]
            mid = struct.unpack("<I", f.read(4))[0]
            w   = struct.unpack("<Q", f.read(8))[0]
            h   = struct.unpack("<Q", f.read(8))[0]
            np_ = MODEL_NPARAMS[mid]
            p   = np.array(struct.unpack(f"<{np_}d", f.read(8 * np_)))
            cams[cid] = {"model": MODEL_NAMES[mid], "width": w, "height": h, "params": p}
    return cams


def _read_images_bin(path):
    results = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid             = struct.unpack("<I",  f.read(4))[0]
            qw,qx,qy,qz    = struct.unpack("<4d", f.read(32))
            tx,ty,tz        = struct.unpack("<3d", f.read(24))
            cid             = struct.unpack("<I",  f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            np_ = struct.unpack("<Q", f.read(8))[0]
            f.read(np_ * 24)
            R = _quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            results.append({"name": name.decode(), "camera_id": cid,
                             "R": R, "t": t, "center": -R.T @ t})
    return results


def _quat_to_rot(qw, qx, qy, qz):
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw),1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),1-2*(qx**2+qy**2)],
    ])


# ═══════════════════════════════════════════════════════════
#  Lightweight PoseOnly Camera（GT poses 用）
# ═══════════════════════════════════════════════════════════

class PoseOnlyCamera:
    def __init__(self, name, R_wc, t_wc, FoVx, FoVy, width, height, device="cuda"):
        self.image_name    = name
        self.FoVx          = float(FoVx)
        self.FoVy          = float(FoVy)
        self.image_width   = int(width)
        self.image_height  = int(height)
        self.original_image = None

        wvt = np.eye(4, dtype=np.float32)
        wvt[:3, :3] = R_wc.T
        wvt[3,  :3] = t_wc
        self.world_view_transform = torch.tensor(wvt, device=device)
        self.projection_matrix    = self._proj(FoVx, FoVy, device)
        self.full_proj_transform  = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = torch.tensor(-R_wc.T @ t_wc,
                                          dtype=torch.float32, device=device)

    @staticmethod
    def _proj(FoVx, FoVy, device, znear=0.01, zfar=100.0):
        tHX = math.tan(FoVx / 2); tHY = math.tan(FoVy / 2)
        top=tHY*znear; bot=-top; right=tHX*znear; left=-right
        P = torch.zeros(4, 4)
        P[0,0]=2*znear/(right-left); P[1,1]=2*znear/(top-bot)
        P[0,2]=(right+left)/(right-left); P[1,2]=(top+bot)/(top-bot)
        P[3,2]=1.0
        P[2,2]=zfar/(zfar-znear); P[2,3]=-(zfar*znear)/(zfar-znear)
        return P.transpose(0,1).to(device)


def load_gt_cameras_from_colmap(colmap_root, device="cuda"):
    """
    讀取 sparse/ 中的 camera。
      - 模式 A（SPInNeRF dual-VGGT）：只收非 inpainted_ 開頭的 GT camera
      - 模式 B（單 COLMAP 場景）：若篩完空集合，fallback 收全部 camera
    """
    sparse_dir = os.path.join(colmap_root, "sparse")
    if not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        sparse_dir = os.path.join(sparse_dir, "0")

    cam_meta = _read_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
    img_meta = _read_images_bin(os.path.join(sparse_dir, "images.bin"))

    def _build_camera(img):
        cm    = cam_meta[img["camera_id"]]
        W, H  = cm["width"], cm["height"]
        p     = cm["params"]
        model = cm["model"]
        if model == "PINHOLE":
            fx, fy = p[0], p[1]
        else:
            fx = fy = p[0]
        FoVx = 2 * np.arctan(W / (2 * fx))
        FoVy = 2 * np.arctan(H / (2 * fy))
        return PoseOnlyCamera(
            name=img["name"], R_wc=img["R"], t_wc=img["t"],
            FoVx=FoVx, FoVy=FoVy, width=W, height=H, device=device,
        )

    # 模式 A：先嘗試 SPInNeRF 過濾
    cameras = [_build_camera(img) for img in img_meta
               if not img["name"].startswith("inpainted_")]

    # 模式 B fallback：所有 image 都是 inpainted_，代表單 COLMAP 場景
    if len(cameras) == 0:
        print("    ℹ️  [Single-COLMAP mode] 偵測到全 inpainted_ 檔名，render 將使用所有 camera（即 train poses）")
        cameras = [_build_camera(img) for img in img_meta]

    cameras.sort(key=lambda c: c.image_name)
    return cameras


# ═══════════════════════════════════════════════════════════
#  Mask Utilities
# ═══════════════════════════════════════════════════════════

def load_masks_sorted(mask_dir, cameras, target_w, target_h):
    """
    從 mask_dir 按檔名排序，依序對應 cameras list（順序必須正確）。
    Mask 已為 0/1 二值，直接轉 float32，不做閾值判斷。
    為每個 camera 設定 .mask = (1, H, W) float32 tensor。
    回傳 sorted 檔名 list（供 debug）。
    """
    mask_files = sorted(
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(('.png', '.jpg'))
    )
    assert len(mask_files) == len(cameras), (
        f"Mask 數量（{len(mask_files)}）與 camera 數量（{len(cameras)}）不符\n"
        f"  mask_dir 有 {len(mask_files)} 檔，cameras 有 {len(cameras)} 個"
    )
    for cam, fname in zip(cameras, mask_files):
        path = os.path.join(mask_dir, fname)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert m is not None, f"無法讀取 mask: {path}"
        m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        # 已為 0/1 二值：直接 astype float，不做 >127 閾值
        cam.mask = torch.tensor(m.astype(np.float32), device="cuda").unsqueeze(0)
    return mask_files


def get_mask_centroid(mask):
    """回傳 mask 重心 (cy, cx)；mask 全黑時回傳 None。"""
    if mask is None:
        return None
    ys, xs = torch.nonzero(mask[0] > 0.5, as_tuple=True)
    if len(ys) == 0:
        return None
    return int(ys.float().mean().item()), int(xs.float().mean().item())


def sample_patch_on_mask(mask, H, W, patch=256, jitter=50):
    """
    在 mask centroid 附近（±jitter pixels 隨機偏移）採樣一個 patch×patch 的 crop。
    若沒有 mask，則在全圖隨機採樣。
    回傳 (y0, y1, x0, x1)。
    """
    centroid = get_mask_centroid(mask)
    if centroid is not None:
        cy, cx = centroid
        cy += random.randint(-jitter, jitter)
        cx += random.randint(-jitter, jitter)
    else:
        cy = random.randint(patch // 2, H - patch // 2)
        cx = random.randint(patch // 2, W - patch // 2)

    y0 = int(max(0, min(cy - patch // 2, H - patch)))
    x0 = int(max(0, min(cx - patch // 2, W - patch)))
    return y0, y0 + patch, x0, x0 + patch


# ═══════════════════════════════════════════════════════════
#  Loss Functions
# ═══════════════════════════════════════════════════════════

def masked_l1(img, gt, mask, fg_w=8.0):
    """
    Per-pixel L1 with foreground up-weighting.
    mask: (1,H,W) float，1=inpainted region。None 則退化為普通 L1。
    fg_w: inpainted region 的權重倍數（bg=1.0）。

    [設計意圖]
    Baselines（GScream）在 inpainted region 用 5~10× 的額外 loss 權重，
    使訓練梯度集中在被 m-LPIPS 評估的 bbox 區域。
    """
    diff = (img - gt).abs()           # (3, H, W)
    if mask is None:
        return diff.mean()
    w = (1.0 - mask) * 1.0 + mask * fg_w   # (1, H, W)，broadcast over 3ch
    return (diff * w).sum() / (w.sum() * diff.shape[0])


def ssim_loss(img1, img2, ws=11, C1=1e-4, C2=9e-4):
    """1 - SSIM（全圖）。返回值越小越好。"""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0); img2 = img2.unsqueeze(0)
    C = img1.shape[1]
    x = torch.arange(ws, dtype=torch.float32, device=img1.device) - ws // 2
    g = torch.exp(-x**2 / 4.5); g = g / g.sum()
    k = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(C,1,ws,ws)
    pad = ws // 2
    def conv(t): return F.conv2d(t, k, padding=pad, groups=C)
    m1 = conv(img1);  m2 = conv(img2)
    s1 = conv(img1*img1)-m1**2; s2 = conv(img2*img2)-m2**2
    s12 = conv(img1*img2)-m1*m2
    num = (2*m1*m2+C1)*(2*s12+C2)
    den = (m1**2+m2**2+C1)*(s1+s2+C2)
    return 1 - (num/den).mean()


def patch_lpips(lpips_fn, img, gt, mask, H, W, patch=256, jitter=50,
                log_patch=False):
    """
    在 mask centroid 附近採樣一個 patch，計算 LPIPS。
    [設計意圖]
    Full-image VGG LPIPS 梯度 noise 很大（spatial receptive field ~40px），
    mask 附近的 patch-level LPIPS 梯度 signal-to-noise 更高，
    且直接對應 m-LPIPS 評估範圍。

    回傳 (loss_value, (y0,y1,x0,x1))；後者供 debug 使用。
    """
    y0, y1, x0, x1 = sample_patch_on_mask(mask, H, W, patch=patch, jitter=jitter)
    img_p = img[:, y0:y1, x0:x1].unsqueeze(0)   # (1,3,P,P)
    gt_p  = gt [:, y0:y1, x0:x1].unsqueeze(0)
    lp = lpips_fn(img_p * 2 - 1, gt_p * 2 - 1).mean()
    return lp, (y0, y1, x0, x1)


def get_lr_func(lr0, lr1, steps):
    """Exponential decay from lr0 to lr1 over `steps` iters."""
    def f(i):
        if i < 0 or lr0 == lr1 == 0: return 0.
        if i >= steps: return lr1
        t = np.clip(i / steps, 0, 1)
        return float(np.exp(np.log(lr0)*(1-t) + np.log(lr1)*t))
    return f


class Pipe:
    compute_cov3D_python = False
    convert_SHs_python   = False
    debug                = False


def sort_inpainted(cams):
    return sorted(cams,
                  key=lambda c: int(c.image_name.rsplit('.', 1)[0].split('_')[-1]))


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def train_and_render(args):
    TOTAL_ITERS = args.total_iters
    FG_WEIGHT   = args.fg_weight
    PATCH_SIZE  = args.patch_size

    print("🚀 3DGS 訓練與渲染（Mask-Weighted Patch-LPIPS 版）")
    print(f"   total_iters={TOTAL_ITERS}  fg_weight={FG_WEIGHT}  patch_size={PATCH_SIZE}")
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = Pipe()
    bg   = torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda")

    # ── Step 1: Point cloud + train poses ─────────────────────────────
    print(f"\n☁️  Step 1: Point cloud + train poses ← {args.colmap_dir}")
    gaussians    = GaussianModel(sh_degree=3)
    scene60      = Scene(args.colmap_dir, gaussians, shuffle=False)
    train_cameras = sort_inpainted(scene60.getTrainCameras())
    del scene60
    print(f"  Gaussians   : {gaussians.get_xyz.shape[0]}")
    print(f"  Train cams  : {len(train_cameras)}  "
          f"[{train_cameras[0].image_name} … {train_cameras[-1].image_name}]")

    # [DEBUG] Train camera 解析度 & FoV
    tc0 = train_cameras[0]
    print(f"\n[DEBUG] Train cam[0] resolution : {tc0.image_width} × {tc0.image_height}")
    print(f"[DEBUG] Train cam[0] FoVx={np.degrees(tc0.FoVx):.2f}°  "
          f"FoVy={np.degrees(tc0.FoVy):.2f}°")

    # ── Step 2: GT poses ───────────────────────────────────────────────
    print(f"\n📐 Step 2: GT poses ← {args.nvs_pose}/sparse/")
    test_cameras = load_gt_cameras_from_colmap(args.nvs_pose, device="cuda")
    assert len(test_cameras) > 0, "找不到 GT cameras"
    print(f"  GT test cams: {len(test_cameras)}  "
          f"[{test_cameras[0].image_name} … {test_cameras[-1].image_name}]")

    # [DEBUG] GT camera 解析度 & FoV，以及與 train cam 的差異
    gc0 = test_cameras[0]
    print(f"\n[DEBUG] GT cam[0] resolution : {gc0.image_width} × {gc0.image_height}")
    print(f"[DEBUG] GT cam[0] FoVx={np.degrees(gc0.FoVx):.2f}°  "
          f"FoVy={np.degrees(gc0.FoVy):.2f}°")

    gt_img_candidates = (
        glob.glob(os.path.join(args.gt_img_dir, "*.png")) +
        glob.glob(os.path.join(args.gt_img_dir, "*.jpg"))
    ) if args.gt_img_dir else []
    if gt_img_candidates:
        _s = cv2.imread(sorted(gt_img_candidates)[0])
        if _s is not None:
            print(f"[DEBUG] GT image on disk     : {_s.shape[1]} × {_s.shape[0]}")
            ok = (_s.shape[1] == gc0.image_width and _s.shape[0] == gc0.image_height)
            print(f"[DEBUG] {'✅ Resolution match OK' if ok else '⚠️  MISMATCH'}")
    else:
        print("[DEBUG] --gt_img_dir not provided, skip disk resolution check")

    dfovx = abs(np.degrees(tc0.FoVx) - np.degrees(gc0.FoVx))
    dfovy = abs(np.degrees(tc0.FoVy) - np.degrees(gc0.FoVy))
    print(f"\n[DEBUG] FoV diff train vs GT: ΔFoVx={dfovx:.3f}°  ΔFoVy={dfovy:.3f}°")
    print(f"[DEBUG] {'✅ FoV 一致' if dfovx < 2.0 else '⚠️  FoV 差距 > 2°，兩次 VGGT run 的 intrinsics 可能不一致'}")

    # ── Step 3: 載入 inpainted 訓練圖片 ───────────────────────────────
    print(f"\n🖼️  Step 3: Loading inpainted images ← {args.train_img_dir}")
    raw = [f for f in os.listdir(args.train_img_dir)
           if f.lower().endswith(('.png', '.jpg'))]
    purify_files = sorted(raw,
                          key=lambda f: int(f.rsplit('.', 1)[0].split('_')[-1]))
    assert len(purify_files) == len(train_cameras), (
        f"圖片數量不符：{len(purify_files)} 張圖 vs {len(train_cameras)} 個 cameras")

    for cam, fname in zip(train_cameras, purify_files):
        path = os.path.join(args.train_img_dir, fname)
        img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, (cam.image_width, cam.image_height))
        cam.original_image = (torch.tensor(img, dtype=torch.float32)
                              .permute(2, 0, 1).cuda() / 255.)
    _H, _W = train_cameras[0].original_image.shape[1:]
    print(f"  {purify_files[0]} … {purify_files[-1]}")
    print(f"[DEBUG] Train image tensor size: {_W} × {_H}")

 # ── Step 3.5: 載入 Masks（對應 60 張 train_cameras）────────────────
    # mask_dir 有 60 張 mask（SPInNeRF label），按 sorted 順序對應 train_cameras
    USE_MASK = (args.mask_dir is not None)
    if USE_MASK:
        print(f"\n🎭 Step 3.5: Loading masks ← {args.mask_dir}")
        mask_files = load_masks_sorted(
            args.mask_dir, train_cameras, _W, _H   # 一次傳整個 list
        )
        print(f"  Loaded {len(mask_files)} masks → {mask_files[0]} … {mask_files[-1]}")

        # [DEBUG] Coverage 統計
        coverages = [cam.mask.mean().item() * 100 for cam in train_cameras]
        print(f"  Coverage: mean={np.mean(coverages):.1f}%  "
              f"min={np.min(coverages):.1f}%  max={np.max(coverages):.1f}%")
        if np.mean(coverages) > 40:
            print("[DEBUG] ⚠️  Mean coverage > 40%，確認 mask 是否只標記物件區域")

        # [DEBUG] 第一張 mask centroid & bbox
        _m0 = train_cameras[0].mask
        ys, xs = torch.nonzero(_m0[0] > 0.5, as_tuple=True)
        if len(ys) > 0:
            print(f"[DEBUG] Mask[0] centroid=({int(ys.float().mean())}, {int(xs.float().mean())})  "
                  f"bbox={int(ys.max()-ys.min())}×{int(xs.max()-xs.min())}  "
                  f"nonzero={len(ys)} px")
    else:
        print("\n[INFO] --mask_dir not provided → uniform loss")
        for cam in train_cameras:
            cam.mask = None

    # test_cameras 無 mask（render 用，不參與 training loss）
    for cam in test_cameras:
        cam.mask = None

    # ── Step 4: Optimizer ──────────────────────────────────────────────
    centers    = torch.stack([c.camera_center for c in train_cameras])
    extent     = torch.norm(centers - centers.mean(0), dim=-1).max().item() * 1.1
    lr_extent  = max(extent, 1.0)        # 防止 VGGT 極小 extent 把 xyz LR 壓爛
    print(f"\n🌍 Scene extent: {extent:.4f}  lr_extent: {lr_extent:.4f}")
    if extent < 0.5:
        print("[DEBUG] ⚠️  extent < 0.5，VGGT scene scale 與 COLMAP 差異大，"
              "已啟用 lr_extent=max(extent,1.0) 保護")

    opt = torch.optim.Adam([
        {"params": [gaussians._xyz],           "lr": 0.00016 * lr_extent, "name": "xyz"},
        {"params": [gaussians._features_dc],   "lr": 0.0025,              "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": 0.000005,            "name": "f_rest"},
        {"params": [gaussians._opacity],       "lr": 0.05,                "name": "opacity"},
        {"params": [gaussians._scaling],       "lr": 0.005,               "name": "scaling"},
        {"params": [gaussians._rotation],      "lr": 0.001,               "name": "rotation"},
    ], lr=0., eps=1e-15)
    gaussians.optimizer = opt

    lr_fn    = get_lr_func(0.00016 * lr_extent, 0.0000016 * lr_extent, TOTAL_ITERS)
    lpips_fn = lpips_module.LPIPS(net="vgg").cuda().eval()
    for _p in lpips_fn.parameters():
        _p.requires_grad_(False)

    # ── Step 5: 訓練 ───────────────────────────────────────────────────
    print(f"\n🔥 Training on {len(train_cameras)} inpainted views ({TOTAL_ITERS} iters) ...")
    print(f"   Loss = 0.70×L1_masked({FG_WEIGHT}x fg) + 0.15×SSIM + 0.15×LPIPS_patch({PATCH_SIZE}px)")

    cycle = train_cameras.copy()
    random.shuffle(cycle)
    ci = 0

    # Debug 追蹤用
    loss_log        = []
    patch_log_count = 0    # 前 3 次 patch 採樣座標記錄次數

    for it in tqdm(range(1, TOTAL_ITERS + 1), desc="Training"):
        # ── LR update ──
        for pg in opt.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = lr_fn(it)

        # ── Sample camera ──
        cam = cycle[ci % len(cycle)]; ci += 1
        if ci % len(cycle) == 0:
            random.shuffle(cycle)

        # ── Forward ──
        pkg = render(cam, gaussians, pipe, bg)
        img = pkg["render"]
        gt  = cam.original_image
        mask = cam.mask   # (1,H,W) or None

        # ── Loss components ──
        # L1: mask-weighted（inpainted region 上 FG_WEIGHT 倍，背景 1 倍）
        ll1_masked = masked_l1(img, gt, mask, fg_w=FG_WEIGHT)
        ll1_raw    = (img - gt).abs().mean()     # 純 L1（僅供 debug 追蹤）

        # SSIM: 全圖（mask-weighted SSIM 計算複雜，效益不大）
        ls = ssim_loss(img, gt)

        # LPIPS: 在 mask centroid 附近的 patch 計算
        do_log_patch = (patch_log_count < 3) or (it % 5000 == 0)
        lp, patch_coords = patch_lpips(
            lpips_fn, img, gt, mask, _H, _W,
            patch=PATCH_SIZE, jitter=50,
            log_patch=do_log_patch,
        )
        if do_log_patch and patch_log_count < 3:
            y0, y1, x0, x1 = patch_coords
            print(f"\n[DEBUG] Patch sample #{patch_log_count+1} at iter {it}: "
                  f"y=[{y0}:{y1}] x=[{x0}:{x1}]", end="")
            if mask is not None:
                # 確認 patch 是否真的涵蓋 mask
                patch_mask_frac = mask[0, y0:y1, x0:x1].mean().item()
                print(f"  mask_frac_in_patch={patch_mask_frac*100:.1f}%", end="")
                if patch_mask_frac < 0.05:
                    print("  ⚠️ patch barely hits mask, consider increasing jitter", end="")
            print()
            patch_log_count += 1

        # Total loss
        loss = 0.6 * ll1_masked + 0.15 * ls + 0.25 * lp
        loss.backward()

        # ── Gaussian management ──
        with torch.no_grad():
            # Densification stats（500 ~ 10000）
            if it < 10000:
                gaussians.max_radii2D[pkg["visibility_filter"]] = torch.max(
                    gaussians.max_radii2D[pkg["visibility_filter"]],
                    pkg["radii"][pkg["visibility_filter"]])
                gaussians.add_densification_stats(
                    pkg["viewspace_points"], pkg["visibility_filter"])

            opt.step()
            opt.zero_grad(set_to_none=True)

            # SH ramp-up：每 1000 iters 升一級（訓練前期更穩定）
            if it % 1000 == 0:
                gaussians.active_sh_degree = min(
                    gaussians.active_sh_degree + 1, gaussians.max_sh_degree)

            # Densify + prune（500 ~ 10000，每 100 iters）
            if 500 < it < 10000:
                if it % 100 == 0:
                    gaussians.densify_and_prune(
                        max_grad=0.0002, min_opacity=0.005,
                        extent=lr_extent, max_screen_size=20)

            # Reset opacity：iter 3000 & iter 7500（對應 15000 iters 均勻分佈）
            # 避免在 densify 窗口結束前太晚 reset 導致點來不及重長
            if it == 3000 or it == 7500:
                gaussians.reset_opacity()
                print(f"\n[DEBUG] reset_opacity at iter {it}  "
                      f"n_gaussians={gaussians.get_xyz.shape[0]}")

        # ── Debug logging（每 1000 iters）──
        if it % 1000 == 0:
            n_gs    = gaussians.get_xyz.shape[0]
            xyz_lr  = next(pg["lr"] for pg in opt.param_groups if pg["name"]=="xyz")
            entry   = {
                "iter":      it,
                "loss":      loss.item(),
                "l1_raw":    ll1_raw.item(),
                "l1_masked": ll1_masked.item(),
                "ssim":      ls.item(),
                "lpips_p":   lp.item(),
                "n_gs":      n_gs,
                "xyz_lr":    xyz_lr,
            }
            loss_log.append(entry)
            print(f"\n[DEBUG] iter={it:5d}  "
                  f"loss={entry['loss']:.4f}  "
                  f"l1_raw={entry['l1_raw']:.4f}  "
                  f"l1_masked={entry['l1_masked']:.4f}  "
                  f"ssim={entry['ssim']:.4f}  "
                  f"lpips_patch={entry['lpips_p']:.4f}  "
                  f"n_gs={n_gs}  "
                  f"xyz_lr={xyz_lr:.2e}")

    # ── Post-training diagnostics ──────────────────────────────────────
    n_final = gaussians.get_xyz.shape[0]
    op      = gaussians.get_opacity.squeeze()
    print(f"\n  Final Gaussians : {n_final}")
    print(f"[DEBUG] Opacity stats: "
          f"mean={op.mean().item():.3f}  "
          f"median={op.median().item():.3f}  "
          f">0.5: {(op>0.5).float().mean().item()*100:.1f}%  "
          f">0.1: {(op>0.1).float().mean().item()*100:.1f}%")

    sc = gaussians.get_scaling
    print(f"[DEBUG] Scale stats (log): "
          f"mean={sc.mean().item():.3f}  "
          f"median={sc.median().item():.3f}  "
          f"max={sc.max().item():.3f}")

    # Training summary table
    print("\n[DEBUG] ── Training summary ──")
    print(f"  {'iter':>6}  {'loss':>7}  {'l1_raw':>7}  "
          f"{'l1_mask':>8}  {'ssim':>6}  {'lpips_p':>8}  {'n_gs':>8}  {'xyz_lr':>8}")
    for e in loss_log:
        print(f"  {e['iter']:6d}  {e['loss']:7.4f}  {e['l1_raw']:7.4f}  "
              f"{e['l1_masked']:8.4f}  {e['ssim']:6.4f}  "
              f"{e['lpips_p']:8.4f}  {e['n_gs']:8d}  {e['xyz_lr']:8.2e}")

    # ── Step 6: Render from GT poses ──────────────────────────────────
    print(f"\n📸 Rendering from {len(test_cameras)} GT poses ...")
    with torch.no_grad():
        for cam in tqdm(test_cameras, desc="Rendering"):
            pkg = render(cam, gaussians, pipe, bg)
            save_image(pkg["render"],
                       os.path.join(args.output_dir, f"{cam.image_name}"))

    # [DEBUG] Render 輸出尺寸確認
    _renders = sorted(glob.glob(os.path.join(args.output_dir, "*.png")))
    if _renders:
        _r = cv2.imread(_renders[0])
        if _r is not None:
            out_w, out_h = _r.shape[1], _r.shape[0]
            gt_w,  gt_h  = test_cameras[0].image_width, test_cameras[0].image_height
            print(f"\n[DEBUG] Render output size : {out_w} × {out_h}")
            print(f"[DEBUG] GT cam said         : {gt_w} × {gt_h}")
            if out_w != gt_w or out_h != gt_h:
                print("[DEBUG] ⚠️  尺寸不一致，eval 時需要 resize render 到 GT 尺寸再計算 metric")
            else:
                print("[DEBUG] ✅  尺寸一致，可直接 eval")

    print(f"\n✅ Done → {args.output_dir}")


# ═══════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = ArgumentParser()

    # 必填
    p.add_argument("--colmap_dir",    required=True,
                   help="VGGT Run1（inpainted frames）→ point cloud + train poses")
    p.add_argument("--nvs_pose",      required=True,
                   help="VGGT Run2（inpainted + GT）→ 只從 sparse/ 讀 GT poses")
    p.add_argument("--train_img_dir", required=True,
                   help="Inpainted 訓練圖片目錄")
    p.add_argument("--output_dir",    required=True,
                   help="Render 輸出目錄")

    # 可選
    p.add_argument("--mask_dir",    default=None,
                   help="每張 inpainted frame 對應的 mask 目錄（有才啟動 mask-weighted loss + patch LPIPS）")
    p.add_argument("--gt_img_dir",  default=None,
                   help="GT 圖片目錄，用於 debug 確認解析度是否與 camera 一致")

    # Hyperparameters（可用於 ablation）
    p.add_argument("--total_iters", type=int,   default=15000,
                   help="訓練步數（default: 15000）")
    p.add_argument("--fg_weight",   type=float, default=8.0,
                   help="Inpainted region L1 loss 上權倍數（default: 8.0；試 5 / 8 / 10）")
    p.add_argument("--patch_size",  type=int,   default=256,
                   help="Patch-LPIPS 的 crop 邊長（default: 256）")

    train_and_render(p.parse_args())