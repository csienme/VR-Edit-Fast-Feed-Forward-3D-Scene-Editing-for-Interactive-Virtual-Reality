"""
train_render_360.py  —  Bear 360° / 單 COLMAP 場景版（v5）

整合修正項目：
  [Bug Fix]
  1. SSIM 符號修正：ssim_loss() 已回傳 (1-SSIM)，直接加進 loss，不再做 1.0 - ssim_loss()
     → 前版 optimizer 在主動破壞結構，造成「塑膠融化紋理」與「透明罩層」

  [Loss]
  2. 移除 obj_mask / compute_bbox_metrics（360° bear 不需要）
  3. deadmask-weighted L1：cv2.inpaint 死角區降權 dead_weight 倍（預設 0.3）
  4. SSIM：全圖計算
  5. LPIPS：全圖隨機 patch（降低梯度 noise）
  6. Loss 配方：0.70 × L1_weighted + 0.15 × SSIM_loss + 0.15 × LPIPS_patch

  [Densification]
  7. Densify 窗口縮短至 500–12000
  8. max_grad 0.0004 → 0.0006，min_opacity 0.015 → 0.05（更嚴格，防爆炸）
  9. 硬上限 2.5M Gaussians：超過就跳過 densify，只做 opacity prune
  10. Reset opacity：iter 3000、8000（兩次，在 densify 窗口內）
  11. iter 12100 做一次最終清掃：prune 所有 opacity < 0.1 的死點
  12. Densification stats 收集窗口同步縮短至 < 12000

  [Camera sort]
  13. 相機排序改為自動偵測：先嘗試數字後綴 (inpainted_N)，失敗則 lexicographic

用法：
  python train_render_360.py \\
      --colmap_dir    purify_bear              \\
      --nvs_pose      purify_bear              \\
      --train_img_dir purify_bear/images       \\
      --deadmask_dir  inpainted_dir/deadmasks  \\   # 可選
      --output_dir    ./renders_bear           \\
      --total_iters   20000                    \\
      --dead_weight   0.3                      \\
      --patch_size    256
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
import lpips as lpips_module


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
            struct.unpack("<I", f.read(4))[0]            # image_id (unused)
            qw,qx,qy,qz = struct.unpack("<4d", f.read(32))
            tx,ty,tz    = struct.unpack("<3d", f.read(24))
            cid         = struct.unpack("<I",  f.read(4))[0]
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
                             "R": R, "t": t})
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
#  PoseOnly Camera
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
    讀 sparse/ 下的所有 camera。
    模式 A (SPInNeRF dual-VGGT)：只收非 inpainted_ 開頭的 entry 作為 test cameras。
    模式 B (bear 360° 單 COLMAP)：若篩完為空，fallback 收全部（即 train poses = test poses）。
    """
    sparse_dir = os.path.join(colmap_root, "sparse")
    if not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        sparse_dir = os.path.join(sparse_dir, "0")

    cam_meta = _read_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
    img_meta = _read_images_bin(os.path.join(sparse_dir, "images.bin"))

    def _build(img):
        cm = cam_meta[img["camera_id"]]
        W, H = cm["width"], cm["height"]
        p    = cm["params"]
        fx   = p[0] if cm["model"] == "PINHOLE" else p[0]
        fy   = p[1] if cm["model"] == "PINHOLE" else p[0]
        return PoseOnlyCamera(
            name=img["name"], R_wc=img["R"], t_wc=img["t"],
            FoVx=2*np.arctan(W/(2*fx)), FoVy=2*np.arctan(H/(2*fy)),
            width=W, height=H, device=device,
        )

    cameras = [_build(img) for img in img_meta
               if not img["name"].startswith("inpainted_")]
    if len(cameras) == 0:
        print("    ℹ️  [360° mode] 偵測到全 inpainted_ 檔名，使用所有 train poses 作為 test")
        cameras = [_build(img) for img in img_meta]

    cameras.sort(key=lambda c: c.image_name)
    return cameras


# ═══════════════════════════════════════════════════════════
#  Camera Sort（自動偵測命名格式）
# ═══════════════════════════════════════════════════════════

def sort_cameras(cams):
    """
    嘗試以數字後綴排序（inpainted_N 命名）。
    失敗則退回 lexicographic 排序（timestamp 命名如 20220819_105120.png）。
    """
    def _key(c):
        stem = c.image_name.rsplit('.', 1)[0]
        try:
            return (0, int(stem.split('_')[-1]))
        except ValueError:
            return (1, stem)
    return sorted(cams, key=_key)


# ═══════════════════════════════════════════════════════════
#  Dead Mask Utilities
# ═══════════════════════════════════════════════════════════

def load_deadmasks_sorted(deadmask_dir, cameras, target_w, target_h):
    """
    從 deadmask_dir 按檔名排序，依序對應 cameras list。
    deadmask=1 表示 cv2.inpaint 填充的死角像素（pseudo-GT 不可信，訓練時降權）。
    設定 cam.dead_mask = (1, H, W) float32 tensor。
    回傳 sorted 檔名 list（供 debug）。
    """
    files = sorted(
        f for f in os.listdir(deadmask_dir)
        if f.lower().endswith(('.png', '.jpg'))
    )
    assert len(files) == len(cameras), (
        f"Deadmask 數量（{len(files)}）與 camera 數量（{len(cameras)}）不符\n"
        f"  deadmask_dir 有 {len(files)} 檔，cameras 有 {len(cameras)} 個"
    )
    for cam, fname in zip(cameras, files):
        path = os.path.join(deadmask_dir, fname)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert m is not None, f"無法讀取 deadmask: {path}"
        m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        cam.dead_mask = torch.tensor(
            (m > 127).astype(np.float32), device="cuda"
        ).unsqueeze(0)   # (1, H, W)
    return files


# ═══════════════════════════════════════════════════════════
#  Loss Functions
# ═══════════════════════════════════════════════════════════

def weighted_l1(img, gt, dead_mask=None, dead_w=0.3):
    """
    Per-pixel L1 with dead-zone down-weighting.

    weight map：
      deadmask=0 (trustworthy)  → 1.0
      deadmask=1 (cv2.inpaint)  → dead_w  (e.g. 0.3)

    None dead_mask → 普通 L1。
    """
    diff = (img - gt).abs()           # (3, H, W)
    if dead_mask is None:
        return diff.mean(), diff.mean()

    w = torch.where(dead_mask > 0.5,
                    torch.full_like(dead_mask, dead_w),
                    torch.ones_like(dead_mask))   # (1, H, W)
    l1_weighted = (diff * w).sum() / (w.sum() * diff.shape[0])
    l1_raw      = diff.mean()
    return l1_weighted, l1_raw


def ssim_loss(img1, img2, ws=11, C1=1e-4, C2=9e-4):
    """
    回傳 (1 - SSIM)，範圍理論上 [0, 2]，正常訓練下 [0, 0.5]。
    值越小 = render 和 GT 結構越相似 = loss 越低 = 正確方向。

    [重要] 請直接把回傳值加進 loss：
        loss += w_ssim * ssim_loss(img, gt)   ✓
        loss += w_ssim * (1.0 - ssim_loss(..)) ✗  ← 符號反向！
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0); img2 = img2.unsqueeze(0)
    C = img1.shape[1]
    x = torch.arange(ws, dtype=torch.float32, device=img1.device) - ws // 2
    g = torch.exp(-x**2 / 4.5); g = g / g.sum()
    k = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(C,1,ws,ws)
    pad = ws // 2
    def conv(t): return F.conv2d(t, k, padding=pad, groups=C)
    m1 = conv(img1); m2 = conv(img2)
    s1  = conv(img1*img1) - m1**2
    s2  = conv(img2*img2) - m2**2
    s12 = conv(img1*img2) - m1*m2
    num = (2*m1*m2 + C1) * (2*s12 + C2)
    den = (m1**2 + m2**2 + C1) * (s1 + s2 + C2)
    return (1 - (num/den).mean()).clamp(min=0.0)   # clamp 防止數值上的微小負值


def random_patch_lpips(lpips_fn, img, gt, H, W, patch=256):
    """全圖隨機採樣 patch，計算 LPIPS loss。比 full-image LPIPS 快且梯度更穩定。"""
    P  = min(patch, H, W)
    y0 = random.randint(0, H - P)
    x0 = random.randint(0, W - P)
    ip = img[:, y0:y0+P, x0:x0+P].unsqueeze(0)
    gp = gt [:, y0:y0+P, x0:x0+P].unsqueeze(0)
    return lpips_fn(ip * 2 - 1, gp * 2 - 1).mean()


def get_lr_func(lr0, lr1, steps):
    """Exponential decay lr0 → lr1 over `steps` iterations."""
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


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def train_and_render(args):
    TOTAL_ITERS = args.total_iters
    DEAD_WEIGHT = args.dead_weight
    PATCH_SIZE  = args.patch_size

    # Densification 常數（集中在這裡方便日後調整）
    DENSIFY_FROM      = 500
    DENSIFY_UNTIL     = 12000      # 提早結束，避免 6.5M 那種爆炸
    DENSIFY_INTERVAL  = 100
    MAX_GAUSSIANS     = 2_500_000  # 超過此數跳過 densify，只做 opacity prune
    MAX_GRAD          = 0.0006     # 嚴格化，前版 0.0004
    MIN_OPACITY_PRUNE = 0.05       # 嚴格化，前版 0.015
    MAX_SCREEN_SIZE   = 10
    CLEANUP_ITER      = DENSIFY_UNTIL + 100   # 12100：最終 opacity 清掃
    CLEANUP_OPACITY   = 0.10                  # 清掃門檻
    RESET_ITERS       = [3000, 8000]          # densify 窗口內兩次 reset

    print("🚀 3DGS 訓練與渲染（Bear 360° / 單 COLMAP 版 v5）")
    print(f"   total_iters={TOTAL_ITERS}  dead_weight={DEAD_WEIGHT}  patch_size={PATCH_SIZE}")
    print(f"   densify {DENSIFY_FROM}–{DENSIFY_UNTIL}  max_gs={MAX_GAUSSIANS//1000}K  "
          f"max_grad={MAX_GRAD}  min_opacity={MIN_OPACITY_PRUNE}")
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = Pipe()
    bg   = torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda")

    # ── Step 1: Point cloud + train poses ─────────────────────────────
    print(f"\n☁️  Step 1: Point cloud + train poses ← {args.colmap_dir}")
    gaussians     = GaussianModel(sh_degree=3)
    scene_obj     = Scene(args.colmap_dir, gaussians, shuffle=False)
    train_cameras = sort_cameras(scene_obj.getTrainCameras())
    del scene_obj
    print(f"  Gaussians   : {gaussians.get_xyz.shape[0]:,}")
    print(f"  Train cams  : {len(train_cameras)}  "
          f"[{train_cameras[0].image_name} … {train_cameras[-1].image_name}]")

    tc0 = train_cameras[0]
    print(f"\n[DEBUG] Train cam[0] resolution : {tc0.image_width} × {tc0.image_height}")
    print(f"[DEBUG] Train cam[0] FoVx={np.degrees(tc0.FoVx):.2f}°  "
          f"FoVy={np.degrees(tc0.FoVy):.2f}°")

    # ── Step 2: GT poses（用於 render，不參與訓練）────────────────────
    print(f"\n📐 Step 2: GT / render poses ← {args.nvs_pose}")
    test_cameras = load_gt_cameras_from_colmap(args.nvs_pose, device="cuda")
    assert len(test_cameras) > 0, "找不到任何 camera"
    print(f"  Render cams : {len(test_cameras)}  "
          f"[{test_cameras[0].image_name} … {test_cameras[-1].image_name}]")

    gc0 = test_cameras[0]
    print(f"\n[DEBUG] Render cam[0] resolution : {gc0.image_width} × {gc0.image_height}")
    print(f"[DEBUG] Render cam[0] FoVx={np.degrees(gc0.FoVx):.2f}°  "
          f"FoVy={np.degrees(gc0.FoVy):.2f}°")

    gt_imgs = (glob.glob(os.path.join(args.gt_img_dir, "*.png")) +
               glob.glob(os.path.join(args.gt_img_dir, "*.jpg"))
               ) if args.gt_img_dir else []
    if gt_imgs:
        _s = cv2.imread(sorted(gt_imgs)[0])
        if _s is not None:
            ok = (_s.shape[1] == gc0.image_width and _s.shape[0] == gc0.image_height)
            print(f"[DEBUG] GT image on disk : {_s.shape[1]} × {_s.shape[0]}  "
                  f"{'✅ match' if ok else '⚠️  MISMATCH'}")
    else:
        print("[DEBUG] --gt_img_dir not provided, skip resolution check")

    dfovx = abs(np.degrees(tc0.FoVx) - np.degrees(gc0.FoVx))
    print(f"\n[DEBUG] FoV diff train vs render: ΔFoVx={dfovx:.3f}°  "
          f"{'✅ OK' if dfovx < 2.0 else '⚠️  > 2°'}")

    # ── Step 3: 載入 inpainted 訓練圖片 ───────────────────────────────
    print(f"\n🖼️  Step 3: Loading inpainted images ← {args.train_img_dir}")
    raw = [f for f in os.listdir(args.train_img_dir)
           if f.lower().endswith(('.png', '.jpg'))]
    purify_files = sort_cameras(
        [type("_", (), {"image_name": f})() for f in raw]
    )
    purify_files = [o.image_name for o in purify_files]

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

    # ── Step 3.5: 載入 Dead Masks（可選）────────────────────────────
    for cam in train_cameras:
        cam.dead_mask = None
    for cam in test_cameras:
        cam.dead_mask = None

    USE_DEAD_MASK = (args.deadmask_dir is not None)
    if USE_DEAD_MASK:
        print(f"\n💀 Step 3.5: Loading dead masks ← {args.deadmask_dir}")
        dm_files = load_deadmasks_sorted(
            args.deadmask_dir, train_cameras, _W, _H
        )
        cov = [c.dead_mask.mean().item() * 100 for c in train_cameras]
        print(f"  Loaded {len(dm_files)} dead_masks")
        print(f"  Coverage: mean={np.mean(cov):.1f}%  "
              f"min={np.min(cov):.1f}%  max={np.max(cov):.1f}%")
        if np.mean(cov) > 40:
            print("[DEBUG] ⚠️  Dead coverage > 40%，確認 deadmask 是否只標記死角（非整張圖）")
        # Dead mask centroid for first camera
        _dm0 = train_cameras[0].dead_mask
        ys, xs = torch.nonzero(_dm0[0] > 0.5, as_tuple=True)
        if len(ys) > 0:
            print(f"[DEBUG] DeadMask[0] centroid=({int(ys.float().mean())}, "
                  f"{int(xs.float().mean())})  "
                  f"bbox={int(ys.max()-ys.min())}×{int(xs.max()-xs.min())}  "
                  f"nonzero={len(ys)} px")
    else:
        print("\n[INFO] --deadmask_dir not provided → uniform L1（無死角降權）")

    # ── Step 4: Optimizer ─────────────────────────────────────────────
    centers   = torch.stack([c.camera_center for c in train_cameras])
    extent    = torch.norm(centers - centers.mean(0), dim=-1).max().item() * 1.1
    lr_extent = max(extent, 1.0)
    print(f"\n🌍 Scene extent: {extent:.4f}  lr_extent: {lr_extent:.4f}")
    if extent < 0.5:
        print("[DEBUG] ⚠️  extent < 0.5，已啟用 lr_extent 下限保護")

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
    print(f"\n🔥 Training on {len(train_cameras)} views ({TOTAL_ITERS} iters)")
    print(f"   Loss = 0.70×L1_weighted(dead={DEAD_WEIGHT}) + 0.15×SSIM + 0.15×LPIPS_patch")
    print(f"   [SSIM] 使用 ssim_loss() 直接作為 loss（已修正符號 bug）")

    cycle = train_cameras.copy()
    random.shuffle(cycle)
    ci       = 0
    loss_log = []

    for it in tqdm(range(1, TOTAL_ITERS + 1), desc="Training"):
        # ── LR update ──────────────────────────────────────────────────
        for pg in opt.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = lr_fn(it)

        # ── Sample camera ──────────────────────────────────────────────
        cam = cycle[ci % len(cycle)]; ci += 1
        if ci % len(cycle) == 0:
            random.shuffle(cycle)

        # ── Forward ────────────────────────────────────────────────────
        pkg       = render(cam, gaussians, pipe, bg)
        img       = pkg["render"]
        gt        = cam.original_image
        dead_mask = getattr(cam, "dead_mask", None)

        # ── L1 with dead-zone weighting ────────────────────────────────
        ll1_masked, ll1_raw = weighted_l1(img, gt, dead_mask, dead_w=DEAD_WEIGHT)

        # ── SSIM loss（全圖；已回傳 1-SSIM，直接 + 進 loss）────────────
        # ⚠️ 注意：ssim_loss() 已經回傳 (1-SSIM)，這裡不能再做 1.0 - ssim_loss()
        ls = ssim_loss(img, gt)

        # ── Random patch LPIPS（全圖隨機，降低 batch=1 梯度 noise）──────
        lp = random_patch_lpips(lpips_fn, img, gt, _H, _W, patch=PATCH_SIZE)

        # ── Total loss ─────────────────────────────────────────────────
        loss = 0.70 * ll1_masked + 0.15 * ls + 0.15 * lp
        loss.backward()

        # ── Gaussian management ────────────────────────────────────────
        with torch.no_grad():
            # Densification stats（收集至 DENSIFY_UNTIL）
            if it < DENSIFY_UNTIL:
                gaussians.max_radii2D[pkg["visibility_filter"]] = torch.max(
                    gaussians.max_radii2D[pkg["visibility_filter"]],
                    pkg["radii"][pkg["visibility_filter"]])
                gaussians.add_densification_stats(
                    pkg["viewspace_points"], pkg["visibility_filter"])

            opt.step()
            opt.zero_grad(set_to_none=True)

            # SH ramp-up：每 1000 iters 升一級
            if it % 1000 == 0:
                gaussians.active_sh_degree = min(
                    gaussians.active_sh_degree + 1, gaussians.max_sh_degree)

            # Densify + prune（DENSIFY_FROM ~ DENSIFY_UNTIL）
            if DENSIFY_FROM < it < DENSIFY_UNTIL and it % DENSIFY_INTERVAL == 0:
                n_now = gaussians.get_xyz.shape[0]
                if n_now < MAX_GAUSSIANS:
                    gaussians.densify_and_prune(
                        max_grad=MAX_GRAD,
                        min_opacity=MIN_OPACITY_PRUNE,
                        extent=lr_extent,
                        max_screen_size=MAX_SCREEN_SIZE,
                    )
                else:
                    # 超過上限：只做 opacity prune，不再 densify
                    pm = (gaussians.get_opacity < MIN_OPACITY_PRUNE).squeeze()
                    if pm.any():
                        gaussians.prune_points(pm)

            # Reset opacity（兩次，在 densify 窗口內均勻分佈）
            if it in RESET_ITERS:
                gaussians.reset_opacity()
                print(f"\n[DEBUG] reset_opacity at iter {it}  "
                      f"n_gaussians={gaussians.get_xyz.shape[0]:,}")

            # 最終清掃：densify 結束後一次性剔除低 opacity 死點
            if it == CLEANUP_ITER:
                pm = (gaussians.get_opacity < CLEANUP_OPACITY).squeeze()
                if pm.any():
                    n_before = gaussians.get_xyz.shape[0]
                    gaussians.prune_points(pm)
                    n_after  = gaussians.get_xyz.shape[0]
                    print(f"\n[CLEANUP] iter {it}: pruned {n_before-n_after:,} low-opacity Gaussians"
                          f"  opacity<{CLEANUP_OPACITY}  ({n_before:,} → {n_after:,})")

        # ── Debug logging（每 1000 iters）─────────────────────────────
        if it % 1000 == 0:
            n_gs   = gaussians.get_xyz.shape[0]
            xyz_lr = next(pg["lr"] for pg in opt.param_groups if pg["name"] == "xyz")
            entry  = {
                "iter":      it,
                "loss":      loss.item(),
                "l1_raw":    ll1_raw.item(),
                "l1_w":      ll1_masked.item(),
                "ssim":      ls.item(),           # 1-SSIM，訓練正常時應穩定下降至 ~0.1
                "lpips_p":   lp.item(),
                "n_gs":      n_gs,
                "xyz_lr":    xyz_lr,
            }
            loss_log.append(entry)
            print(f"\n[DEBUG] iter={it:5d}  "
                  f"loss={entry['loss']:.4f}  "
                  f"l1_raw={entry['l1_raw']:.4f}  "
                  f"l1_w={entry['l1_w']:.4f}  "
                  f"ssim(1-SSIM)={entry['ssim']:.4f}  "
                  f"lpips_patch={entry['lpips_p']:.4f}  "
                  f"n_gs={n_gs:,}  "
                  f"xyz_lr={xyz_lr:.2e}")

    # ── Post-training diagnostics ──────────────────────────────────────
    n_final = gaussians.get_xyz.shape[0]
    op      = gaussians.get_opacity.squeeze()
    sc      = gaussians.get_scaling
    print(f"\n  Final Gaussians : {n_final:,}")
    print(f"[DEBUG] Opacity stats : mean={op.mean():.3f}  median={op.median():.3f}  "
          f">0.5: {(op>0.5).float().mean()*100:.1f}%  "
          f">0.1: {(op>0.1).float().mean()*100:.1f}%")
    print(f"[DEBUG] Scale  stats  : mean={sc.mean():.4f}  median={sc.median():.4f}  "
          f"max={sc.max():.4f}")

    # 訓練曲線摘要
    print("\n[DEBUG] ── Training summary ──")
    print(f"  {'iter':>6}  {'loss':>7}  {'l1_raw':>7}  "
          f"{'l1_w':>7}  {'ssim':>7}  {'lpips_p':>8}  {'n_gs':>9}  {'xyz_lr':>8}")
    for e in loss_log:
        print(f"  {e['iter']:6d}  {e['loss']:7.4f}  {e['l1_raw']:7.4f}  "
              f"{e['l1_w']:7.4f}  {e['ssim']:7.4f}  "
              f"{e['lpips_p']:8.4f}  {e['n_gs']:9,}  {e['xyz_lr']:8.2e}")

    # ── 健康警告 ───────────────────────────────────────────────────────
    if n_final > 3_000_000:
        print(f"\n[WARN] ⚠️  Final Gaussians {n_final:,} > 3M，"
              f"考慮提高 MAX_GRAD 或 MIN_OPACITY_PRUNE 後重跑")
    ssim_vals = [e["ssim"] for e in loss_log[-5:]]
    if any(v < 0 for v in ssim_vals):
        print("[WARN] ⚠️  末段 ssim(1-SSIM) 出現負值，表示 loss 尚未完全收斂")
    else:
        print(f"\n[HEALTH] ✅  SSIM loss 末段均為正值（mean={np.mean(ssim_vals):.4f}）")

    # ── Step 6: Render from GT / render poses ─────────────────────────
    print(f"\n📸 Rendering from {len(test_cameras)} poses ...")
    with torch.no_grad():
        for cam in tqdm(test_cameras, desc="Rendering"):
            pkg = render(cam, gaussians, pipe, bg)
            save_image(pkg["render"],
                       os.path.join(args.output_dir, cam.image_name))

    _renders = sorted(glob.glob(os.path.join(args.output_dir, "*.png")))
    if _renders:
        _r = cv2.imread(_renders[0])
        if _r is not None:
            ow, oh = _r.shape[1], _r.shape[0]
            gw, gh = test_cameras[0].image_width, test_cameras[0].image_height
            print(f"\n[DEBUG] Render output size : {ow} × {oh}")
            print(f"[DEBUG] Expected (cam said): {gw} × {gh}  "
                  f"{'✅' if ow==gw and oh==gh else '⚠️  MISMATCH'}")

    print(f"\n✅ Done → {args.output_dir}")


# ═══════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = ArgumentParser()

    # 必填
    p.add_argument("--colmap_dir",    required=True,
                   help="VGGT/COLMAP 重建目錄（point cloud + train poses）")
    p.add_argument("--nvs_pose",      required=True,
                   help="Render 用 pose 來源（360° 通常和 colmap_dir 相同）")
    p.add_argument("--train_img_dir", required=True,
                   help="Inpainted 訓練圖片目錄")
    p.add_argument("--output_dir",    required=True,
                   help="Render 輸出目錄")

    # 可選
    p.add_argument("--deadmask_dir", default=None,
                   help="Dead mask 目錄（cv2.inpaint 死角，訓練時降權）")
    p.add_argument("--gt_img_dir",   default=None,
                   help="GT 圖片目錄（僅用於 debug 確認解析度，不參與訓練）")

    # Hyperparameters
    p.add_argument("--total_iters", type=int,   default=20000,
                   help="訓練步數（預設 20000；densify 在 12000 結束）")
    p.add_argument("--dead_weight", type=float, default=0.3,
                   help="Dead mask 區域的 L1 降權倍率（預設 0.3；有效範圍 0.1–0.5）")
    p.add_argument("--patch_size",  type=int,   default=256,
                   help="隨機 patch LPIPS 的邊長（預設 256）")

    train_and_render(p.parse_args())