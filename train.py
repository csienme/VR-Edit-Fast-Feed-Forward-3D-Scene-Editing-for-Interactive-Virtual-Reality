"""
train_.py  —  Bear 360° / 單 COLMAP 場景版（v6 - Uncertainty-Aware Dynamic Weighting）

[訓練專用版]
完成訓練後將 Gaussian 儲存至指定 .ply 路徑，供 render.py 獨立載入使用。

用法：
    python train_.py \
        --colmap_dir    "${PURIFY_DIR}" \
        --train_img_dir "${PURIFY_DIR}/images" \
        --deadmask_dir  "${DEADMASK_DIR}" \
        --output_gaussian "${OUTPUT_DIR}/gaussians.ply" \
        --total_iters   20000 \
        --dead_weight   0.3 \
        --patch_size    256
"""

import os, struct, math, random
import cv2, numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer.render import render
import lpips as lpips_module


# ═══════════════════════════════════════════════════════════
#  COLMAP Binary Reader & Camera Classes（不變）
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
            struct.unpack("<I", f.read(4))[0]
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
            results.append({"name": name.decode(), "camera_id": cid, "R": R, "t": t})
    return results

def _quat_to_rot(qw, qx, qy, qz):
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw),1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),1-2*(qx**2+qy**2)],
    ])

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
        self.camera_center = torch.tensor(-R_wc.T @ t_wc, dtype=torch.float32, device=device)

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

def sort_cameras(cams):
    def _key(c):
        stem = c.image_name.rsplit('.', 1)[0]
        try:
            return (0, int(stem.split('_')[-1]))
        except ValueError:
            return (1, stem)
    return sorted(cams, key=_key)

def load_deadmasks_sorted(deadmask_dir, cameras, target_w, target_h):
    files = sorted(f for f in os.listdir(deadmask_dir) if f.lower().endswith(('.png', '.jpg')))
    assert len(files) == len(cameras), "Deadmask 數量與 camera 數量不符"
    for cam, fname in zip(cameras, files):
        path = os.path.join(deadmask_dir, fname)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        cam.dead_mask = torch.tensor((m > 127).astype(np.float32), device="cuda").unsqueeze(0)
    return files


# ═══════════════════════════════════════════════════════════
#  💡 核心貢獻：Uncertainty-Aware Loss Functions
# ═══════════════════════════════════════════════════════════

def weighted_l1_dynamic(img, gt, dead_mask=None, dead_w=0.3, step=0, dw_warmup=2000, dw_alpha=5.0):
    """
    結合 DeadMask 與「動態幻覺感知權重」的 L1 Loss。
    回傳計算後的最終 weight_map，供後續的 SSIM 與 LPIPS 使用，確保全局梯度一致。
    """
    diff = (img - gt).abs()  # (3, H, W)

    # 1. 取得 Base Weight（來自 Dead Mask）
    if dead_mask is not None:
        w_base = torch.where(dead_mask > 0.5, torch.full_like(dead_mask, dead_w), torch.ones_like(dead_mask))
    else:
        w_base = torch.ones(1, img.shape[1], img.shape[2], device=img.device)

    # 2. 計算 Dynamic Consensus Weighting
    if step >= dw_warmup:
        with torch.no_grad():  # 非常重要：避免權重本身產生反向傳播的梯度崩潰
            # 殘差越高 -> 越可能是幻覺破壞共識 -> w_dyn 越小
            pixel_err = diff.mean(dim=0, keepdim=True).detach()
            w_dyn = torch.exp(-dw_alpha * pixel_err)
    else:
        w_dyn = torch.ones_like(w_base)

    # 最終權重圖
    w_final = w_base * w_dyn

    l1_weighted = (diff * w_final).sum() / (w_final.sum() * diff.shape[0] + 1e-8)
    l1_raw      = diff.mean()

    return l1_weighted, l1_raw, w_final, w_dyn.mean()


def ssim_loss(img1, img2, weight_map=None, ws=11, C1=1e-4, C2=9e-4):
    """支援像素級權重（Pixel-wise Weight Map）的 SSIM"""
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

    ssim_map = num / den  # shape: (1, 3, H, W)

    if weight_map is not None:
        # 將 weight_map (1, H, W) 擴展至 (1, 1, H, W) 與 ssim_map 對應相乘
        ssim_val = (ssim_map * weight_map.unsqueeze(1)).sum() / (weight_map.sum() * C + 1e-8)
        return (1 - ssim_val).clamp(min=0.0)

    return (1 - ssim_map.mean()).clamp(min=0.0)


def random_patch_lpips(lpips_fn, img, gt, weight_map, H, W, patch=256):
    """感知 LPIPS 支援動態降權：若採樣到幻覺災區，降低該 Patch 對總梯度的影響"""
    P  = min(patch, H, W)
    y0 = random.randint(0, H - P)
    x0 = random.randint(0, W - P)

    # 計算該 Patch 範圍內的平均權重
    w_patch = weight_map[:, y0:y0+P, x0:x0+P].mean().item() if weight_map is not None else 1.0

    ip = img[:, y0:y0+P, x0:x0+P].unsqueeze(0)
    gp = gt [:, y0:y0+P, x0:x0+P].unsqueeze(0)

    # 乘上該 Patch 的共識信任度
    return lpips_fn(ip * 2 - 1, gp * 2 - 1).mean() * w_patch


def get_lr_func(lr0, lr1, steps):
    def f(i):
        if i < 0 or lr0 == lr1 == 0: return 0.
        if i >= steps: return lr1
        t = np.clip(i / steps, 0, 1)
        return float(np.exp(np.log(lr0)*(1-t) + np.log(lr1)*t))
    return f

class Pipe:
    compute_cov3D_python = False; convert_SHs_python = False; debug = False


# ═══════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════

def train(args):
    TOTAL_ITERS = args.total_iters
    DEAD_WEIGHT = args.dead_weight
    PATCH_SIZE  = args.patch_size

    DENSIFY_FROM      = 500
    DENSIFY_UNTIL     = 12000
    DENSIFY_INTERVAL  = 100
    MAX_GAUSSIANS     = 2_500_000
    MAX_GRAD          = 0.0006
    MIN_OPACITY_PRUNE = 0.05
    MAX_SCREEN_SIZE   = 10
    CLEANUP_ITER      = DENSIFY_UNTIL + 100
    CLEANUP_OPACITY   = 0.10
    RESET_ITERS       = [3000, 8000]

    print("🚀 3DGS 訓練（Bear 360° - Dynamic Weighting 升級版）")
    print(f"   total_iters={TOTAL_ITERS}  dead_weight={DEAD_WEIGHT}  patch_size={PATCH_SIZE}")
    print(f"   [Dynamic Weighting] 啟用! Warmup: {args.dw_warmup} steps, Alpha: {args.dw_alpha}")

    pipe = Pipe()
    bg   = torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda")

    # ── Step 1~3: Load Data ──
    gaussians     = GaussianModel(sh_degree=3)
    scene_obj     = Scene(args.colmap_dir, gaussians, shuffle=False)
    train_cameras = sort_cameras(scene_obj.getTrainCameras())
    del scene_obj

    raw = [f for f in os.listdir(args.train_img_dir) if f.lower().endswith(('.png', '.jpg'))]
    purify_files = sort_cameras([type("_", (), {"image_name": f})() for f in raw])
    purify_files = [o.image_name for o in purify_files]

    for cam, fname in zip(train_cameras, purify_files):
        path = os.path.join(args.train_img_dir, fname)
        img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, (cam.image_width, cam.image_height))
        cam.original_image = (torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).cuda() / 255.)

    _H, _W = train_cameras[0].original_image.shape[1:]

    for cam in train_cameras:
        cam.dead_mask = None

    if args.deadmask_dir is not None:
        load_deadmasks_sorted(args.deadmask_dir, train_cameras, _W, _H)

    # ── Step 4: Optimizer ──
    centers   = torch.stack([c.camera_center for c in train_cameras])
    extent    = torch.norm(centers - centers.mean(0), dim=-1).max().item() * 1.1
    lr_extent = max(extent, 1.0)

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
    for _p in lpips_fn.parameters(): _p.requires_grad_(False)

    # ── Step 5: Training Loop ──
    cycle = train_cameras.copy()
    random.shuffle(cycle)
    ci       = 0
    loss_log = []

    for it in tqdm(range(1, TOTAL_ITERS + 1), desc="Training"):
        for pg in opt.param_groups:
            if pg["name"] == "xyz": pg["lr"] = lr_fn(it)

        cam = cycle[ci % len(cycle)]; ci += 1
        if ci % len(cycle) == 0: random.shuffle(cycle)

        pkg       = render(cam, gaussians, pipe, bg)
        img       = pkg["render"]
        gt        = cam.original_image
        dead_mask = getattr(cam, "dead_mask", None)

        # ── 核心計算：傳入當前 step 以啟動動態幻覺感知權重 ──
        ll1_masked, ll1_raw, w_final, mean_dyn_w = weighted_l1_dynamic(
            img, gt, dead_mask, dead_w=DEAD_WEIGHT,
            step=it, dw_warmup=args.dw_warmup, dw_alpha=args.dw_alpha
        )

        # ── SSIM 與 LPIPS 全面繼承動態權重，防止幻覺細節污染結構 ──
        ls = ssim_loss(img, gt, weight_map=w_final)
        lp = random_patch_lpips(lpips_fn, img, gt, weight_map=w_final, H=_H, W=_W, patch=PATCH_SIZE)

        loss = 0.70 * ll1_masked + 0.15 * ls + 0.15 * lp
        loss.backward()

        # Gaussian management（不變）
        with torch.no_grad():
            if it < DENSIFY_UNTIL:
                gaussians.max_radii2D[pkg["visibility_filter"]] = torch.max(
                    gaussians.max_radii2D[pkg["visibility_filter"]], pkg["radii"][pkg["visibility_filter"]])
                gaussians.add_densification_stats(pkg["viewspace_points"], pkg["visibility_filter"])

            opt.step()
            opt.zero_grad(set_to_none=True)

            if it % 1000 == 0:
                gaussians.active_sh_degree = min(gaussians.active_sh_degree + 1, gaussians.max_sh_degree)

            if DENSIFY_FROM < it < DENSIFY_UNTIL and it % DENSIFY_INTERVAL == 0:
                n_now = gaussians.get_xyz.shape[0]
                if n_now < MAX_GAUSSIANS:
                    gaussians.densify_and_prune(max_grad=MAX_GRAD, min_opacity=MIN_OPACITY_PRUNE, extent=lr_extent, max_screen_size=MAX_SCREEN_SIZE)
                else:
                    pm = (gaussians.get_opacity < MIN_OPACITY_PRUNE).squeeze()
                    if pm.any(): gaussians.prune_points(pm)

            if it in RESET_ITERS: gaussians.reset_opacity()

            if it == CLEANUP_ITER:
                pm = (gaussians.get_opacity < CLEANUP_OPACITY).squeeze()
                if pm.any(): gaussians.prune_points(pm)

        if it % 1000 == 0:
            n_gs   = gaussians.get_xyz.shape[0]
            entry  = {
                "iter":  it,
                "loss":  loss.item(),
                "l1_w":  ll1_masked.item(),
                "ssim":  ls.item(),
                "dyn_w": mean_dyn_w.item(),
                "n_gs":  n_gs,
            }
            loss_log.append(entry)
            print(f"\n[DEBUG] iter={it:5d}  loss={entry['loss']:.4f}  l1_w={entry['l1_w']:.4f}  "
                  f"ssim(1-SSIM)={entry['ssim']:.4f}  dyn_w_mean={entry['dyn_w']:.3f}  n_gs={n_gs:,}")

    # ── Step 6: Save Gaussians（用 torch.save state dict）──
    gauss_dir = os.path.dirname(os.path.abspath(args.output_gaussian))
    os.makedirs(gauss_dir, exist_ok=True)

    state = {
        "active_sh_degree":  gaussians.active_sh_degree,
        "_xyz":              gaussians._xyz.detach().cpu(),
        "_features_dc":      gaussians._features_dc.detach().cpu(),
        "_features_rest":    gaussians._features_rest.detach().cpu(),
        "_scaling":          gaussians._scaling.detach().cpu(),
        "_rotation":         gaussians._rotation.detach().cpu(),
        "_opacity":          gaussians._opacity.detach().cpu(),
    }
    torch.save(state, args.output_gaussian)

    n_final = gaussians.get_xyz.shape[0]
    print(f"\n✅ 訓練完成！共 {n_final:,} 個 Gaussians")
    print(f"💾 Gaussian 已儲存 → {args.output_gaussian}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--colmap_dir",       required=True,
                   help="COLMAP 場景目錄（含 sparse/）")
    p.add_argument("--train_img_dir",    required=True,
                   help="Inpainted 訓練圖片目錄")
    p.add_argument("--deadmask_dir",     default=None,
                   help="Dead zone mask 目錄（可選）")
    p.add_argument("--output_gaussian",  required=True,
                   help="訓練完成的 Gaussian 儲存路徑（.ply），例如 output/gaussians.ply")
    p.add_argument("--total_iters",      type=int,   default=20000)
    p.add_argument("--dead_weight",      type=float, default=0.3)
    p.add_argument("--patch_size",       type=int,   default=256)
    p.add_argument("--dw_warmup",        type=int,   default=1500,
                   help="開始啟動不確定性感知的 Iteration")
    p.add_argument("--dw_alpha",         type=float, default=7.0,
                   help="懲罰係數，越大對幻覺殘差越嚴格")
    train(p.parse_args())