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
#  YAML Config Support  (CLI > YAML > Python defaults)
# ═══════════════════════════════════════════════════════════
def load_yaml_config(config_path):
    if config_path is None:
        return {}
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def apply_config_to_args(args, config, defaults):
    flat = {}
    for section, params in config.items():
        if isinstance(params, dict):
            flat.update(params)
        else:
            flat[section] = params
    for key, default_val in defaults.items():
        if getattr(args, key, None) is not None:
            continue
        if key in flat:
            setattr(args, key, flat[key])
        else:
            setattr(args, key, default_val)
    return args


def dump_resolved_config(args, defaults, output_path):
    import yaml
    used = {k: getattr(args, k) for k in defaults.keys()}
    with open(output_path, 'w') as f:
        yaml.dump({'train': used}, f, default_flow_style=False, sort_keys=False)




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

def weighted_l1_dynamic(img, gt, dead_mask=None, dead_w=0.3, step=0, dw_warmup=2000,
                        dw_alpha=5.0, dw_dead_scale=1.0):
    """
    結合 DeadMask 與「動態幻覺感知權重」的 L1 Loss。
    回傳計算後的最終 weight_map，供後續的 SSIM 與 LPIPS 使用，確保全局梯度一致。

    dw_dead_scale: 控制動態幻覺權重 w_dyn 在 dead 區「內」的作用強度。
        1.0 = 舊行為（w_dyn 在 dead 區完整作用，會把被評估區壓糊）
        0.0 = w_dyn 完全不作用於 dead 區（dead 區純由 dead_weight 監督，最利 masked 指標）
        中間值 = 線性內插。dead 區「外」永遠維持完整 w_dyn（保留抗幻覺）。
    """
    diff = (img - gt).abs()  # (3, H, W)

    # 1. 取得 Base Weight（來自 Dead Mask）
    if dead_mask is not None:
        in_dead = (dead_mask > 0.5).float()  # (1, H, W)  1 = dead 區
        w_base  = torch.where(dead_mask > 0.5, torch.full_like(dead_mask, dead_w), torch.ones_like(dead_mask))
    else:
        in_dead = torch.zeros(1, img.shape[1], img.shape[2], device=img.device)
        w_base  = torch.ones(1, img.shape[1], img.shape[2], device=img.device)

    # 2. 計算 Dynamic Consensus Weighting
    if step >= dw_warmup:
        with torch.no_grad():  # 非常重要：避免權重本身產生反向傳播的梯度崩潰
            # 殘差越高 -> 越可能是幻覺破壞共識 -> w_dyn 越小
            pixel_err = diff.mean(dim=0, keepdim=True).detach()
            w_dyn = torch.exp(-dw_alpha * pixel_err)
            # ★ dead 區「內」按 dw_dead_scale 衰減壓制：scale=0 → 1.0(不壓), scale=1 → w_dyn(舊)
            w_dyn_dead = 1.0 - dw_dead_scale * (1.0 - w_dyn)
            # w_dyn = w_dyn * (1.0 - in_dead) + w_dyn_dead * in_dead
            w_dyn = torch.ones_like(w_dyn) * (1.0 - in_dead) + w_dyn * in_dead
            
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


def random_patch_lpips(lpips_fn, img, gt, weight_map, H, W, patch=256,
                       dead_mask=None, mask_prob=0.0):
    """
    感知 LPIPS。新增 mask-biased 採樣：
      以 mask_prob 機率把 patch 中心對準 dead 區，確保感知監督真正落在「masked 指標
      評估的區域」。否則隨機 patch 幾乎不會蓋到佔比很小的 mask，loss_lpips_w 形同空轉。
    """
    P  = min(patch, H, W)

    forced = (dead_mask is not None and dead_mask.sum() > 0 and random.random() < mask_prob)
    if forced:
        ys, xs = torch.where(dead_mask.squeeze() > 0.5)
        cy = int(ys.float().mean().item()); cx = int(xs.float().mean().item())
        y0 = int(np.clip(cy - P // 2, 0, H - P))
        x0 = int(np.clip(cx - P // 2, 0, W - P))
    else:
        y0 = random.randint(0, H - P)
        x0 = random.randint(0, W - P)

    # 刻意選中的 mask patch 給滿權重（不再被 dead_weight 壓回去，否則兩機制互相抵消）；
    # 隨機 patch 沿用原本共識信任度（保留 dead 區外的抗幻覺降權）。
    if forced:
        w_patch = 1.0
    else:
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

def prune_floaters(gaussians, opacity_thresh, scale_pct):
    """
    機制 2：Floater 剪枝
    ====================
    直接從 VGGT depth 匯出的 COLMAP，在被遮罩區域的點來自低 confidence 的
    隱藏背景深度估計，初始化時容易產生「半透明 + 異常大」的 floater Gaussian。

    這個函式剪掉「低不透明度 AND 大尺度」的 Gaussian（floater 的典型特徵），
    用 AND 條件確保只移除真正可疑的點，不誤傷合法的大型背景 Gaussian。

    Args:
        opacity_thresh: 低於此不透明度才考慮剪枝（e.g. 0.15）
        scale_pct:      尺度超過此百分位才考慮剪枝（e.g. 0.99 = 前 1% 最大）
    Returns:
        被剪掉的 Gaussian 數量
    """
    opacity = gaussians.get_opacity.squeeze()             # (N,)
    scales  = gaussians.get_scaling.max(dim=1).values     # (N,) 取最大軸尺度

    if scales.numel() == 0:
        return 0

    scale_thresh = torch.quantile(scales, scale_pct)
    floater_mask = (opacity < opacity_thresh) & (scales > scale_thresh)

    n_pruned = int(floater_mask.sum())
    if n_pruned > 0:
        gaussians.prune_points(floater_mask)
    return n_pruned


class Pipe:
    compute_cov3D_python = False; convert_SHs_python = False; debug = False


# ═══════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════

def train(args):
    TOTAL_ITERS = args.total_iters
    DEAD_WEIGHT = args.dead_weight
    PATCH_SIZE  = args.patch_size

    # From args (extracted parameters)
    DENSIFY_FROM      = args.densify_from
    DENSIFY_UNTIL     = args.densify_until
    DENSIFY_INTERVAL  = args.densify_interval
    MAX_GAUSSIANS     = args.max_gaussians
    MAX_GRAD          = args.max_grad
    MIN_OPACITY_PRUNE = args.min_opacity_prune
    RESET_ITERS       = args.reset_iters

    # Loss weights (LPIPS-critical knobs)
    L1_W    = args.loss_l1_w
    SSIM_W  = args.loss_ssim_w
    LPIPS_W = args.loss_lpips_w

    # Hardcoded (3DGS standard, rarely tuned)
    MAX_SCREEN_SIZE   = 10
    CLEANUP_OPACITY   = 0.10
    CLEANUP_ITER      = DENSIFY_UNTIL + 100

    # 機制 2: Floater 剪枝參數
    FLOATER_PRUNE_ITERS   = args.floater_prune_iters     # list of iters
    FLOATER_OPACITY_THRESH = args.floater_opacity_thresh
    FLOATER_SCALE_PCT      = args.floater_scale_pct

    print("🚀 3DGS 訓練（Bear 360° - Dynamic Weighting 升級版）")
    print(f"   total_iters={TOTAL_ITERS}  dead_weight={DEAD_WEIGHT}  patch_size={PATCH_SIZE}")
    print(f"   [Dynamic Weighting] Warmup={args.dw_warmup}  Alpha={args.dw_alpha}")
    print(f"   [Loss weights]      L1={L1_W}  SSIM={SSIM_W}  LPIPS={LPIPS_W}")
    print(f"   [Densification]     {DENSIFY_FROM}-{DENSIFY_UNTIL} every {DENSIFY_INTERVAL} iter  max_grad={MAX_GRAD}")
    print(f"   [Floater Prune]     iters={FLOATER_PRUNE_ITERS}  opacity<{FLOATER_OPACITY_THRESH}  scale>p{FLOATER_SCALE_PCT}")

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
            step=it, dw_warmup=args.dw_warmup, dw_alpha=args.dw_alpha,
            dw_dead_scale=args.dw_dead_scale
        )

        # ── SSIM 繼承動態權重；LPIPS 改 mask-biased 採樣（聚焦被評估區）──
        ls = ssim_loss(img, gt, weight_map=w_final)
        lp = random_patch_lpips(lpips_fn, img, gt, weight_map=w_final, H=_H, W=_W,
                                patch=PATCH_SIZE, dead_mask=dead_mask,
                                mask_prob=args.lpips_mask_prob)

        loss = L1_W * ll1_masked + SSIM_W * ls + LPIPS_W * lp
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

            # 機制 2: Floater 剪枝（避開 opacity reset 後的 iter）
            if it in FLOATER_PRUNE_ITERS:
                n_pruned = prune_floaters(
                    gaussians,
                    opacity_thresh=FLOATER_OPACITY_THRESH,
                    scale_pct=FLOATER_SCALE_PCT,
                )
                print(f"\n🧹 [Floater Prune] iter={it}  removed {n_pruned:,} floaters  "
                      f"→ {gaussians.get_xyz.shape[0]:,} remain")

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

    # ── Required paths ──
    p.add_argument("--colmap_dir",       required=True,
                   help="COLMAP 場景目錄（含 sparse/）")
    p.add_argument("--train_img_dir",    required=True,
                   help="Inpainted 訓練圖片目錄")
    p.add_argument("--deadmask_dir",     default=None,
                   help="Dead zone mask 目錄（可選）")
    p.add_argument("--output_gaussian",  required=True,
                   help="訓練完成的 Gaussian 儲存路徑")

    # ── Config & seed ──
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file for training parameters.")
    p.add_argument("--seed", type=int, default=None)

    # ── Training schedule (all default=None for sentinel detection) ──
    p.add_argument("--total_iters",      type=int,   default=None)
    p.add_argument("--dead_weight",      type=float, default=None)
    p.add_argument("--patch_size",       type=int,   default=None)
    p.add_argument("--dw_warmup",        type=int,   default=None)
    p.add_argument("--dw_alpha",         type=float, default=None)
    p.add_argument("--dw_dead_scale",    type=float, default=None,
                   help="w_dyn 在 dead 區內的作用強度 (0=不壓制 dead 區, 1=舊行為)")
    p.add_argument("--lpips_mask_prob",  type=float, default=None,
                   help="LPIPS patch 對準 dead 區的機率 (0=純隨機, 0.5~0.7 建議)")

    # ── Loss weights (LPIPS-critical) ──
    p.add_argument("--loss_l1_w",        type=float, default=None)
    p.add_argument("--loss_ssim_w",      type=float, default=None)
    p.add_argument("--loss_lpips_w",     type=float, default=None)

    # ── Densification & pruning ──
    p.add_argument("--densify_from",       type=int,   default=None)
    p.add_argument("--densify_until",      type=int,   default=None)
    p.add_argument("--densify_interval",   type=int,   default=None)
    p.add_argument("--max_gaussians",      type=int,   default=None)
    p.add_argument("--max_grad",           type=float, default=None)
    p.add_argument("--min_opacity_prune",  type=float, default=None)
    p.add_argument("--reset_iters",        type=str,   default=None,
                   help="Comma-separated iters for opacity reset, e.g. '3000,8000'")

    # ── 機制 2: Floater pruning ──
    p.add_argument("--floater_prune_iters",   type=str,   default=None,
                   help="Comma-separated iters to run floater pruning, e.g. '2500,5500,9000'")
    p.add_argument("--floater_opacity_thresh", type=float, default=None,
                   help="Prune Gaussians with opacity below this (default 0.15)")
    p.add_argument("--floater_scale_pct",      type=float, default=None,
                   help="Prune Gaussians with scale above this percentile (default 0.99)")

    args = p.parse_args()

    # ── Apply YAML config + defaults ──
    TRAIN_DEFAULTS = {
        'seed':              33,
        'total_iters':       20000,
        'dead_weight':       0.3,
        'patch_size':        256,
        'dw_warmup':         1500,
        'dw_alpha':          7.0,
        'dw_dead_scale':     1.0,   # 預設維持舊行為；測試 YAML 會設 0.0
        'lpips_mask_prob':   0.0,   # 預設純隨機；測試 YAML 會設 0.6
        'loss_l1_w':         0.70,
        'loss_ssim_w':       0.15,
        'loss_lpips_w':      0.15,
        'densify_from':      500,
        'densify_until':     12000,
        'densify_interval':  100,
        'max_gaussians':     2_500_000,
        'max_grad':          0.0006,
        'min_opacity_prune': 0.05,
        'reset_iters':       '3000,8000',
        'floater_prune_iters':    '2500,5500,9000',
        'floater_opacity_thresh': 0.15,
        'floater_scale_pct':      0.99,
    }
    config = load_yaml_config(args.config)
    args = apply_config_to_args(args, config, TRAIN_DEFAULTS)

    # Parse reset_iters: support both string "3000,8000" (CLI/YAML) and list (already-parsed)
    if isinstance(args.reset_iters, str):
        args.reset_iters = [int(x.strip()) for x in args.reset_iters.split(',')]

    # Parse floater_prune_iters similarly
    if isinstance(args.floater_prune_iters, str):
        args.floater_prune_iters = [int(x.strip()) for x in args.floater_prune_iters.split(',')]

    # ── Set seed ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Dump resolved config ──
    gauss_dir = os.path.dirname(os.path.abspath(args.output_gaussian))
    os.makedirs(gauss_dir, exist_ok=True)
    dump_resolved_config(args, TRAIN_DEFAULTS,
                         os.path.join(gauss_dir, 'train_resolved.yaml'))

    train(args)