"""
train_render.py  —  Dual-VGGT Zero Data Leakage 版

  --colmap_dir  : VGGT Run1（60張 inpainted）→ point cloud + train poses
  --nvs_pose    : VGGT Run2（60張 inpainted + 40張 GT，GT 排後面）
                  → 只從 sparse/cameras.bin + images.bin 讀 GT poses
                  → 完全不需要對應圖片，不透過 Scene
  --train_img_dir: 60張 inpainted 圖
  --output_dir  : render 輸出

用法：
  python train_render.py \
      --colmap_dir    purify_scene_only60    \
      --nvs_pose      purify_scene_hybrid_v2 \
      --train_img_dir <60張inpainted圖目錄>   \
      --output_dir    ./renders_40
"""

import os, struct, math, random
import cv2, numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import save_image

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer.render import render
import lpips


# ──────────────────────────────────────────────
# Minimal COLMAP binary reader
# ──────────────────────────────────────────────

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
            cid  = struct.unpack("<I", f.read(4))[0]
            mid  = struct.unpack("<I", f.read(4))[0]
            w    = struct.unpack("<Q", f.read(8))[0]
            h    = struct.unpack("<Q", f.read(8))[0]
            np_  = MODEL_NPARAMS[mid]
            p    = np.array(struct.unpack(f"<{np_}d", f.read(8*np_)))
            cams[cid] = {"model": MODEL_NAMES[mid], "width": w, "height": h, "params": p}
    return cams


def _read_images_bin(path):
    results = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz      = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<I", f.read(4))[0]
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
    n = np.sqrt(qw**2+qx**2+qy**2+qz**2)
    qw,qx,qy,qz = qw/n,qx/n,qy/n,qz/n
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw),1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),1-2*(qx**2+qy**2)],
    ])


# ──────────────────────────────────────────────
# Lightweight camera object for GT poses
# (no image loading, compatible with 3DGS render())
# ──────────────────────────────────────────────

class PoseOnlyCamera:
    def __init__(self, name, R_wc, t_wc, FoVx, FoVy, width, height, device="cuda"):
        self.image_name   = name
        self.FoVx         = float(FoVx)
        self.FoVy         = float(FoVy)
        self.image_width  = int(width)
        self.image_height = int(height)
        self.original_image = None

        # world_view_transform: 3DGS CUDA reads column-major
        # p_cam = R_wc @ p_world + t  →  wvt upper-left = R_wc.T (column-major trick)
        wvt = np.eye(4, dtype=np.float32)
        wvt[:3, :3] = R_wc.T
        wvt[3,  :3] = t_wc
        self.world_view_transform = torch.tensor(wvt, device=device)

        self.projection_matrix = self._proj(FoVx, FoVy, device)
        self.full_proj_transform = (
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
    直接讀 sparse/ 底下的 cameras.bin + images.bin，
    只回傳非 inpainted_ 開頭的 cameras（即 GT views）。
    不需要對應圖片存在。
    """
    # 找 sparse 目錄（支援 sparse/ 或 sparse/0/）
    sparse_dir = os.path.join(colmap_root, "sparse")
    if not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        sparse_dir = os.path.join(sparse_dir, "0")

    cam_meta = _read_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
    img_meta = _read_images_bin(os.path.join(sparse_dir, "images.bin"))

    cameras = []
    for img in img_meta:
        if img["name"].startswith("inpainted_"):
            continue   # 跳過 inpainted，只要 GT
        cm = cam_meta[img["camera_id"]]
        W, H = cm["width"], cm["height"]
        p = cm["params"]
        model = cm["model"]

        # FoV from intrinsics
        if model == "PINHOLE":
            fx, fy = p[0], p[1]
        else:   # SIMPLE_PINHOLE, SIMPLE_RADIAL, etc.
            fx = fy = p[0]
        FoVx = 2 * np.arctan(W / (2 * fx))
        FoVy = 2 * np.arctan(H / (2 * fy))

        cameras.append(PoseOnlyCamera(
            name=img["name"],
            R_wc=img["R"], t_wc=img["t"],
            FoVx=FoVx, FoVy=FoVy,
            width=W, height=H,
            device=device,
        ))

    cameras.sort(key=lambda c: c.image_name)
    return cameras


# ──────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────

def l1_loss(a, b):
    return (a - b).abs().mean()

def ssim_loss(img1, img2, ws=11, C1=1e-4, C2=9e-4):
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0); img2 = img2.unsqueeze(0)
    C = img1.shape[1]
    x = torch.arange(ws, dtype=torch.float32, device=img1.device) - ws // 2
    g = torch.exp(-x**2 / 4.5); g = g / g.sum()
    k = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(C,1,ws,ws)
    pad = ws // 2
    def conv(t): return F.conv2d(t, k, padding=pad, groups=C)
    m1=conv(img1); m2=conv(img2)
    s1=conv(img1*img1)-m1**2; s2=conv(img2*img2)-m2**2; s12=conv(img1*img2)-m1*m2
    num=(2*m1*m2+C1)*(2*s12+C2); den=(m1**2+m2**2+C1)*(s1+s2+C2)
    return 1-(num/den).mean()

def get_lr_func(lr0, lr1, steps):
    def f(i):
        if i<0 or lr0==lr1==0: return 0.
        if i>=steps: return lr1
        t=np.clip(i/steps,0,1)
        return float(np.exp(np.log(lr0)*(1-t)+np.log(lr1)*t))
    return f

class Pipe:
    compute_cov3D_python=False; convert_SHs_python=False; debug=False

def sort_inpainted(cams):
    return sorted(cams,
                  key=lambda c: int(c.image_name.rsplit('.',1)[0].split('_')[-1]))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def train_and_render(args):
    print("🚀 3DGS 訓練與渲染（Dual-VGGT, Zero Data Leakage）")
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = Pipe()
    bg = torch.tensor([0.,0.,0.], dtype=torch.float32, device="cuda")

    # ── Step 1: point cloud + train poses（Run1，60張）────────────────
    print(f"\n☁️  Step 1: Point cloud + train poses ← {args.colmap_dir}")
    gaussians = GaussianModel(sh_degree=3)
    scene60   = Scene(args.colmap_dir, gaussians, shuffle=False)
    train_cameras = sort_inpainted(scene60.getTrainCameras())
    del scene60
    print(f"  Gaussians   : {gaussians.get_xyz.shape[0]}")
    print(f"  Train cams  : {len(train_cameras)}  "
          f"[{train_cameras[0].image_name} … {train_cameras[-1].image_name}]")

    # ── Step 2: GT poses（Run2 COLMAP binary，不需圖片）──────────────
    print(f"\n📐 Step 2: GT poses (direct binary read) ← {args.nvs_pose}/sparse/")
    test_cameras = load_gt_cameras_from_colmap(args.nvs_pose, device="cuda")
    print(f"  GT test cams: {len(test_cameras)}  "
          f"[{test_cameras[0].image_name} … {test_cameras[-1].image_name}]")
    assert len(test_cameras) > 0, \
        "找不到 GT cameras，確認 nvs_pose sparse 中有非 inpainted_ 開頭的 image 記錄"

    # ── Step 3: 替換 train RGB ──────────────────────────────────────
    print(f"\n🖼️  Step 3: Loading inpainted images ← {args.train_img_dir}")
    raw = [f for f in os.listdir(args.train_img_dir)
           if f.lower().endswith(('.png','.jpg'))]
    purify_files = sorted(raw,
                          key=lambda f: int(f.rsplit('.',1)[0].split('_')[-1]))
    assert len(purify_files) == len(train_cameras), (
        f"圖片數量不符：{len(purify_files)} 張圖 vs {len(train_cameras)} 個 cameras")

    for cam, fname in zip(train_cameras, purify_files):
        path = os.path.join(args.train_img_dir, fname)
        img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, (cam.image_width, cam.image_height))
        cam.original_image = (torch.tensor(img, dtype=torch.float32)
                              .permute(2,0,1).cuda() / 255.)
    print(f"  {purify_files[0]} … {purify_files[-1]}")

    # ── Step 4: Optimizer ─────────────────────────────────────────
    centers = torch.stack([c.camera_center for c in train_cameras])
    extent  = torch.norm(centers - centers.mean(0), dim=-1).max().item() * 1.1
    print(f"\n🌍 Scene extent: {extent:.4f}")

    with torch.no_grad():
        mss = math.log(extent * 0.1)
        sc  = gaussians._scaling.detach()
        m   = sc > mss
        if m.any():
            sc[m] = mss
            gaussians._scaling = torch.nn.Parameter(sc.requires_grad_(True))

    opt = torch.optim.Adam([
        {"params":[gaussians._xyz],          "lr":0.00016*extent, "name":"xyz"},
        {"params":[gaussians._features_dc],   "lr":0.0025,         "name":"f_dc"},
        {"params":[gaussians._features_rest], "lr":0.000005,       "name":"f_rest"},
        {"params":[gaussians._opacity],       "lr":0.05,           "name":"opacity"},
        {"params":[gaussians._scaling],       "lr":0.005,          "name":"scaling"},
        {"params":[gaussians._rotation],      "lr":0.001,          "name":"rotation"},
    ], lr=0., eps=1e-15)
    gaussians.optimizer = opt
    lr_fn = get_lr_func(0.00016*extent, 0.0000016*extent, 10000)

    lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()
    for p in lpips_fn.parameters(): p.requires_grad_(False)

    # ── Step 5: 訓練（只用 60 張 inpainted）────────────────────────
    print("\n🔥 Training on 60 inpainted views ...")
    cycle = train_cameras.copy(); random.shuffle(cycle); ci = 0

    for it in tqdm(range(1, 5001), desc="Training"):
        for pg in opt.param_groups:
            if pg["name"] == "xyz": pg["lr"] = lr_fn(it)

        cam = cycle[ci % len(cycle)]; ci += 1
        if ci % len(cycle) == 0: random.shuffle(cycle)

        pkg = render(cam, gaussians, pipe, bg)
        img = pkg["render"]; gt = cam.original_image

        ll1 = l1_loss(img, gt)
        lp  = lpips_fn(img.unsqueeze(0)*2-1, gt.unsqueeze(0)*2-1).mean()
        ls  = ssim_loss(img, gt)
        loss = 0.7*ll1 + 0.1*ls + 0.2*lp
        loss.backward()

        with torch.no_grad():
            if it < 3500 and gaussians.get_xyz.shape[0] < 800000:
                gaussians.max_radii2D[pkg["visibility_filter"]] = torch.max(
                    gaussians.max_radii2D[pkg["visibility_filter"]],
                    pkg["radii"][pkg["visibility_filter"]])
                gaussians.add_densification_stats(
                    pkg["viewspace_points"], pkg["visibility_filter"])

            opt.step(); opt.zero_grad(set_to_none=True)

            if it % 250 == 0:
                gaussians.active_sh_degree = min(
                    gaussians.active_sh_degree+1, gaussians.max_sh_degree)
            if 200 < it < 3500:
                if it % 100 == 0:
                    gaussians.densify_and_prune(
                        max_grad=0.0002, min_opacity=0.005,
                        extent=extent, max_screen_size=20)
                if it % 33 == 0:
                    pm = (gaussians.get_opacity < 0.005).squeeze()
                    if pm.any(): gaussians.prune_points(pm)
                if it % 1000 == 0:
                    gaussians.reset_opacity()

    print(f"  Final Gaussians: {gaussians.get_xyz.shape[0]}  "
          f"opacity mean={gaussians.get_opacity.mean().item():.3f}")

    # ── Step 6: Render from GT poses ───────────────────────────────
    print(f"\n📸 Rendering from {len(test_cameras)} GT poses ...")
    with torch.no_grad():
        for cam in tqdm(test_cameras, desc="Rendering"):
            pkg = render(cam, gaussians, pipe, bg)
            save_image(pkg["render"],
                       os.path.join(args.output_dir, f"{cam.image_name}"))
    print(f"✅ Done → {args.output_dir}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--colmap_dir",    required=True,
                   help="VGGT Run1：60張 inpainted（point cloud + train poses）")
    p.add_argument("--nvs_pose",      required=True,
                   help="VGGT Run2：60+40張（只從 sparse/ 讀 GT camera poses，不需圖片）")
    p.add_argument("--train_img_dir", required=True,
                   help="60張 inpainted 圖目錄")
    p.add_argument("--output_dir",    required=True)
    train_and_render(p.parse_args())