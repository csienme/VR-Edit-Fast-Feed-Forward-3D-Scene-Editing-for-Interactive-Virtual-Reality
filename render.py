"""
render.py  —  Bear 360° / Novel View Rendering（從已訓練 Gaussian 載入）

[渲染專用版]
從 train_.py 輸出的 .ply 載入訓練完畢的 Gaussian，
對指定 COLMAP pose 目錄中的所有視角進行渲染。

用法：
    python render.py \
        --nvs_pose          "${PURIFY_HYBRID_DIR}" \
        --gaussian_path     "${OUTPUT_DIR}/gaussians.ply" \
        --render_output_dir "${RENDER_DIR}"
"""

import os, struct, math
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import save_image

from scene.gaussian_model import GaussianModel
from gaussian_renderer.render import render
from torch import nn


# ═══════════════════════════════════════════════════════════
#  COLMAP Binary Reader & Camera Classes（與 train_.py 相同）
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

def load_gt_cameras_from_colmap(colmap_root, device="cuda"):
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

    cameras = [_build(img) for img in img_meta if not img["name"].startswith("inpainted_")]
    if len(cameras) == 0:
        cameras = [_build(img) for img in img_meta]

    cameras.sort(key=lambda c: c.image_name)
    return cameras

class Pipe:
    compute_cov3D_python = False; convert_SHs_python = False; debug = False


# ═══════════════════════════════════════════════════════════
#  Rendering
# ═══════════════════════════════════════════════════════════

def do_render(args):
    os.makedirs(args.render_output_dir, exist_ok=True)
    pipe = Pipe()
    bg   = torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda")

    # ── 載入已訓練的 Gaussians ──
    print(f"📦 載入 Gaussian: {args.gaussian_path}")
    gaussians = GaussianModel(sh_degree=3)

    state = torch.load(args.gaussian_path, map_location="cpu")
    gaussians.active_sh_degree = state["active_sh_degree"]
    gaussians._xyz           = nn.Parameter(state["_xyz"].cuda())
    gaussians._features_dc   = nn.Parameter(state["_features_dc"].cuda())
    gaussians._features_rest = nn.Parameter(state["_features_rest"].cuda())
    gaussians._scaling       = nn.Parameter(state["_scaling"].cuda())
    gaussians._rotation      = nn.Parameter(state["_rotation"].cuda())
    gaussians._opacity       = nn.Parameter(state["_opacity"].cuda())

    print(f"   → {gaussians.get_xyz.shape[0]:,} 個 Gaussians，active_sh_degree={gaussians.active_sh_degree}")

    # ── 載入 NVS test cameras（與原版完全相同）──
    test_cameras = load_gt_cameras_from_colmap(args.nvs_pose, device="cuda")
    print(f"\n📸 Rendering {len(test_cameras)} views → {args.render_output_dir}")

    with torch.no_grad():
        for cam in tqdm(test_cameras, desc="Rendering"):
            pkg = render(cam, gaussians, pipe, bg)
            save_image(pkg["render"], os.path.join(args.render_output_dir, cam.image_name))

    print(f"\n✅ Done → {args.render_output_dir}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--nvs_pose",          required=True,
                   help="包含 NVS test pose 的 COLMAP 目錄（含 sparse/）")
    p.add_argument("--gaussian_path",     required=True,
                   help="train_.py 輸出的 .ply 路徑，例如 output/gaussians.ply")
    p.add_argument("--render_output_dir", required=True,
                   help="渲染圖片輸出目錄")
    do_render(p.parse_args())