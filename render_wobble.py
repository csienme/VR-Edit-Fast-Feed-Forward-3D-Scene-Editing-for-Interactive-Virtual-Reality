import os
import struct
import math
import numpy as np
import torch
import imageio
from tqdm import tqdm
from argparse import ArgumentParser
from torch import nn
import mediapy as media
# 依賴你環境中的模組
from scene.gaussian_model import GaussianModel
from gaussian_renderer.render import render

# ═══════════════════════════════════════════════════════════
#  COLMAP Binary Reader & Camera Classes (擷取自你的 render.py)
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

class Pipe:
    compute_cov3D_python = False
    convert_SHs_python = False
    debug = False

# ═══════════════════════════════════════════════════════════
#  Wobble Trajectory Math
# ═══════════════════════════════════════════════════════════

def get_reference_camera_params(colmap_root, ref_image_name):
    """從 COLMAP 解析並提取單一基準相機的內外參"""
    sparse_dir = os.path.join(colmap_root, "sparse")
    if not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        sparse_dir = os.path.join(sparse_dir, "0")

    cam_meta = _read_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
    img_meta = _read_images_bin(os.path.join(sparse_dir, "images.bin"))

    ref_img = None
    for img in img_meta:
        if ref_image_name in img["name"]:
            ref_img = img
            break
            
    if ref_img is None:
        print(f"⚠️ 找不到 {ref_image_name}，將預設使用第一張照片做為基準。")
        ref_img = img_meta[0]

    cm = cam_meta[ref_img["camera_id"]]
    W, H = cm["width"], cm["height"]
    p = cm["params"]
    fx = p[0] if cm["model"] == "PINHOLE" else p[0]
    fy = p[1] if cm["model"] == "PINHOLE" else p[0]
    
    FoVx = 2 * np.arctan(W / (2 * fx))
    FoVy = 2 * np.arctan(H / (2 * fy))

    return ref_img["R"], ref_img["t"], FoVx, FoVy, W, H

def generate_wobble_cameras(R_wc, t_cw, FoVx, FoVy, W, H, args):
    """
    產生環繞晃動的虛擬相機序列 (基於 LookAt 幾何約束，並支援 Dolly 推進)
    """
    cameras = []
    
    # 提取初始相機在世界座標的中心點 C_w_orig 與姿態向量
    C_w_orig = -R_wc.T @ t_cw
    
    Right = R_wc[0, :]
    Down = R_wc[1, :]
    Forward = R_wc[2, :]

    # [博士級運鏡升級]：將基準相機沿著視線方向 (Forward) 往前推進
    # 如果 zoom 是正數，相機就會變近；如果是負數，就會退後
    C_w = C_w_orig + args.zoom * Forward

    # 設定注視點 (Focal Point)，我們讓它維持在原始相機往前 focal_dist 的地方
    F_w = C_w_orig + args.focal_dist * Forward

    for i in range(args.frames):
        t = i / args.frames
        angle = 2 * np.pi * t
        
        # 1. 在相機平面的 X-Y 軸上進行平滑圓周微擾 (Wobble)
        # 注意：我們現在是繞著推進後的 C_w 在晃動
        offset = args.radius * (np.cos(angle) * Right - np.sin(angle) * Down)
        C_w_new = C_w + offset
        
        # 2. 重新計算 Forward (死盯 Focal Point)
        Forward_new = F_w - C_w_new
        Forward_new /= np.linalg.norm(Forward_new)
        
        # 3. 外積重構 Right 與 Down
        Right_new = np.cross(Down, Forward_new)
        Right_new /= np.linalg.norm(Right_new)
        Down_new = np.cross(Forward_new, Right_new)
        
        # 4. 組合新的 R 與 t (World-to-Camera)
        R_new = np.stack([Right_new, Down_new, Forward_new], axis=0)
        t_new = -R_new @ C_w_new
        
        # 實例化
        cam = PoseOnlyCamera(
            name=f"wobble_{i}", 
            R_wc=R_new, t_wc=t_new, 
            FoVx=FoVx, FoVy=FoVy, 
            width=W, height=H, device="cuda"
        )
        cameras.append(cam)
        
    return cameras

# ═══════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════

def main(args):
    # 1. 載入模型權重 (完全依照你的 render.py 寫法)
    print(f"📦 [1/4] 載入 Gaussian 權重: {args.gaussian_path}")
    gaussians = GaussianModel(sh_degree=3)
    
    state = torch.load(args.gaussian_path, map_location="cpu")
    gaussians.active_sh_degree = state.get("active_sh_degree", 3)
    gaussians._xyz           = nn.Parameter(state["_xyz"].cuda())
    gaussians._features_dc   = nn.Parameter(state["_features_dc"].cuda())
    gaussians._features_rest = nn.Parameter(state["_features_rest"].cuda())
    gaussians._scaling       = nn.Parameter(state["_scaling"].cuda())
    gaussians._rotation      = nn.Parameter(state["_rotation"].cuda())
    gaussians._opacity       = nn.Parameter(state["_opacity"].cuda())

    print(f"   → 成功載入 {gaussians.get_xyz.shape[0]:,} 個 Gaussians")

    # 2. 提取基準相機
    print(f"📷 [2/4] 從 COLMAP 解析基準視角: {args.ref_image}")
    R, t, FoVx, FoVy, W, H = get_reference_camera_params(args.colmap_dir, args.ref_image)

    # 3. 軌跡規劃
    print(f"🌀 [3/4] 規劃 Wobble 軌跡 (半徑: {args.radius}, 焦距深度: {args.focal_dist})...")
    wobble_cameras = generate_wobble_cameras(R, t, FoVx, FoVy, W, H, args)

    # 4. 渲染與影像編碼
    print(f"🎬 [4/4] 開始渲染 {args.frames} 幀影像...")
    pipe = Pipe()
    bg = torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda")
    frames = []

    with torch.no_grad():
        for cam in tqdm(wobble_cameras, desc="Rendering Frames"):
            # 呼叫 3DGS 渲染核心
            pkg = render(cam, gaussians, pipe, bg)
            img_tensor = pkg["render"].clamp(0.0, 1.0)
            
            # Tensor 轉 Numpy (H, W, C)
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(img_np)

# 記得在檔案最上方（或其他 import 的地方）加入這行：
    # import mediapy as media
    
    # === 儲存影片核心邏輯 (採用業界標準的 mediapy 寫法) ===
    out_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"💾 正在使用 mediapy 編碼高相容性 H.264 影片，請稍候...")
    
    # 確保影像長寬是偶數 (H.264 標準的硬性規定)
    H, W = frames[0].shape[:2]
    if H % 2 != 0 or W % 2 != 0:
        frames = [f[:H - (H % 2), :W - (W % 2)] for f in frames]
        H, W = frames[0].shape[:2]

    # 完全照抄你貼的強健參數設定 (crf=18 是視覺無損的高畫質標準)
    video_kwargs = {
        'shape': (H, W),
        'codec': 'h264',
        'fps': args.fps,
        'crf': 18,
    }

    try:
        # 使用 mediapy 的 Context Manager 寫入影片
        with media.VideoWriter(out_path, **video_kwargs) as writer:
            for frame in frames:
                # 確保資料型態為 uint8，否則編碼器會報錯
                frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
                writer.add_image(frame_uint8)
                
        print(f"\n✅ 完美！動態展示影片已成功儲存至: {out_path} (保證可用 VSCode 與 Chrome 開啟)")
    except Exception as e:
        print(f"\n❌ mediapy 寫入失敗，請確認錯誤訊息: {e}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--colmap_dir", type=str, default="eval_results_custom/my_scene/colmap", help="COLMAP 目錄路徑 (包含 sparse 隱藏資料夾)")
    parser.add_argument("--gaussian_path", type=str, default="eval_results_custom/my_scene/renders/gaussians.pth", help="訓練好的 .pth 權重路徑")
    parser.add_argument("--ref_image", type=str, default="inpainted_0.png", help="作為基準的圖片名稱 (需存在於 images.bin)")
    parser.add_argument("--output_dir", type=str, default="eval_results_custom/my_scene", help="輸出影片的存放目錄")
    parser.add_argument("--output_name", type=str, default="demo_wobble.mp4", help="輸出的影片檔名")
    
    # Wobble 軌跡參數
    parser.add_argument("--radius", type=float, default=0.08, help="相機晃動的半徑幅度 (視場景 Scale 而定)")
    parser.add_argument("--focal_dist", type=float, default=3.0, help="相機到目標注視點的深度距離")
    parser.add_argument("--frames", type=int, default=120, help="影片總幀數")
    parser.add_argument("--fps", type=int, default=30, help="影片幀率 (Frames per second)")
    
    parser.add_argument("--zoom", type=float, default=0.0, help="相機往前推進的距離 (正值=拉近, 負值=拉遠)")
    args = parser.parse_args()
    main(args)