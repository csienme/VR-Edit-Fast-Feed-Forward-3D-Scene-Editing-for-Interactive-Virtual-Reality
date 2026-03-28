import os
import math
import numpy as np
from PIL import Image
from typing import NamedTuple
import pycolmap

# ==============================================================================
# 定義資料結構
# ==============================================================================
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.float32
    FovX: np.float32
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

# ==============================================================================
# 輔助數學函數
# ==============================================================================
def focal2fov(focal, pixels):
    """將焦距轉換為視角 (Field of View)"""
    return 2 * math.atan(pixels / (2 * focal))

def getNerfppNorm(cam_info):
    """計算場景的中心點與半徑 (用於 3DGS 的空間正規化)"""
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = np.eye(4)
        W2C[:3, :3] = cam.R.T
        W2C[:3, 3] = cam.T
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

# ==============================================================================
# 核心：COLMAP 讀取函數 (百毒不侵版)
# ==============================================================================
def readColmapSceneInfo(path, images_folder="images", eval_mode=False):
    """
    穩健且高度容錯的 COLMAP 讀取器
    """
    # 🛡️ 升級 1：智慧路徑穿透 (自動尋找 0/ 或根目錄)
    sparse_root = os.path.join(path, "sparse")
    if os.path.exists(os.path.join(sparse_root, "0")):
        sparse_path = os.path.join(sparse_root, "0")
    elif os.path.exists(sparse_root):
        sparse_path = sparse_root
    else:
        sparse_path = path

    print(f"🌍 正在讀取 COLMAP 幾何: {sparse_path}")
    reconstruction = pycolmap.Reconstruction(sparse_path)
    
    cam_infos = []
    
    # 讀取所有相機與影像資訊
    for image_id, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]
        
        # 🛡️ 升級 2：安全的相機參數解析 (不再依賴字串比對，直接看長度)
        if len(camera.params) >= 4:
            fx, fy, cx, cy = camera.params[:4]
        else:
            f, cx, cy = camera.params[:3]
            fx = fy = f

        FovY = focal2fov(fy, camera.height)
        FovX = focal2fov(fx, camera.width)

        # 🛡️ 相容新舊版 pycolmap 的 R, T 讀取
        if hasattr(image, 'cam_from_world'):
            R = image.cam_from_world.rotation.matrix().transpose()
            T = image.cam_from_world.translation
        else:
            R = image.qvec2rotmat()
            T = image.tvec

        # 🛡️ 升級 3：強大的副檔名容錯機制 (Extension Tolerance)
        base_image_name = os.path.basename(image.name)
        image_path = os.path.join(path, images_folder, base_image_name)
        
        if not os.path.exists(image_path):
            name_no_ext = os.path.splitext(base_image_name)[0]
            found = False
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                fallback_path = os.path.join(path, images_folder, name_no_ext + ext)
                if os.path.exists(fallback_path):
                    image_path = fallback_path
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"❌ 找不到圖片！無論是 {base_image_name} 還是 .png/.jpg 變體都不存在於: {os.path.join(path, images_folder)}")

        image_name = os.path.basename(image_path).split(".")[0]
        pil_image = Image.open(image_path)
        
        cam_info = CameraInfo(
            uid=image_id, R=R, T=T, FovY=FovY, FovX=FovX, 
            image=pil_image, image_path=image_path, image_name=image_name, 
            width=camera.width, height=camera.height
        )
        cam_infos.append(cam_info)

    # 依照影像名稱排序，確保時間序列一致
    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 🛡️ 升級 4：讀取點雲與空點雲防護罩
    xyz, rgb = [], []
    for point3D_id, point3D in reconstruction.points3D.items():
        xyz.append(point3D.xyz)
        rgb.append(point3D.color / 255.0)

    if len(xyz) == 0:
        print("⚠️ 警告：COLMAP 註冊表中沒有點雲 (points3D 缺失)！正在自動生成隨機點雲以維持高斯球初始化...")
        radius = nerf_normalization["radius"]
        xyz = (np.random.rand(10000, 3) - 0.5) * (radius * 1.5)
        rgb = np.random.rand(10000, 3)
    else:
        xyz = np.array(xyz)
        rgb = np.array(rgb)

    normals = np.zeros_like(xyz)
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)
    ply_path = os.path.join(sparse_path, "points.ply")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    return scene_info