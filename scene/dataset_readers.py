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
# 核心：COLMAP 讀取函數
# ==============================================================================
def readColmapSceneInfo(path, images_folder="images", eval_mode=False):
    """
    利用 pycolmap 直接讀取 VGGT 輸出的 sparse 資料夾
    """
    cameras_ext = "bin"
    sparse_path = os.path.join(path, "sparse")
    if not os.path.exists(sparse_path):
        # 容錯處理：如果使用者直接傳入 sparse 資料夾
        if os.path.basename(path) == "sparse":
            sparse_path = path
            path = os.path.dirname(path)
        else:
            raise FileNotFoundError(f"找不到 COLMAP sparse 資料夾: {sparse_path}")

    print(f"Reading COLMAP reconstruction from {sparse_path}")
    reconstruction = pycolmap.Reconstruction(sparse_path)
    
    cam_infos = []
    
    # 1. 讀取所有相機與影像資訊
    for image_id, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]
        
# 支援 VGGT 輸出的 PINHOLE 與 SIMPLE_PINHOLE (基於 params 長度判斷)
        if len(camera.params) == 4:
            fx, fy, cx, cy = camera.params
        elif len(camera.params) == 3:
            f, cx, cy = camera.params
            fx = fy = f
        else:
            raise ValueError(f"不預期的相機參數長度: {len(camera.params)}，無法解析為 Pinhole 模型。")

        FovY = focal2fov(fy, camera.height)
        FovX = focal2fov(fx, camera.width)

        # 讀取 R, T (COLMAP 預設為 World-to-Camera)
        R = image.cam_from_world.rotation.matrix().transpose()
        T = image.cam_from_world.translation

        image_path = os.path.join(path, images_folder, image.name)
        image_name = os.path.basename(image_path).split(".")[0]
        
        # 延遲讀取圖片 (節省記憶體，交給 Dataset Loader 處理)
        pil_image = Image.open(image_path)
        
        cam_info = CameraInfo(
            uid=image_id, R=R, T=T, FovY=FovY, FovX=FovX, 
            image=pil_image, image_path=image_path, image_name=image_name, 
            width=camera.width, height=camera.height
        )
        cam_infos.append(cam_info)

    # 依照影像名稱排序，確保順序與你原本的 0~59 幀一致
    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    # 區分訓練集與測試集 (在此我們先全部作為訓練集)
    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 2. 讀取 3D 點雲 (VGGT 預測的鷹架)
    xyz = []
    rgb = []
    for point3D_id, point3D in reconstruction.points3D.items():
        xyz.append(point3D.xyz)
        rgb.append(point3D.color / 255.0) # COLMAP 顏色是 0-255，轉為 0-1

    xyz = np.array(xyz)
    rgb = np.array(rgb)
    normals = np.zeros_like(xyz) # 初始點雲通常沒有法向量
    
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