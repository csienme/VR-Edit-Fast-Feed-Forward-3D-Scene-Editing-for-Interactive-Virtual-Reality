import os
import random
import torch
import numpy as np
from scene.dataset_readers import readColmapSceneInfo
from scene.gaussian_model import GaussianModel
# 💥 匯入我們寫好的真正 GPU Camera 類別
from scene.cameras import Camera

class Scene:
    def __init__(self, colmap_path, gaussians: GaussianModel, shuffle=True):
        """
        建構 3DGS 訓練場景
        """
        self.model_path = colmap_path
        self.gaussians = gaussians

        print(f"🌍 正在讀取 COLMAP 場景: {self.model_path}")
        self.scene_info = readColmapSceneInfo(self.model_path)

        if self.gaussians.get_xyz.shape[0] == 0:
            print("🚀 啟動高斯模型初始化程序...")
            self.gaussians.create_from_pcd(self.scene_info.point_cloud)

        # 💥 關鍵修復：將 CPU 上的 CameraInfo 轉換為帶有 GPU Tensor 的 Camera 物件
        print("📸 將影像與相機矩陣載入 GPU...")
        self.train_cameras = self._cam_info_to_camera(self.scene_info.train_cameras)
        self.test_cameras = self._cam_info_to_camera(self.scene_info.test_cameras)

        # 如果需要訓練，打亂相機順序有助於 SGD 收斂
        if shuffle:
            random.shuffle(self.train_cameras)

    def _cam_info_to_camera(self, cam_infos):
        """
        負責將 dataset_readers 讀進來的基本資料，實體化為 3DGS 的 Camera 模組
        """
        camera_list = []
        for id, c in enumerate(cam_infos):
            # 將 PIL 圖片轉換為 0.0~1.0 的 PyTorch Tensor (C, H, W)
            image_tensor = torch.from_numpy(np.array(c.image)).permute(2, 0, 1).float() / 255.0
            
            # 建立真正的 GPU 相機 (自動計算 world_view_transform 與 projection_matrix)
            cam = Camera(
                colmap_id=c.uid, 
                R=c.R, 
                T=c.T, 
                FoVx=c.FovX,  # 將 FovX 映射給 FoVx
                FoVy=c.FovY, 
                image=image_tensor, 
                image_name=c.image_name, 
                uid=id
            )
            camera_list.append(cam)
        return camera_list

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras