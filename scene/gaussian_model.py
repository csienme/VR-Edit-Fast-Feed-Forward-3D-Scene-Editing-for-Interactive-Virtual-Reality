import torch
import numpy as np
from torch import nn
from sklearn.neighbors import NearestNeighbors

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def build_scaling_rotation(s, r):
    # 建構縮放矩陣
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    # 建構旋轉矩陣 (從四元數 Quaternion)
    r = r / torch.norm(r, dim=1, keepdim=True)
    R[:, 0, 0] = 1 - 2 * (r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    R[:, 0, 1] = 2 * (r[:, 1] * r[:, 2] - r[:, 0] * r[:, 3])
    R[:, 0, 2] = 2 * (r[:, 1] * r[:, 3] + r[:, 0] * r[:, 2])
    R[:, 1, 0] = 2 * (r[:, 1] * r[:, 2] + r[:, 0] * r[:, 3])
    R[:, 1, 1] = 1 - 2 * (r[:, 1] * r[:, 1] + r[:, 3] * r[:, 3])
    R[:, 1, 2] = 2 * (r[:, 2] * r[:, 3] - r[:, 0] * r[:, 1])
    R[:, 2, 0] = 2 * (r[:, 1] * r[:, 3] - r[:, 0] * r[:, 2])
    R[:, 2, 1] = 2 * (r[:, 2] * r[:, 3] + r[:, 0] * r[:, 1])
    R[:, 2, 2] = 1 - 2 * (r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2])

    return R @ L

class GaussianModel:
    def __init__(self, sh_degree):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        # 演化引擎所需狀態
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0.01
        self.spatial_lr_scale = 1.0

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_covariance(self, scaling_modifier=1.0):
        return build_scaling_rotation(self.get_scaling * scaling_modifier, self.get_rotation)

    def create_from_pcd(self, pcd, spatial_lr_scale=1.0):
        print("💥 從點雲初始化高斯球 (Initialize 3D Gaussians)...")
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = (fused_color - 0.5) / 0.28209
        
        print(f"   📊 初始高斯球數量: {fused_point_cloud.shape[0]}")

        pcd_np = np.asarray(pcd.points)
        nn_model = NearestNeighbors(n_neighbors=4, algorithm="auto", metric="euclidean").fit(pcd_np)
        distances, _ = nn_model.kneighbors(pcd_np)
        # 修正：加上平方以防止高斯球巨大化
        dist2 = np.clip((distances[:, 1:]**2).mean(axis=1), a_min=0.0000001, a_max=None)
        scales = torch.log(torch.sqrt(torch.tensor(dist2))[..., None].repeat(1, 3)).float().cuda()
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    # ==========================================================
    # 以下為 3DGS 靈魂：細胞分裂、剪枝與 Optimizer 狀態管理
    # ==========================================================
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """ 收集投影到 2D 螢幕上的位置梯度，做為分裂依據 """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """ 執行分裂與剪枝 """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 1. 複製 (Clone): 針對梯度高且體積小的高斯球
        self.densify_and_clone(grads, max_grad, extent)
        
        # 2. 分裂 (Split): 針對梯度高且體積過大的高斯球
        self.densify_and_split(grads, max_grad, extent)

        # 3. 剪枝 (Prune): 刪除透明度過低或體積異常龐大的高斯球
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        
        # 自己建構旋轉矩陣來偏移新生成的點
        r = self.get_rotation[selected_pts_mask].repeat(N,1)
        rots = torch.zeros((r.shape[0], 3, 3), device="cuda")
        rots[:, 0, 0] = 1 - 2 * (r[:, 2]**2 + r[:, 3]**2)
        rots[:, 0, 1] = 2 * (r[:, 1] * r[:, 2] - r[:, 0] * r[:, 3])
        rots[:, 0, 2] = 2 * (r[:, 1] * r[:, 3] + r[:, 0] * r[:, 2])
        rots[:, 1, 0] = 2 * (r[:, 1] * r[:, 2] + r[:, 0] * r[:, 3])
        rots[:, 1, 1] = 1 - 2 * (r[:, 1]**2 + r[:, 3]**2)
        rots[:, 1, 2] = 2 * (r[:, 2] * r[:, 3] - r[:, 0] * r[:, 1])
        rots[:, 2, 0] = 2 * (r[:, 1] * r[:, 3] - r[:, 0] * r[:, 2])
        rots[:, 2, 1] = 2 * (r[:, 2] * r[:, 3] + r[:, 0] * r[:, 1])
        rots[:, 2, 2] = 1 - 2 * (r[:, 1]**2 + r[:, 2]**2)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = torch.log(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimable_tensors["xyz"]
        self._features_dc = optimable_tensors["f_dc"]
        self._features_rest = optimable_tensors["f_rest"]
        self._opacity = optimable_tensors["opacity"]
        self._scaling = optimable_tensors["scaling"]
        self._rotation = optimable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimable_tensors["opacity"]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest,
             "opacity": new_opacities, "scaling" : new_scaling, "rotation" : new_rotation}

        optimable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimable_tensors["xyz"]
        self._features_dc = optimable_tensors["f_dc"]
        self._features_rest = optimable_tensors["f_rest"]
        self._opacity = optimable_tensors["opacity"]
        self._scaling = optimable_tensors["scaling"]
        self._rotation = optimable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # ==========================================================
    # Optimizer 狀態魔術：安全的 Pytorch 記憶體擴展與截斷
    # ==========================================================
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimable_tensors[group["name"]] = group["params"][0]
        return optimable_tensors

    def _prune_optimizer(self, mask):
        optimable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimable_tensors[group["name"]] = group["params"][0]
        return optimable_tensors

    def replace_tensor_to_optimizer(self, tensor, name):
        optimable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimable_tensors[group["name"]] = group["params"][0]
        return optimable_tensors