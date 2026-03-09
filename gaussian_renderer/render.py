import math
import torch

try:
    from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except Exception as e:
    print("🚨 無法載入 r3dg_rasterization！")
    exit(1)

def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    """
    核心渲染函數：完全釋放 SH 球諧函數的物理渲染能力
    """
    means3D = pc.get_xyz.contiguous()
    opacity = pc.get_opacity.contiguous()
    scales = pc.get_scaling.contiguous()
    rotations = pc.get_rotation.contiguous()
    
    P = means3D.shape[0]

    # 💥 解封 1：直接傳遞完整 16 通道的高階球諧函數給 CUDA！
    shs = pc.get_features.contiguous() 
    
    # 填補 3DGIS C++ 底層要求的 PBR 記憶體黑洞
    sh_objs = torch.zeros((P, 8), dtype=torch.float32, device="cuda")
    features = torch.zeros((P, 0), dtype=torch.float32, device="cuda")

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 💥 解封 2：讓 C++ 根據目前的活躍 SH 階數來計算物理光影
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(viewpoint_camera.image_width / 2.0),
        cy=float(viewpoint_camera.image_height / 2.0),
        bg=bg_color.contiguous(),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.contiguous(),
        projmatrix=viewpoint_camera.full_proj_transform.contiguous(),
        sh_degree=pc.active_sh_degree,  # 動態調整 0 -> 1 -> 2 -> 3 階
        campos=viewpoint_camera.camera_center.contiguous(),
        prefiltered=False,
        backward_geometry=False,
        computer_pseudo_normal=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 💥 解封 3：傳入 shs，並嚴格將 colors_precomp 設為 None！
    raster_tuple = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs,                  # 原汁原味的 SH 送進去了！
        sh_objs=sh_objs,          
        colors_precomp=None,      # 讓 CUDA 底層親自算顏色！
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        features=features         
    )

    # 精準解包
    (num_rendered, num_contrib, rendered_image, rendered_objects, 
     rendered_opacity, rendered_depth, rendered_feature, 
     rendered_pseudo_normal, rendered_surface_xyz, radii) = raster_tuple

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    }