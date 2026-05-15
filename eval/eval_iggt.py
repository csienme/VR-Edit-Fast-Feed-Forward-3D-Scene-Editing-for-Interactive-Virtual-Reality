import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import copy
import shutil
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cv2

# PAUL MOD

# Ensure project root is in sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    evaluate_scene_and_save,
    compute_original_coords,   # ← 新增
)
from vggt.utils.geometry import unproject_depth_map_to_point_map   # ← 新增
from vggt.utils.helper import (                                      # ← 新增
    create_pixel_coordinate_grid,
    randomly_limit_trues,
)

try:
    import pycolmap
    _PYCOLMAP_AVAILABLE = True
except ImportError:
    _PYCOLMAP_AVAILABLE = False
    print("⚠️  pycolmap not found — COLMAP export disabled")

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False


# ==============================================================================
# COLMAP helper functions (ported from eval_custom_colmap_masked.py)
# ==============================================================================
def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    if camera_type == "PINHOLE":
        return np.array([
            intrinsics[fidx][0, 0],
            intrinsics[fidx][1, 1],
            intrinsics[fidx][0, 2],
            intrinsics[fidx][1, 2],
        ])
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        return np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    else:
        raise ValueError(f"Camera type {camera_type} not supported")


def _batch_np_matrix_to_pycolmap_wo_track(
    points3d, points_xyf, points_rgb,
    extrinsics, intrinsics, image_size,
    shared_camera=False, camera_type="PINHOLE",
):
    N = len(extrinsics)
    P = len(points3d)
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(
            points3d[vidx], pycolmap.Track(), points_rgb[vidx]
        )

    camera = None
    for fidx in range(N):
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)
            camera = pycolmap.Camera(
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=pycolmap_intri,
                camera_id=fidx + 1,
            )
            reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]),
            extrinsics[fidx][:3, 3],
        )
        image = pycolmap.Image(
            id=fidx + 1,
            name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        points2D_list = []
        point2D_idx = 0
        belongs = points_xyf[:, 2].astype(np.int32) == fidx
        belongs_idx = np.nonzero(belongs)[0]

        for pt3d_batch_idx in belongs_idx:
            point3D_id = pt3d_batch_idx + 1
            point2D_xy = points_xyf[pt3d_batch_idx][:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except Exception:
            print(f"  ⚠️  frame {fidx + 1}: no 2D points assigned")
            image.registered = False

        reconstruction.add_image(image)

    return reconstruction


def _rename_colmap_recons_and_rescale_camera(
    reconstruction, image_names, original_coords, img_size,
    shift_point2d_to_original_res=False, shared_camera=False,
):
    rescale_camera = True
    for pyimageid in reconstruction.images:
        pyimage  = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_names[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            pred_params[-2:] = real_image_size / 2
            pycamera.params  = pred_params
            pycamera.width   = int(real_image_size[0])
            pycamera.height  = int(real_image_size[1])

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


# ==============================================================================
# [新增] 核心：從第一次 VGGT 推論結果直接匯出 COLMAP（不需要第二次推論）
# ==============================================================================
def export_colmap_from_vggt(
    extrinsic_np: np.ndarray,    # (N, 3, 4)  w2c — 直接來自 infer_vggt_and_reconstruct
    intrinsic_np: np.ndarray,    # (N, 3, 3)  在 vgg_input 解析度下
    raw_depth_maps: np.ndarray,  # (N, H, W)  未過濾深度（無 NaN）
    depth_conf_np: np.ndarray,   # (N, H, W)  信心分數
    vgg_input: torch.Tensor,     # (N, 3, H, W) 正規化輸入圖（fallback 顏色）
    image_paths: list,           # 原始圖片路徑列表（用來計算 original_coords）
    inpainted_dir: Path,         # inpainted_*.png 所在目錄
    colmap_dir: Path,            # 輸出：colmap/sparse/ 和 colmap/images/
    depth_conf_thresh: float = 3.0,
    max_points: int = 100_000,
):
    """
    用第一次 VGGT 推論的 pose/depth 直接生成 COLMAP 格式。

    相較於原本的做法（inpainted → 再跑一次 VGGT → COLMAP），此函式：
      • 無第二次 VGGT 推論（省時 ~50%）
      • 相機位置來自 original frame 的 VGGT（含 attention bias 的 bg depth）
      • 點雲顏色取自 inpainted 結果（物件已移除）
      • COLMAP images 直接引用 inpainted_*.png

    呼叫前提：inpainting loop 已跑完，inpainted_dir 下已有所有 inpainted_*.png
    """
    if not _PYCOLMAP_AVAILABLE:
        print("⚠️  pycolmap not available, skipping COLMAP export")
        return

    from PIL import Image as PILImage

    N = extrinsic_np.shape[0]
    _, _, grid_h, grid_w = vgg_input.shape  # VGGT 內部解析度
    img_size_wh = np.array([grid_w, grid_h])

    print(f"\n📦 [COLMAP Export] 使用第一次 VGGT 推論結果，不重新推論")
    print(f"   N={N} frames | vgg grid={grid_w}×{grid_h} | conf_thresh={depth_conf_thresh}")

    # ── 1. 用 raw depth 反投影到 3D 世界座標 ──────────────────────
    # raw_depth_maps 無 NaN；信心過濾在點選擇階段進行（同 colmap_masked 做法）
    points_3d = unproject_depth_map_to_point_map(
        raw_depth_maps, extrinsic_np, intrinsic_np
    )
    # points_3d: (N, H, W, 3)

    # ── 2. 點雲顏色：優先用 inpainted 圖（物件已移除），fallback 用 vgg_input ──
    inpainted_files = sorted(
        inpainted_dir.glob("inpainted_*.png"),
        key=lambda p: int(p.stem.split("_")[-1])
    )

    if len(inpainted_files) == N:
        print(f"   🎨 從 {N} 張 inpainted 圖取顏色 (resize to {grid_w}×{grid_h})")
        inpainted_tensors = []
        for p in inpainted_files:
            img = PILImage.open(str(p)).convert("RGB")
            img = img.resize((grid_w, grid_h), PILImage.BILINEAR)
            arr = np.array(img).astype(np.float32) / 255.0   # (H, W, 3)
            inpainted_tensors.append(arr.transpose(2, 0, 1))  # (3, H, W)
        color_np = np.stack(inpainted_tensors, axis=0)        # (N, 3, H, W)
        color_tensor = torch.from_numpy(color_np)
    else:
        print(f"   ⚠️  inpainted 檔案數 ({len(inpainted_files)}) ≠ N ({N})，使用 vgg_input 顏色")
        color_tensor = vgg_input

    points_rgb = (color_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)  # (N, H, W, 3)

    # ── 3. 建立像素座標網格（含 frame index）─────────────────────
    num_frames, height, width, _ = points_3d.shape
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    # ── 4. 信心過濾 + 隨機降採樣（同 colmap_masked）─────────────
    conf_mask = depth_conf_np >= depth_conf_thresh
    conf_mask = randomly_limit_trues(conf_mask, max_points)

    pts3d_f = points_3d[conf_mask]
    xyf_f   = points_xyf[conf_mask]
    rgb_f   = points_rgb[conf_mask]

    print(f"   📍 有效點數：{len(pts3d_f):,} / {num_frames * height * width:,}")

    # ── 5. 建立 pycolmap Reconstruction ──────────────────────────
    reconstruction = _batch_np_matrix_to_pycolmap_wo_track(
        pts3d_f, xyf_f, rgb_f,
        extrinsic_np, intrinsic_np, img_size_wh,
        shared_camera=False, camera_type="PINHOLE",
    )

    # ── 6. 將 image name 改為 inpainted 檔名 + 縮放回原始解析度 ──
    if len(inpainted_files) == N:
        base_names = [p.name for p in inpainted_files]
    else:
        base_names = [f"image_{i+1}" for i in range(N)]

    # compute_original_coords 用原始圖（尺寸相同，省得重開 inpainted）
    original_coords_np = compute_original_coords(image_paths).cpu().numpy()

    reconstruction = _rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_names,
        original_coords_np,
        img_size=grid_w,             # VGGT 內部寬度
        shift_point2d_to_original_res=True,
        shared_camera=False,
    )

    # ── 7. 寫出 sparse/ ──────────────────────────────────────────
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(sparse_dir))
    print(f"   💾 COLMAP sparse → {sparse_dir}")

    # ── 8. 把 inpainted 圖複製到 colmap/images/ ──────────────────
    images_out = colmap_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for src in inpainted_files:
        dst = images_out / src.name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            n_copied += 1
    print(f"   💾 複製 {n_copied} 張 inpainted 圖 → {images_out}")

    # ── 9. 匯出點雲 PLY（可視化用）───────────────────────────────
    if _TRIMESH_AVAILABLE:
        try:
            trimesh.PointCloud(pts3d_f, colors=rgb_f).export(
                str(sparse_dir / "points.ply")
            )
            print(f"   💾 PLY → {sparse_dir / 'points.ply'}")
        except Exception as e:
            print(f"   ⚠️  PLY 匯出失敗: {e}")

    print(f"✅ [COLMAP Export] 完成\n")


# ==================================================
# ================ PAUL CUSTOM START ==============
# ==================================================
def find_best_center_reference_view(all_cam_to_world_mat):
    """
    計算所有相機在 3D 空間中的幾何質心，並回傳最接近質心的相機 Index 作為最佳 Reference View。
    """
    print("\n" + "="*50)
    print("🔭 啟動 [自動空間中心視角選擇器]")

    cam_centers = []
    for w2c in all_cam_to_world_mat:
        c2w = np.linalg.inv(w2c)
        camera_position = c2w[:3, 3]
        cam_centers.append(camera_position)

    cam_centers = np.array(cam_centers)
    centroid    = np.mean(cam_centers, axis=0)
    distances   = np.linalg.norm(cam_centers - centroid, axis=1)
    best_ref_idx = np.argmin(distances)
    min_distance = distances[best_ref_idx]

    print(f"📍 空間質心座標: {centroid}")
    print(f"🏆 最佳中心相機 Index 判定為: V_{best_ref_idx} (距離質心 {min_distance:.4f} 米)")
    print("="*50 + "\n")

    return int(best_ref_idx)


# ==================================================
# ================ PAUL CUSTOM END ==============
# ==================================================


def main():
    """
    Evaluation script for a Custom Dataset.
    Supports optional evaluation and custom dataset structure.
    """
    parser = argparse.ArgumentParser(
        description="Run FastVGGT evaluation on a Custom Dataset"
    )

    # Required: dataset path
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Dataset path containing subfolders: color, depth, gt_ply, pose",
    )

    # Optional: enable evaluation
    parser.add_argument(
        "--enable_evaluation",
        action="store_true",
        help="Enable evaluation (requires pose and ply data)",
    )

    # Output path
    parser.add_argument(
        "--output_path",
        type=Path,
        default="./eval_results_custom",
        help="Output path for evaluation results",
    )

    # Model parameters
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./model_tracker_fixed_e20.pt",
        help="Model checkpoint file path",
    )

    parser.add_argument("--merging", type=int, default=0, help="Merging parameter")

    parser.add_argument(
        "--merge_ratio",
        type=float,
        default=0.9,
        help="Token merge ratio (0.0-1.0)",
    )

    # Processing parameters
    parser.add_argument(
        "--input_frame",
        type=int,
        default=300,
        help="Maximum number of frames to process per scene",
    )

    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=3.0,
        help="Depth confidence threshold to filter low-confidence depth values",
    )

    # Evaluation parameters (only used when evaluation is enabled)
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="Maximum distance threshold used in Chamfer Distance computation",
    )

    parser.add_argument("--plot", action="store_true", help="Whether to generate plots")

    parser.add_argument(
        "--vis_attn_map",
        action="store_true",
        help="Visualize attention maps during inference",
    )

    # PAUL_MOD START
    # =====================================================================
    parser.add_argument(
        "--enable_mask_prop",
        action="store_true",
        help="啟動 Mask 跨視角傳播驗證",
    )
    parser.add_argument(
        "--mask_path",
        type=Path,
        default=None,
        help="存放 Ground Truth Mask 的資料夾路徑 (例如 mini_test/label)",
    )

    parser.add_argument(
        "--enable_gen_3d_prop",
        action="store_true",
        help="啟動 3DGIC 範式: 2D 生成式修補 + 3D 昇維映射",
    )

    parser.add_argument(
        "--generate",
        type=str,
        help="Specify 'all frame' to render all frames in the data_path.",
    )

    parser.add_argument("--exp_name", type=str, help="exp_name.")

    parser.add_argument(
        "--n_skip",
        type=int,
        default=0,
        help="跳過前 N 張圖（字母排序後），只處理剩餘圖。例如 --n_skip 40 在100張目錄中只取後60張。",
    )  # mask_path 不適用 n_skip！

    parser.add_argument(
        "--inpaint_method", default="cv2", choices=["cv2", "lama", "sd"]
    )

    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Clean output root directory",
    )

    # [新增] 控制是否在 Step 1 就產出 COLMAP（預設開啟）
    parser.add_argument(
        "--export_colmap",
        action="store_true",
        default=True,
        help="在 inpainting 完成後直接匯出 COLMAP（取代原本的 Step 2）",
    )
    parser.add_argument(
        "--colmap_max_points",
        type=int,
        default=100_000,
        help="COLMAP 點雲最大點數",
    )
    # =====================================================================
    # PAUL_MOD END

    args = parser.parse_args()
    torch.manual_seed(33)

    # Check data path exists
    if not args.data_path.exists():
        print(f"❌ Error: Data path does not exist: {args.data_path}")
        return

    # Check required subdirectories
    color_dir = args.data_path
    pose_dir  = args.data_path / "pose"

    if not color_dir.exists():
        print(f"❌ Error: color directory does not exist: {color_dir}")
        return

    print(f"📁 Dataset path: {args.data_path}")

    # If evaluation is enabled, check pose and gt_ply directories
    if args.enable_evaluation:
        if not pose_dir.exists():
            print(f"❌ Error: Evaluation requires pose directory: {pose_dir}")
            return
        gt_ply_dir = args.data_path / "gt_ply"
        if not gt_ply_dir.exists():
            print(f"❌ Error: Evaluation requires gt_ply directory: {gt_ply_dir}")
            return
        print(f"📊 Evaluation will use Ground Truth")
    else:
        print(f"🏃 Inference only, no evaluation")

    # ==================================================
    # Clean output directory structure
    # ==================================================
    if args.output_root is not None:
        output_scene_dir = args.output_root
    else:
        dataset_name = args.data_path.parent.parent.name
        scene_name = args.exp_name if args.exp_name else args.data_path.parent.name
        output_scene_dir = args.output_path / dataset_name / scene_name

    output_scene_dir.mkdir(parents=True, exist_ok=True)

    # 子資料夾
    inpainted_dir = output_scene_dir / "inpainted"
    deadmask_dir  = output_scene_dir / "deadmasks"
    colmap_dir    = output_scene_dir / "colmap"    # ← 新增

    inpainted_dir.mkdir(parents=True, exist_ok=True)
    deadmask_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    if (output_scene_dir / "metrics.json").exists() and args.enable_evaluation:
        print(f"⚠️  Results already exist, skipping: {output_scene_dir / 'metrics.json'}")
        return

    # Force use of bf16 dtype
    dtype = torch.bfloat16

    # Load VGGT model
    print(f"🔄 Loading model: {args.ckpt_path}")
    model = VGGT(
        merging=args.merging,
        merge_ratio=args.merge_ratio,
        vis_attn_map=args.vis_attn_map,
    )
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    incompat = model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    print(f"✅ Model loaded")

    # Load scene data
    image_paths = get_sorted_image_paths(color_dir)
    if args.n_skip > 0:
        image_paths = image_paths[args.n_skip:]
        print(f"⏭️  Skipped first {args.n_skip} images, using remaining {len(image_paths)} images")
    if len(image_paths) == 0:
        print(f"❌ Error: No images found in {color_dir}")
        return

    print(f"🖼️  Found {len(image_paths)} images")

    # Process pose data (if evaluation is enabled)
    poses_gt = None
    first_gt_pose = None
    available_pose_frame_ids = None
    c2ws = None

    if args.enable_evaluation:
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_dir)
        if poses_gt is None or first_gt_pose is None or available_pose_frame_ids is None:
            print(f"❌ Error: Failed to load pose data")
            return
        print(f"📐 Loaded {len(poses_gt)} poses")

    # Frame selection
    if args.enable_evaluation and available_pose_frame_ids is not None:
        selected_frame_ids, selected_image_paths, selected_pose_indices = (
            build_frame_selection(image_paths, available_pose_frame_ids, args.input_frame)
        )
        c2ws = poses_gt[selected_pose_indices]
        image_paths = selected_image_paths
    else:
        num_frames = min(len(image_paths), args.input_frame)
        selected_frame_ids = list(range(num_frames))
        image_paths = image_paths[:num_frames]

    print(f"📋 Selected {len(image_paths)} frames for processing")

    try:
        # Load images
        print(f"🔄 Loading images...")
        images = load_images_rgb(image_paths)

        if not images or len(images) < 3:
            print(f"❌ Error: Not enough valid images (need at least 3)")
            return

        frame_ids = selected_frame_ids
        images_array = np.stack(images)
        vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
        print(f"📐 Image patch dimensions: {patch_width}x{patch_height}")

        # Update attention layer patch dimensions in the model
        model.update_patch_dimensions(patch_width, patch_height)

        # ====================================================================
        # 🚀 核心改裝：讀取 0/1 Mask Tensor 並傳給模型 (嚴格順序對應版)
        # ====================================================================
        inpaint_mask = None
        mask_path_list = []
        if args.mask_path is not None and args.mask_path.exists():
            print(f"🌀 讀取 3D Inpainting 遮罩...")
            S_len, _, grid_h, grid_w = vgg_input.shape

            import re
            def natural_sort_key(s):
                return [
                    int(text) if text.isdigit() else text.lower()
                    for text in re.split("([0-9]+)", str(s))
                ]

            mask_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.JPG")
            for ext in mask_extensions:
                mask_path_list.extend(args.mask_path.glob(ext))

            mask_path_list = sorted(mask_path_list, key=natural_sort_key)

            if len(mask_path_list) != len(image_paths):
                raise ValueError(
                    f"❌ 嚴重錯誤：圖片數量 ({len(image_paths)}) 與 Mask 數量 ({len(mask_path_list)}) 不對等！"
                )

            masks_tensor = torch.zeros((1, S_len, grid_h, grid_w), dtype=dtype, device="cuda")

            from PIL import Image
            for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_path_list)):
                mask_img = Image.open(mask_path).convert("L")
                mask_img = mask_img.resize((grid_w, grid_h), Image.NEAREST)
                mask_np = np.array(mask_img) > 0
                masks_tensor[0, i] = torch.from_numpy(mask_np).to(dtype=dtype, device="cuda")

            inpaint_mask = masks_tensor
            print(f"✅ {len(mask_path_list)} 張遮罩已按嚴格順序載入，Attention Bias 引擎啟動準備完成。")

        # Inference + Reconstruction
        print(f"🚀 Start inference and reconstruction...")
        (
            extrinsic_np,
            intrinsic_np,
            all_world_points,
            all_point_colors,
            all_cam_to_world_mat,
            inference_time_ms,
            dense_depth_maps,
            depth_conf_np,
            dense_features_np,
            raw_depth_maps,
        ) = infer_vggt_and_reconstruct(
            model, vgg_input, dtype, args.depth_conf_thresh, image_paths,
            inpaint_mask=inpaint_mask,
        )
        print(f"⏱️  Inference time: {inference_time_ms:.2f}ms")

        # ==================================================
        # [新增] 執行 Generative 3D Inpainting (3DGIC Pipeline)
        # ==================================================
        if args.enable_gen_3d_prop:
            if args.mask_path is None or not args.mask_path.exists():
                print("❌ 錯誤: 啟動了 --enable_gen_3d_prop 但未提供 --mask_path")
            else:
                print(f"\n🚀 啟動 Generative 3D Inpainting 映射")

                output_img_dir = inpainted_dir

                from eval.generative_inpaint_module_360 import generative_multi_ref_propagation

                global_ref_cache = {}
                from eval.dead_zone_inpainter import build_inpainter
                global_ref_cache["_src_dilation_px"] = 11
                global_ref_cache["_tgt_dilation_px"] = 5
                global_ref_cache["_use_poisson"]     = False
                global_ref_cache["_phot_z_thresh"]   = 2.5
                global_ref_cache["_phot_ring_px"]    = 20
                global_ref_cache["_local_bg_radius"] = 20

                global_ref_cache["_debug_dump_dir"]       = "debug_dump"
                global_ref_cache["_debug_target_indices"] = [0]
                global_ref_cache["_min_trusted_blob"] = 1000
                global_ref_cache["_bilateral_d"]      = 15
                global_ref_cache["_bilateral_sigma"]  = 30.0
                global_ref_cache["_inpainter"] = build_inpainter(args.inpaint_method)

                # v5 Shadow Detection params
                global_ref_cache["_shadow_search_px"]  = 100
                global_ref_cache["_shadow_thresh_k"]   = 4.0
                global_ref_cache["_min_shadow_blob"]   = 150
                global_ref_cache["_bright_untrust_k"]  = 2.0

                ALL_FRAMES = len(image_paths)
                if args.generate == "all frame":
                    print("generate all frame...")
                    target_indices_to_test = list(range(ALL_FRAMES))
                else:
                    target_indices_to_test = input(
                        "請輸入想測試/修補的 Target 視角 Index（逗號分隔，例如 0,27,56）: "
                    )
                    target_indices_to_test = [int(x) for x in target_indices_to_test.split(",")]

                print(f"\n🚀 [3DGIC] 對 {len(target_indices_to_test)} 個 target 執行 inpainting...")
                for tgt_idx in target_indices_to_test:
                    red_area, _ = generative_multi_ref_propagation(
                        ref_indices=[],
                        target_idx=tgt_idx,
                        image_paths=image_paths,
                        mask_dir=args.mask_path,
                        raw_depth_maps=raw_depth_maps,
                        all_cam_to_world_mat=all_cam_to_world_mat,
                        intrinsics=intrinsic_np,
                        output_dir=output_img_dir,
                        ref_cache=global_ref_cache,
                        mask_paths=mask_path_list if mask_path_list else None,
                    )
                    print(f"   ✅ V_{tgt_idx} 完成 (refined hole={red_area} px)")

                print(f"\n🏆 全部 {len(target_indices_to_test)} 個 target 處理完成")
                print("=" * 60 + "\n")

                # ==================================================
                # [新增 v6] Inpainting 結束後直接匯出 COLMAP
                # 取代原本 bash 的 Step 2 + Step 3
                # ==================================================
                if args.export_colmap and args.generate == "all frame":
                    export_colmap_from_vggt(
                        extrinsic_np      = extrinsic_np,
                        intrinsic_np      = intrinsic_np,
                        raw_depth_maps    = raw_depth_maps,
                        depth_conf_np     = depth_conf_np,
                        vgg_input         = vgg_input,
                        image_paths       = image_paths,      # 原始圖路徑（算 original_coords 用）
                        inpainted_dir     = inpainted_dir,    # inpainted_*.png 在這裡
                        colmap_dir        = colmap_dir,       # 輸出到這裡
                        depth_conf_thresh = args.depth_conf_thresh,
                        max_points        = args.colmap_max_points,
                    )
                elif args.export_colmap and args.generate != "all frame":
                    print("⚠️  COLMAP export 只在 --generate 'all frame' 時執行（需要所有視角）")

        # PAUL_MOD END

        # Check results
        if not all_cam_to_world_mat or not all_world_points:
            print(f"❌ Error: Failed to obtain valid camera poses or point clouds")
            return

        print(f"🎉 Done!")

    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()