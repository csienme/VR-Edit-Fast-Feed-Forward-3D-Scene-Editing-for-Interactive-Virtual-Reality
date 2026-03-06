import argparse
from pathlib import Path
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


# PAUL MOD
import generative_inpaint_module as generative_inpaint_module

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
)

# Import pose visualization libraries (optional EVO support)
try:
    from evo.core.trajectory import PoseTrajectory3D
    import evo.tools.plot as plot

    EVO_AVAILABLE = True
except ImportError:
    # EVO is optional; we have a matplotlib-based fallback
    EVO_AVAILABLE = False


def visualize_predicted_poses(
    all_cam_to_world_mat, frame_ids, output_scene_dir, scene_name="custom_dataset"
):
    """
    Visualize the predicted camera pose trajectory (no GT comparison required).

    Args:
        all_cam_to_world_mat: List of camera-to-world transform matrices
        frame_ids: List of frame IDs
        output_scene_dir: Output directory
        scene_name: Scene name
    """
    # Provide basic pose visualization even without EVO
    if not EVO_AVAILABLE:
        print("⚠️  EVO not installed; using basic matplotlib visualization")

    try:
        # Convert to numpy array
        poses_est = np.array(all_cam_to_world_mat)

        if len(poses_est) < 2:
            print("⚠️  Not enough poses to generate trajectory plot")
            return

        print(f"🎨 Generating pose trajectory visualization...")

        # Extract translation part
        positions = poses_est[:, :3, 3]  # shape: (N, 3)

        # Create figure - show XZ-plane projection only
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # XZ-plane projection
        ax.plot(
            positions[:, 0],
            positions[:, 2],
            "b-",
            linewidth=2,
            label="Predicted Trajectory",
        )
        ax.scatter(
            positions[0, 0], positions[0, 2], color="green", s=100, label="Start"
        )
        ax.scatter(positions[-1, 0], positions[-1, 2], color="red", s=100, label="End")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{scene_name} - XZ-plane projection")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save image
        pose_plot_path = output_scene_dir / "predicted_trajectory.png"
        plt.savefig(pose_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Trajectory visualization saved: {pose_plot_path}")

    except Exception as e:
        print(f"⚠️  Failed to generate pose visualization: {e}")
        import traceback

        traceback.print_exc()


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
    
    # 1. 萃取所有相機的 3D 座標 (Translation vector)
    for w2c in all_cam_to_world_mat:
        # 將 World-to-Camera 矩陣反轉為 Camera-to-World
        c2w = np.linalg.inv(w2c)
        # C2W 矩陣的右上角 3x1 向量，就是相機在世界座標系中的 (X, Y, Z) 位置
        camera_position = c2w[:3, 3] 
        cam_centers.append(camera_position)
        
    cam_centers = np.array(cam_centers)
    
    # 2. 計算所有相機軌跡的「幾何質心 (Centroid)」
    centroid = np.mean(cam_centers, axis=0)
    
    # 3. 計算每一台相機距離質心的歐式距離 (L2 Norm)
    distances = np.linalg.norm(cam_centers - centroid, axis=1)
    
    # 4. 找出距離質心最近的那台相機
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


    #PAUL_MOD START
    # === [新增] 驗證模組參數 ===
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



    # === [新增] Oracle RGB 紋理映射測試參數 ===
    parser.add_argument(
        "--enable_rgb_prop", 
        action="store_true", 
        help="啟動 Oracle RGB 紋理反向映射測試 (需要 61 張圖：1 張乾淨 + 60 張有物體)"
    )
    parser.add_argument(
        "--clean_ref_img_path", 
        type=Path, 
        default=None, 
        help="乾淨無物體 (GT) 的 V_0 影像路徑，做為採樣 RGB 的神諭來源"
    )

    parser.add_argument(
        "--enable_gen_3d_prop", 
        action="store_true", 
        help="啟動 3DGIC 範式: 2D 生成式修補 + 3D 昇維映射"
    )
    # ==========================
    #PAUL_MOD END


    args = parser.parse_args()
    torch.manual_seed(33)

    # Check data path exists
    if not args.data_path.exists():
        print(f"❌ Error: Data path does not exist: {args.data_path}")
        return

    # Check required subdirectories
    color_dir = args.data_path  # / "images"
    pose_dir = args.data_path / "pose"

    if not color_dir.exists():
        print(f"❌ Error: color directory does not exist: {color_dir}")
        return

    print(f"📁 Dataset path: {args.data_path}")
    # print(f"🔧 Enable evaluation: {'Yes' if args.enable_evaluation else 'No'}")

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

    # Create output directory
    args.output_path.mkdir(parents=True, exist_ok=True)
    output_scene_dir = args.output_path / "custom_dataset"

    # Check if already processed
    if (output_scene_dir / "metrics.json").exists() and args.enable_evaluation:
        print(
            f"⚠️  Results already exist, skipping: {output_scene_dir / 'metrics.json'}"
        )
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
    # if incompat.missing_keys or incompat.unexpected_keys:
    #     print(f"⚠️  Partially incompatible keys when loading model: {incompat}")
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    print(f"✅ Model loaded")

    # Load scene data
    image_paths = get_sorted_image_paths(color_dir)
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
        if (
            poses_gt is None
            or first_gt_pose is None
            or available_pose_frame_ids is None
        ):
            print(f"❌ Error: Failed to load pose data")
            return
        print(f"📐 Loaded {len(poses_gt)} poses")

    # Frame selection
    if args.enable_evaluation and available_pose_frame_ids is not None:
        # Use pose data for frame selection
        selected_frame_ids, selected_image_paths, selected_pose_indices = (
            build_frame_selection(
                image_paths, available_pose_frame_ids, args.input_frame
            )
        )
        c2ws = poses_gt[selected_pose_indices]
        image_paths = selected_image_paths
    else:
        # Simply take the first N frames
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

        # Inference + Reconstruction
        print(f"🚀 Start inference and reconstruction...")
        (
            extrinsic_np,
            intrinsic_np,
            all_world_points,
            all_point_colors,
            all_cam_to_world_mat,
            inference_time_ms,
            dense_depth_maps, # PAUL MOD add new return value for depth maps
        ) = infer_vggt_and_reconstruct(
            model, vgg_input, dtype, args.depth_conf_thresh, image_paths
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
                



                # 引用我們剛剛建立的新模組
                from eval.generative_inpaint_module import generative_multi_ref_propagation

            # =================================================================
                # 🚀 終極測試：動態死角貪婪覆蓋 (Greedy Disocclusion Coverage)
                # =================================================================
                # 定義全局 LaMa 快取字典，防止 GPU 運算爆炸
                global_ref_cache = {}
                
                # 1. 創世：利用上一篇寫好的函數，選出空間正中心的相機作為最初的 Ref_1
                best_center_idx = find_best_center_reference_view(all_cam_to_world_mat)
                active_ref_indices = [best_center_idx]
                
                # 你想測試/修補的所有 Target 視角
                target_indices_to_test = [5, 10, 15, 20, 30, 40, 50, 55] 
                
                # 設定死角容忍閾值 (例如紅點小於 200 個像素，我們就視為肉眼不可見，宣告勝利)
                TOLERANCE_AREA = 200  
                round_count = 1
                
                while True:
                    print(f"\n" + "="*60)
                    print(f"🌍 [貪婪迴圈 第 {round_count} 回合] 目前的 Reference 陣容: {active_ref_indices}")
                    print("="*60)
                    
                    max_red_area = 0
                    worst_target_idx = -1
                    
                    # 讓所有的 Target 進行修補，並統計災情
                    for tgt_idx in target_indices_to_test:
                        if tgt_idx in active_ref_indices:
                            continue # 自己是 Ref 就不用再當 Target
                            
                        red_area = generative_multi_ref_propagation(
                            ref_indices=active_ref_indices, 
                            target_idx=tgt_idx, 
                            image_paths=image_paths, 
                            mask_dir=args.mask_path, 
                            dense_depth_maps=dense_depth_maps, 
                            all_cam_to_world_mat=all_cam_to_world_mat, 
                            intrinsics=intrinsic_np, 
                            output_dir=output_scene_dir,
                            ref_cache=global_ref_cache  # 💥 傳入快取！
                        )
                        
                        # 揪出本回合紅斑最嚴重的苦主
                        if red_area > max_red_area:
                            max_red_area = red_area
                            worst_target_idx = tgt_idx
                            
                    # 結算本回合戰況
                    print(f"⚖️ 本回合最慘 Target: V_{worst_target_idx} (最大紅斑面積: {max_red_area})")
                    
                    if max_red_area <= TOLERANCE_AREA:
                        print(f"🏆 貪婪覆蓋大獲全勝！所有視角的死角已被全數消滅 (耗費 {len(active_ref_indices)} 個 Ref Views)。")
                        break
                        
                    # 🚨 核心邏輯：如果紅斑還是太大，代表目前的 Ref 陣容完全看不到那個死角
                    # 解法：直接將最慘的 Target 拔擢為新的 Ref，讓 LaMa 在那裡憑空創造世界！
                    print(f"👑 系統決定拔擢最慘的 V_{worst_target_idx} 成為新的 Reference View！")
                    active_ref_indices.append(worst_target_idx)
                    round_count += 1


        #PAUL_MOD END

        # ================================
        # ================================



        # Check results
        if not all_cam_to_world_mat or not all_world_points:
            print(f"❌ Error: Failed to obtain valid camera poses or point clouds")
            return

        # print(f"✅ Inference done, obtained {len(all_world_points)} point sets")

        # Evaluation and saving
        if args.enable_evaluation:
            print(f"📊 Start evaluation...")
            gt_ply_dir = args.data_path / "gt_ply"
            metrics = evaluate_scene_and_save(
                "custom_dataset",
                c2ws,
                first_gt_pose,
                frame_ids,
                all_cam_to_world_mat,
                all_world_points,
                output_scene_dir,
                gt_ply_dir,
                args.chamfer_max_dist,
                inference_time_ms,
                args.plot,
            )
            if metrics is not None:
                print("📈 Evaluation results:")
                for key, value in metrics.items():
                    if key in [
                        "chamfer_distance",
                        "ate",
                        "are",
                        "rpe_rot",
                        "rpe_trans",
                        "inference_time_ms",
                    ]:
                        print(f"  {key}: {float(value):.4f}")

            # Also visualize predicted poses in evaluation branch
            if args.plot:
                visualize_predicted_poses(
                    all_cam_to_world_mat, frame_ids, output_scene_dir, "custom_dataset"
                )
        else:
            # Save reconstruction only, no evaluation
            print(f"💾 Saving reconstruction...")
            output_scene_dir.mkdir(parents=True, exist_ok=True)

            # Save camera poses
            poses_output_path = output_scene_dir / "estimated_poses.txt"
            with open(poses_output_path, "w") as f:
                for i, pose in enumerate(all_cam_to_world_mat):
                    f.write(f"# Frame {frame_ids[i]}\n")
                    for row in pose:
                        f.write(" ".join(map(str, row)) + "\n")
                    f.write("\n")

            # Save point cloud
            if all_world_points:
                points_output_path = output_scene_dir / "reconstructed_points.ply"

                # Merge all frames' point clouds and colors
                try:
                    merged_point_cloud = np.vstack(all_world_points)
                    merged_colors = (
                        np.vstack(all_point_colors).astype(np.uint8)
                        if all_point_colors is not None and len(all_point_colors) > 0
                        else None
                    )
                    print(
                        f"📊 Merged point clouds: {len(all_world_points)} frames, total {len(merged_point_cloud)} points"
                    )

                    # If too many points, randomly sample 100000 points
                    max_points = 100000
                    if len(merged_point_cloud) > max_points:
                        print(
                            f"🔽 Too many points, randomly sampling {max_points} points..."
                        )
                        # Randomly choose indices
                        indices = np.random.choice(
                            len(merged_point_cloud), size=max_points, replace=False
                        )
                        merged_point_cloud = merged_point_cloud[indices]
                        if merged_colors is not None:
                            merged_colors = merged_colors[indices]
                        print(
                            f"✅ Sampling done, kept {len(merged_point_cloud)} points"
                        )

                    # Save as PLY (with color)
                    with open(points_output_path, "w") as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {len(merged_point_cloud)}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        if merged_colors is not None:
                            f.write("property uchar red\n")
                            f.write("property uchar green\n")
                            f.write("property uchar blue\n")
                        f.write("end_header\n")
                        if merged_colors is None:
                            for point in merged_point_cloud:
                                if not (np.isnan(point).any() or np.isinf(point).any()):
                                    f.write(
                                        f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n"
                                    )
                        else:
                            for point, color in zip(merged_point_cloud, merged_colors):
                                # Check point validity
                                if not (np.isnan(point).any() or np.isinf(point).any()):
                                    r = int(np.clip(color[0], 0, 255))
                                    g = int(np.clip(color[1], 0, 255))
                                    b = int(np.clip(color[2], 0, 255))
                                    f.write(
                                        f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n"
                                    )

                    print(f"💾 Point cloud saved to: {points_output_path}")

                except Exception as e:
                    print(f"⚠️  Error saving point cloud: {e}")
                    # If merge fails, try to log per-frame info
                    print(f"🔍 Point cloud debug info:")
                    for i, frame_points in enumerate(all_world_points):
                        print(
                            f"  Frame {i}: {frame_points.shape if hasattr(frame_points, 'shape') else type(frame_points)}"
                        )
                        if (
                            hasattr(frame_points, "shape")
                            and len(frame_points.shape) >= 2
                        ):
                            print(
                                f"    Shape: {frame_points.shape}, Dtype: {frame_points.dtype}"
                            )
                            if frame_points.shape[0] > 0:
                                print(
                                    f"    Range: x[{np.min(frame_points[:, 0]):.3f}, {np.max(frame_points[:, 0]):.3f}] "
                                    f"y[{np.min(frame_points[:, 1]):.3f}, {np.max(frame_points[:, 1]):.3f}] "
                                    f"z[{np.min(frame_points[:, 2]):.3f}, {np.max(frame_points[:, 2]):.3f}]"
                                )

            print(f"📁 Results saved to: {output_scene_dir}")

            # Visualize predicted pose trajectory
            if args.plot:
                visualize_predicted_poses(
                    all_cam_to_world_mat, frame_ids, output_scene_dir, "custom_dataset"
                )

        print(f"🎉 Done!")

    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
