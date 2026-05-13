import argparse
from pathlib import Path
import numpy as np
import torch
import os
import sys
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
)





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
    #=====================================================================
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

    parser.add_argument(
        "--enable_gen_3d_prop", 
        action="store_true", 
        help="啟動 3DGIC 範式: 2D 生成式修補 + 3D 昇維映射"
    )

    parser.add_argument(
        "--generate", 
        type=str, 
        help="Specify 'all frame' to render all frames in the data_path."
    )
    
    parser.add_argument(
        "--exp_name", 
        type=str, 
        help="exp_name."
    )
    parser.add_argument(
        "--n_skip",
        type=int,
        default=0,
        help="跳過前 N 張圖（字母排序後），只處理剩餘圖。例如 --n_skip 40 在100張目錄中只取後60張。"
    )#mask_path 不適用 n_skip喔！！！！！！！！！！！！！！！！

    parser.add_argument("--inpaint_method", default="cv2", choices=["cv2","lama","sd"])


    parser.add_argument(
    "--output_root",
    type=Path,
    default=None,
    help="Clean output root directory",
)
    #=====================================================================
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

    # ==================================================
    # Clean output directory structure
    # ==================================================

    if args.output_root is not None:
        output_scene_dir = args.output_root
    else:
        dataset_name = args.data_path.parent.parent.name
        scene_name = args.exp_name if args.exp_name else args.data_path.parent.name

        output_scene_dir = (
            args.output_path
            / dataset_name
            / scene_name
        )

    output_scene_dir.mkdir(parents=True, exist_ok=True)

    # 子資料夾
    inpainted_dir = output_scene_dir / "inpainted"
    deadmask_dir = output_scene_dir / "deadmasks"

    inpainted_dir.mkdir(parents=True, exist_ok=True)
    deadmask_dir.mkdir(parents=True, exist_ok=True)

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

        # ====================================================================
        # 🚀 核心改裝：讀取 0/1 Mask Tensor 並傳給模型 (嚴格順序對應版)
        # ====================================================================
        inpaint_mask = None
        if args.mask_path is not None and args.mask_path.exists():
            print(f"🌀 讀取 3D Inpainting 遮罩...")
            S_len, _, grid_h, grid_w = vgg_input.shape
            
            import re
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]
            
            mask_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp','*.JPG')
            mask_path_list = []
            for ext in mask_extensions:
                mask_path_list.extend(args.mask_path.glob(ext))
                
            mask_path_list = sorted(mask_path_list, key=natural_sort_key)

            if len(mask_path_list) != len(image_paths):
                raise ValueError(f"❌ 嚴重錯誤：圖片數量 ({len(image_paths)}) 與 Mask 數量 ({len(mask_path_list)}) 不對等！")

            masks_tensor = torch.zeros((1, S_len, grid_h, grid_w), dtype=dtype, device='cuda')
            
            from PIL import Image
            for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_path_list)):
                mask_img = Image.open(mask_path).convert('L')
                mask_img = mask_img.resize((grid_w, grid_h), Image.NEAREST)
                mask_np = np.array(mask_img) > 0 
                masks_tensor[0, i] = torch.from_numpy(mask_np).to(dtype=dtype, device='cuda')
                
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
            depth_conf_np,     # 🟢 修正：對應 eval_utils 中返回的 9 個變數
            dense_features_np,
            raw_depth_maps      # 🟢 新增：接住未過濾的連續深度鷹架 (給 Diffusion 結合用) # 🟢 修正：對應 eval_utils 中返回的 9 個變數
        ) = infer_vggt_and_reconstruct(
            model, vgg_input, dtype, args.depth_conf_thresh, image_paths, inpaint_mask=inpaint_mask
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

                # 引用我們剛剛建立的新模組
                from eval.generative_inpaint_module_360 import generative_multi_ref_propagation

                # =================================================================
                # 3DGIC 風格 Inpainting：每個 target 獨立用所有其他 view 當 donor
                # 不再需要 greedy ref 升格，因為 mask refinement 是 closed-form
                # =================================================================
                global_ref_cache = {}
                from eval.dead_zone_inpainter import build_inpainter
                global_ref_cache["_src_dilation_px"] = 11
                global_ref_cache["_tgt_dilation_px"] = 5
                global_ref_cache["_use_poisson"]     = False
                global_ref_cache["_phot_z_thresh"]   = 2.5#3.0
                global_ref_cache["_phot_ring_px"]    = 20
                global_ref_cache["_local_bg_radius"] = 20

                # Debug：選一張黑影特別嚴重的 frame，存中間狀態
                global_ref_cache["_debug_dump_dir"]       = "debug_dump"
                global_ref_cache["_debug_target_indices"] = [0]   # 改成你要 debug 的那張的 idx
                global_ref_cache["_min_trusted_blob"] = 1000  # Fix A: 孤立 patch 最小存活面積
                global_ref_cache["_bilateral_d"]      = 15    # Fix D: bilateral 直徑
                global_ref_cache["_bilateral_sigma"]  = 30.0  # Fix D: color sigma
                global_ref_cache["_inpainter"] = build_inpainter(args.inpaint_method)


                # v5 Shadow Detection params
                global_ref_cache["_shadow_search_px"]  = 100    # 搜尋距離，長陰影場景改 100
                global_ref_cache["_shadow_thresh_k"]   = 4.0   # 亮度門檻，越低抓越多
                global_ref_cache["_min_shadow_blob"]   = 150   # 最小陰影 blob (px)
                global_ref_cache["_bright_untrust_k"]  = 2.0   # M2 target-side 門檻

                ALL_FRAMES = len(image_paths)
                if args.generate == "all frame":
                    print("generate all frame...")
                    target_indices_to_test = list(range(ALL_FRAMES))
                else:
                    target_indices_to_test = input(
                        '請輸入想測試/修補的 Target 視角 Index（逗號分隔，例如 0,27,56）: '
                    )
                    target_indices_to_test = [int(x) for x in target_indices_to_test.split(',')]

                print(f"\n🚀 [3DGIC] 對 {len(target_indices_to_test)} 個 target 執行 inpainting...")
                for tgt_idx in target_indices_to_test:
                    red_area, _ = generative_multi_ref_propagation(
                        ref_indices=[],   # 不再使用，但保留參數相容
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
                        
                print("="*60 + "\n")


        #PAUL_MOD END
        # ================================
        # ================================



        # Check results
        if not all_cam_to_world_mat or not all_world_points:
            print(f"❌ Error: Failed to obtain valid camera poses or point clouds")
            return

        # print(f"✅ Inference done, obtained {len(all_world_points)} point sets")


        print(f"🎉 Done!")

    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()