"""
Main pipeline script for EfficientTAM video segmentation and evaluation.

Steps:
1. Extract frames from an input video.
2. Run EfficientTAM model inference to generate segmentation masks.
3. Compute IoU metrics versus ground truth.
4. Perform wing deflection and speed correlation analysis.
"""

import os
import torch
import hydra
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, time

# Local imports
from deflection.estimation import process_video
from preprocessing.utils import extract_frames, speed_detection, labelme_json_to_mask
from segmentation.snake_segmentation import snake_segmentation
from segmentation.efficienttam_inference import efficienttam_inference
from segmentation.iou_evaluation import compute_iou_with_visualization, plot_iou_per_frame
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

# Configuration
@dataclass
class ProjectConfig:
    video_path: str
    wing_width_csv_path: str
    output_dir: str
    frames_dir: str
    masks_dir: str
    ann_frames_dir: str
    iou_dir: str
    speed_dir: str
    snake_dir: str
    deflection_dir: str
    exclude_center_fraction = 0.1

def main():
    """
    Run the full EfficientTAM segmentation + deflection analysis pipeline.
    """
    
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
        capability = torch.cuda.get_device_capability()
    else:
        device = torch.device("cpu")
        capability = None
    print(f"[INFO] Using device: {device}, CUDA capability: {capability}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"

    # Initialize Hydra config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=model_cfg_dir, version_base=None)
    hydra.compose(config_name=model_cfg_name)

    # Build EfficientTAM predictor
    predictor = build_efficienttam_video_predictor(model_cfg_name, checkpoint, device=device)    
    
    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]
    output_dir = os.path.join(output_root, base_name)
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.today().date()
    video_dt = datetime.combine(today, video_start_time)
    race_dt = datetime.combine(today, race_start_time)
    difference = video_dt - race_dt

    configs = ProjectConfig(
        video_path=video_path,
        wing_width_csv_path=wing_width_csv_path,
        output_dir=output_dir,
        frames_dir=os.path.join(output_dir, "frames"),
        masks_dir=os.path.join(output_dir, "masks"),
        ann_frames_dir=os.path.join(output_dir, "ann_frames"),
        iou_dir=os.path.join(output_dir, "iou"),
        speed_dir=os.path.join(output_dir, "speed"),
        snake_dir=os.path.join(output_dir, "snake"),
        deflection_dir=os.path.join(output_dir, "deflection_estimation")
    )

    print(f"[INFO] Processing: {video_name}")

    # Step 1: Frame extraction
    extract_frames(video_path=video_path, output_dir=configs.frames_dir)
    # Step 2: Speed detection
    speed_detection(video_path=video_path, start_time_hms=difference, configs=configs)
    # Step 3: Ground-truth mask preparation
    gt_mask_path = labelme_json_to_mask(json_path=json_mask_path, configs=configs)
    
    # Step 4: Snake segmentation
    snake_segmentation(configs)
    # Step 5: EfficientTAM inference
    efficienttam_inference(predictor=predictor, configs=configs)
    # Step 6: IoU computation
    iou_values, frame_numbers = [], []
    for mask in os.listdir(configs.masks_dir):
        frame_id = int(mask.split("_")[0])
        pred_path = os.path.join(configs.masks_dir, mask)
        save_path = os.path.join(configs.iou_dir, f"{frame_id:06d}_overlay.png")
        iou = compute_iou_with_visualization(pred_mask_path=pred_path, gt_mask_path=gt_mask_path, save_path=save_path)
        iou_values.append(iou)
        frame_numbers.append(frame_id)
    
    # Plot IoU results
    if iou_values:
        plot_iou_per_frame(frame_numbers, iou_values, save_path=os.path.join(configs.iou_dir, "iou_scatter_plot.png"))
    
    # Step 7: Deflection + Speed analysis
    overlay_dir = os.path.join(configs.iou_dir, "overlay")
    process_video(video_name=base_name, masks_dir=overlay_dir, configs=configs)

    print(f"[INFO] Pipeline completed successfully. Results saved in: {output_dir}")

if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parents[1]

    # --- INPUT FILES ---
    video_path = os.path.join(BASE_DIR, "data", "AstonMartin_Stroll_Suzuka_2025.mp4") 
    json_mask_path = os.path.join(BASE_DIR, "data", "AstonMartin_Stroll_Suzuka_2025.json")
    wing_width_csv_path=os.path.join(BASE_DIR, "data", "wing_width_measurements.csv")

    # --- TIME PARAMETERS ---
    video_start_time = time(hour=1, minute=17, second=9)
    race_start_time = time(hour=0, minute=8, second=54)

    # --- OUTPUT ROOT ---
    output_root = os.path.join(BASE_DIR, "results")

    # Model setup
    model_cfg_dir = os.path.join(BASE_DIR, "configs", "efficienttam")
    model_cfg_name = "efficienttam_s.yaml"
    checkpoint = os.path.join(BASE_DIR, "checkpoints", "efficienttam_s.pt")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Missing checkpoint file at: {checkpoint}\nPlease run `python setup.py` to download it."
        )

    if not os.path.exists(os.path.join(model_cfg_dir, model_cfg_name)):
        raise FileNotFoundError(
            f"Missing config file at: {model_cfg_dir}\nPlease run `python setup.py` to download it."
        )
    
    main()