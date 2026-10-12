import os

from typing import List, Optional
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

def compute_iou_with_visualization(
    pred_mask_path: str,
    gt_mask_path: str,
    save_path: Optional[str] = None,
    white_threshold: int = 250,
) -> float:
    """
    Compute the Intersection over Union (IoU) between a predicted segmentation mask
    (with white background) and a grayscale ground-truth mask.

    Args:
        pred_mask_path (str): Path to the predicted mask (RGB image with white background).
        gt_mask_path (str): Path to the ground-truth mask (grayscale image).
        save_path (str, optional): Path to save the overlay visualization.
        white_threshold (int, optional): Pixel intensity threshold to identify white background.

    Returns:
        float: The IoU score between predicted and ground-truth masks.
    """

    # Load masks
    gt = np.array(Image.open(gt_mask_path).convert("L"))
    pred_mask = np.array(Image.open(pred_mask_path).convert("RGB"))

    # Convert predicted RGB mask to binary (foreground != white)
    white_pixels = np.all(pred_mask > white_threshold, axis=-1)
    pred_mask = ~white_pixels

    # Squeeze singleton dimensions (e.g. (1, H, W) → (H, W))
    if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask[0]

    # Convert boolean to uint8 if needed
    if pred_mask.dtype == np.bool_:
        pred_mask = pred_mask.astype(np.uint8)

    # Resize predicted mask to match ground truth size if needed
    if pred_mask.shape != gt.shape:
        print(f"Shape mismatch: gt {gt.shape}, pred {pred_mask.shape}")
        print("Resize predicted mask to match ground truth size")
        pred_mask = cv2.resize(pred_mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # Normalize both masks to [0, 255]
    gt_norm = ((gt - gt.min()) / (gt.max() - gt.min()) * 255).astype(np.uint8)
    pred_norm = ((pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min()) * 255).astype(np.uint8)

    # Ensure binary
    pred_bin = (pred_norm > 127).astype(np.uint8)
    gt_bin = (gt_norm > 127).astype(np.uint8)

    # Invert ground truth mask if foreground is black (0) and background is white (255)
    gt_bin = (gt_bin == 0).astype(np.uint8)

    # Compute intersection and union
    intersection = np.logical_and(pred_bin, gt_bin).astype(np.uint8)
    union = np.logical_or(pred_bin, gt_bin).astype(np.uint8)

    if union.sum() == 0:
        iou = float(intersection.sum() == 0)
    else:
        iou = intersection.sum() / float(union.sum())

    # Visualization
    if save_path:
        overlap = pred_bin * 2 + gt_bin  # 0=bg, 1=gt, 2=pred, 3=both
        color_map = {
            0: [0, 0, 0],        # background - black
            1: [0, 255, 0],      # ground truth only - green
            2: [255, 0, 0],      # prediction only - red
            3: [255, 255, 0],    # intersection - yellow
        }
        h, w = pred_bin.shape
        visual = np.zeros((h, w, 3), dtype=np.uint8)
        for val, color in color_map.items():
            visual[overlap == val] = color

        overlay_dir = os.path.join(os.path.dirname(save_path), "overlay")
        os.makedirs(overlay_dir, exist_ok=True)
        cv2.imwrite(os.path.join(overlay_dir, os.path.basename(save_path)), cv2.cvtColor(visual, cv2.COLOR_RGB2BGR))
    
    return iou

def plot_iou_per_frame(
    frame_numbers: List[int],
    iou_values: List[float],
    save_path: str,
) -> None:
    """
    Create and save a scatter plot of IoU values for each frame in a video.

    Args:
        frame_numbers (List[int]): List of frame indices.
        iou_values (List[float]): Corresponding IoU values.
        save_path (str): Path to save the scatter plot.
    """

    plt.figure(figsize=(12, 6))
    plt.scatter(frame_numbers, iou_values, c="blue", label="IoU", alpha=0.7)

    mean_iou = np.mean(iou_values)
    plt.axhline(mean_iou, color="red", linestyle="--", label=f"Mean IoU = {mean_iou:.3f}")

    plt.xlabel("Frame Number")
    plt.ylabel("IoU")
    plt.title("IoU per Frame")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved IoU scatter plot to {save_path}")