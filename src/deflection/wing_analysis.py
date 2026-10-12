import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted

def conversion(video_name: str, deflection_pixel: float, configs, wing_width_mm: int = 1050) -> float | None:
    """
    Convert pixel deflection to millimeters using the known real-world wing width.

    Args:
        video_name (str): Name of the video (used to match wing width in CSV).
        deflection_pixel (float): Deflection in pixels.
        configs: Configuration object containing `wing_width_csv_path`.

    Returns:
        float | None: Deflection in millimeters, or None if unavailable.
    """
    if deflection_pixel is None:
        return None
    
    try:
        df = pd.read_csv(configs.wing_width_csv_path)
        matching_rows = df[df["image_name"].str.startswith(video_name)]
        if matching_rows.empty:
            print(f"[WARNING] No wing width measurement found for '{video_name}'")
            return None
        
        pixel_width = matching_rows["wing_width_pixels"].iloc[0]
        conversion_factor = wing_width_mm / pixel_width # 1050 mm = real-world wing width
        return deflection_pixel * conversion_factor
    
    except Exception as e:
        print(f"[ERROR] Conversion failed for {video_name}: {e}")
        return None

def get_wing_edge_points(mask_color: np.ndarray, exclude_center_fraction: float):
    """
    Identify top-edge (upper boundary) points of the car’s rear wing.

    Args:
        mask_color (np.ndarray): Color mask image (BGR format).
        exclude_center_fraction (float): Fraction of image width to exclude in the center.

    Returns:
        tuple:
            mean_top_y (float): Mean Y coordinate of top edge.
            median_top_y (float): Median Y coordinate of top edge.
            left_point (tuple): Leftmost top point (x, y).
            right_point (tuple): Rightmost top point (x, y).
    """
    h, w = mask_color.shape[:2]
    
    # Identify yellow pixels (BGR = [0, 255, 255])
    yellow_pixels = np.all(mask_color == [0, 255, 255], axis=-1)
    if not np.any(yellow_pixels):
        print("[WARNING] No yellow pixels found in mask.")
        return None, None, None, None
    
    center_left = int(w * (0.5 - exclude_center_fraction / 2))
    center_right = int(w * (0.5 + exclude_center_fraction / 2))

    ys, xs = np.where(yellow_pixels)
    
    # Left top point
    left_mask = (xs < center_left)
    left_ys = ys[left_mask]
    left_xs = xs[left_mask]
    left_point = None
    if len(left_ys) > 0:
        min_y_left = np.min(left_ys)
        left_indices = np.where(left_ys == min_y_left)[0]
        left_xs_at_min_y = left_xs[left_indices]
        left_point = (np.min(left_xs_at_min_y), min_y_left)
    
    # Right top point
    right_mask = (xs >= center_right)
    right_ys = ys[right_mask]
    right_xs = xs[right_mask]
    
    right_point = None
    if len(right_ys) > 0:
        min_y_right = np.min(right_ys)
        right_indices = np.where(right_ys == min_y_right)[0]
        right_xs_at_min_y = right_xs[right_indices]
        right_point = (np.max(right_xs_at_min_y), min_y_right)
    
    if left_point is None or right_point is None:
        print("[WARNING] Could not find left or right top points.")
        return None, None, left_point, right_point
    
    # Compute top edge points between left and right
    top_ys = []
    for x in range(left_point[0], right_point[0] + 1):
        if center_left <= x < center_right:
            continue
        mask_x = (xs == x)
        ys_in_x = ys[mask_x]
        
        if len(ys_in_x) > 0:
            min_y_in_x = np.min(ys_in_x)
            top_ys.append(min_y_in_x)
    
    if not top_ys:
        print("[WARNING] No valid top-edge points found in the selected range.")
        return None, None, left_point, right_point
    
    mean_top_y = np.mean(top_ys)
    median_top_y = np.median(top_ys)
    
    return mean_top_y, median_top_y, left_point, right_point


def computation_on_frames(masks_dir: str, video_name: str, ref_mean_top_y: float, ref_median_top_y:float, speed_data: pd.DataFrame, configs):
    """
    Compute deflection (in pixels and mm) for each frame mask.

    Args:
        masks_dir (str): Directory containing mask images.
        video_name (str): Name of the video being processed.
        ref_mean_top_y (float): Mean Y coordinate of reference frame.
        ref_median_top_y (float): Median Y coordinate of reference frame.
        speed_data (pd.DataFrame): Speed data per frame.
        configs: Configuration object with `exclude_center_fraction` and `wing_width_csv_path`.

    Returns:
        list[dict]: Deflection results for each frame.
    """
    mask_files = natsorted(glob(os.path.join(masks_dir, '*.png')))
    if not mask_files:
        print(f"[ERROR] No mask files found in directory: {masks_dir}")
        return None

    print(f"\n[INFO] Starting deflection computation for video: {video_name}")
    results = []
    for i, mask_path in enumerate(mask_files):
        if i % 10 == 0:  # Print progress every 10 frames
            print(f"[INFO] Processing frame {i+1}/{len(mask_files)}")
            
        mask_name = os.path.basename(mask_path)
        mask_color = cv2.imread(mask_path)

        if mask_color is None:
            print(f"[WARNING] Could not read mask: {mask_path}")
            continue
        
        # Extract frame number from filename
        try:
            frame_num = int(mask_name.split('_')[0])
        except ValueError:
            print(f"[WARNING] Invalid frame name: {mask_name}")
            continue
        
        # Get wing edge metrics
        mean_top_y, median_top_y, _, _ = get_wing_edge_points(mask_color, configs.exclude_center_fraction)

        # Compute deflections (pixels)
        mean_deflection = ref_mean_top_y - mean_top_y if mean_top_y is not None and ref_mean_top_y is not None else None
        median_deflection = ref_median_top_y - median_top_y if median_top_y is not None and ref_median_top_y is not None else None

        # Convert to mm
        mean_deflection_mm = conversion(video_name, mean_deflection, configs)
        median_deflection_mm = conversion(video_name, median_deflection, configs)

        # Match speed for this frame
        frame_speed_data = speed_data[speed_data["Frame"] == frame_num]
        speed = frame_speed_data["Speed_kmh"].values[0] if not frame_speed_data.empty else float('nan')

        results.append({
            "frame": frame_num,
            "mask": mask_name,
            "mean_top_y": mean_top_y,
            "median_top_y": median_top_y,
            "mean_deflection": mean_deflection,
            "median_deflection": median_deflection,
            "speed_kmh": speed,
            "ref_top_y": ref_mean_top_y,
            "mean_deflection_mm": mean_deflection_mm,
            "median_deflection_mm": median_deflection_mm
        })
        
        #print(f"[{video_name}] Frame {frame_num} - MeanY: {mean_top_y} - MedianY: {median_top_y} - MeanDefl: {mean_deflection} - MedianDefl: {median_deflection} - Velocità: {speed} km/h")
    
    print(f"[INFO] Completed deflection computation for {video_name}")
    return results

