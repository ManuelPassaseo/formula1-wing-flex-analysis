import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from deflection.wing_analysis import conversion, get_wing_edge_points

def reference_image(masks_dir: str, speed_data: pd.DataFrame, speed_file_path: str):
    """
    Selects the reference image corresponding to the minimum speed frame.

    Args:
        masks_dir (str): Directory containing mask images.
        speed_data (pd.DataFrame): CSV data containing "Frame" and "Speed_kmh" columns.
        speed_file_path (str): Path to the CSV file for error reference.

    Returns:
        tuple: (reference_image, frame_minimum_speed) or (None, None) if error.
    """
        
    required_columns = {"Frame", "Speed_kmh"}
    if not required_columns.issubset(speed_data.columns):
        print(f"[ERROR] Missing columns {required_columns - set(speed_data.columns)} in {speed_file_path}")
        return None, None
    
    mask_files = natsorted(glob(os.path.join(masks_dir, '*.png')))
    if not mask_files:
        print(f"[ERROR] No mask files found in {masks_dir}")
        return None, None
    
    # Extract frame numbers
    available_frames = []
    for mask_path in mask_files:
        mask_name = os.path.basename(mask_path)
        try:
            frame_num = int(mask_name.split('_')[0])
            available_frames.append(frame_num)
        except ValueError:
            print(f"[WARNING] Could not extract frame number from {mask_path}")
            continue
    
    # Check frame numbers in CSV and overlay
    available_speed_data = speed_data[speed_data['Frame'].isin(available_frames)]
    if available_speed_data.empty:
        print("[ERROR] No matching speed data for overlay frames.")
        return None, None
    
    # Find minimum speed frame
    idx_min = available_speed_data["Speed_kmh"].idxmin()
    frame_min = int(available_speed_data.loc[idx_min, "Frame"])
    min_speed = float(available_speed_data.loc[idx_min, "Speed_kmh"])
    frame_str = f"{frame_min:06d}"

    # Use the frame with minimum speed as reference
    ref_matches = glob(os.path.join(masks_dir, f"{frame_str}*.png"))
    if not ref_matches:
        print(f"[WARNING] No reference image found for frame {frame_str} in {masks_dir}")
        return None, None
    
    ref_image_path = ref_matches[0]  # first match
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        print(f"[ERROR] Could not read reference image {ref_image_path}")
        return None, None
    
    print(f"[INFO] Using reference: {os.path.basename(ref_image_path)} (speed: {min_speed:.2f} km/h)")
    return ref_image, frame_min


def create_analysis_video(video_name: str, masks_dir: str, configs, method: str, ref_top_y: float, speed_data: pd.DataFrame):
    """
    Creates a visual analysis video showing wing deflection progression.

    Args:
        video_name (str): Base video name.
        masks_dir (str): Directory containing mask images.
        configs: Configuration object (must contain deflection_dir, exclude_center_fraction).
        method (str): Either "mean" or "median".
        ref_top_y (float): Reference Y coordinate.
        speed_data (pd.DataFrame): Speed data with "Frame" and "Speed_kmh".
    """
    os.makedirs(configs.deflection_dir, exist_ok=True)
    output_video_path = os.path.join(configs.deflection_dir, f"{video_name}_wing_analysis_{method}.mp4")
    
    # Get all mask files
    mask_files = natsorted(glob(os.path.join(masks_dir, '*.png')))
    if not mask_files:
        print(f"[ERROR] No mask files found in {masks_dir}")
        return
    
    # Get frame size from first image
    first_frame = cv2.imread(mask_files[0])
    if first_frame is None:
        print("[ERROR] Could not read first frame for video initialization.")
        return
        
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
    
    print(f"[INFO] Creating analysis video ({method}) for {video_name}...")
    
    for i, mask_path in enumerate(mask_files):
        if i % 10 == 0:
            print(f"[INFO] Processing frame {i+1}/{len(mask_files)}")
            
        mask_name = os.path.basename(mask_path)
        mask_color = cv2.imread(mask_path)
        if mask_color is None:
            print(f"[WARNING] Could not read {mask_path}")
            continue
            
        try:
            frame_num = int(mask_name.split('_')[0])
        except ValueError:
            print(f"[WARNING] Invalid frame name format: {mask_name}")
            continue
        
        # Speed for frame
        frame_speed_data = speed_data[speed_data["Frame"] == frame_num]
        speed = frame_speed_data["Speed_kmh"].values[0] if not frame_speed_data.empty else float('nan')
        
        # Get top points
        mean_top_y, median_top_y, left_point, right_point = get_wing_edge_points(mask_color, configs.exclude_center_fraction)
        
        # Choose the appropriate y value based on method
        top_y = mean_top_y if method == "mean" else median_top_y
        
        # Calculate deflection
        deflection = ref_top_y - top_y if top_y is not None else None
        deflection_mm = conversion(video_name, deflection, configs)
        
        # Create a copy of the image for drawing
        display_img = mask_color.copy()
        
        # Draw center exclusion zone
        center_left = int(width * (0.5 - configs.exclude_center_fraction / 2))
        center_right = int(width * (0.5 + configs.exclude_center_fraction / 2))
        cv2.rectangle(display_img, (center_left, 0), (center_right, height), (0, 0, 255), 1)
        
        # Draw top points if available
        if left_point is not None:
            cv2.circle(display_img, left_point, 5, (255, 0, 0), -1)
        if right_point is not None:
            cv2.circle(display_img, right_point, 5, (255, 0, 0), -1)
        
        # Draw the approximation line
        if top_y is not None:
            y_pos = int(top_y)
            cv2.line(display_img, (0, y_pos), (width, y_pos), (128, 128, 128), 2)
            
            # Add text showing the y position
            cv2.putText(display_img, f"{method.capitalize()} Y: {y_pos}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add label above the wing line (on the left side)
            cv2.putText(display_img, "Wing Line Approx", (10, y_pos-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Add reference line
        if ref_top_y is not None:
            ref_y_pos = int(ref_top_y)
            cv2.line(display_img, (0, ref_y_pos), (width, ref_y_pos), (255, 255, 255), 2)
            cv2.putText(display_img, f"Ref Y: {ref_y_pos}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add label above the reference line (on the left side)
            cv2.putText(display_img, "Reference Line", (width - 250, ref_y_pos-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add deflection info
        if deflection is not None:
            cv2.putText(display_img, f"Deflection: {deflection:.1f} px", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if deflection_mm is not None:
            cv2.putText(display_img, f"{method.capitalize()}(mm): {deflection_mm:.1f} mm", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add speed info
        cv2.putText(display_img, f"Speed: {speed:.1f} km/h", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(display_img, f"Frame: {frame_num}", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add method info
        cv2.putText(display_img, f"Method: {method}", (width - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(display_img)
    
    out.release()
    print(f"[INFO] Analysis video ({method}) saved to: {output_video_path}")


def create_time_series_plot(video_name: str, results: list, results_dir: str, window_size: int = 15):
    """
    Plots deflection (mm) and speed over time (frames).

    Args:
        video_name (str): Base video name.
        results (list[dict]): Results with deflection and speed data.
        results_dir (str): Output directory for saving the plot.
        window_size (int): Smoothing window size for moving average.
    """
    # Extract data for plotting
    frames = [r["frame"] for r in results if r["mean_deflection_mm"] is not None]
    mean_deflections = [r["mean_deflection_mm"] for r in results if r["mean_deflection_mm"] is not None]
    speeds = [r["speed_kmh"] for r in results if r["mean_deflection_mm"] is not None]
    
    if not frames:
        print("[WARNING] No valid data to plot time series.")
        return
    
    # Smooth flexion using moving average
    smoothed_deflections = np.convolve(
        mean_deflections, np.ones(window_size)/window_size, mode='valid'
    )
    smoothed_frames = frames[window_size-1:]
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot smoothed flexion
    color = 'tab:blue'
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Flexion [mm]", color=color)
    ax1.plot(smoothed_frames, smoothed_deflections, color=color, label=f"Flexion (smoothed, {window_size} frames)")
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot speed on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("Speed [km/h]", color=color)
    ax2.plot(frames, speeds, color=color, label="Speed")
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and grid
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{video_name}_time_series_plot.png"))
    plt.close()
    print(f"[INFO] Time-series plot saved to {results_dir}")


def analyze_deflection_speed(df: pd.DataFrame, x_col: str = "speed_kmh", y_col: str = "median_deflection_mm", save_dir: str = None, plot_title: str = ""):
    """
    Performs a quadratic regression analysis of deflection vs. speed.

    Args:
        df (pd.DataFrame): Data containing deflection and speed.
        x_col (str): Column name for speed.
        y_col (str): Column name for deflection.
        save_dir (str): Directory to save the output plot.
        plot_title (str): Plot title.
    """
    valid_data = df[[x_col, y_col]].dropna()
    if valid_data.empty:
        print("[WARNING] No valid data for deflection-speed analysis.")
        return
    
    x, y = valid_data[x_col].values, valid_data[y_col].values
    r, _ = pearsonr(x, y)
    
    x_fit = x**2
    coeffs = np.polyfit(x_fit, y, deg=1)
    y_pred = coeffs[0] * x_fit + coeffs[1]
    
    x_sorted = np.linspace(x.min(), x.max(), 300)
    fit_curve = coeffs[0] * x_sorted**2 + coeffs[1]
    equation = f"y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}"
            
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, color="blue", alpha=0.7)
    plt.plot(x_sorted, fit_curve, 'r-', linewidth=2, 
             label=f'QUADRATIC fit (R²={r2:.3f})')
    
    plt.xlabel(x_col)
    plt.ylabel("deflection_mm")
    plt.title(f"{plot_title}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = os.path.join(save_dir, f"{y_col}_vs_{x_col}_quadratic.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Quadratic deflection-speed analysis saved to: {save_path}")