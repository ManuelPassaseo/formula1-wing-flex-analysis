import os
import csv
import pandas as pd
from deflection.video_processing import *
from deflection.wing_analysis import computation_on_frames

def create_summary_csv(video_results_dir: str, video_name: str, results) -> None:
    output_csv = os.path.join(video_results_dir, f"{video_name}_deflection_results.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "mask", "mean_top_y", "median_top_y", 
            "mean_deflection", "median_deflection", "speed_kmh", 
            "ref_top_y", "mean_deflection_mm", "median_deflection_mm"])
        writer.writeheader()
        writer.writerows(results)

def process_video(video_name: str, masks_dir: str, configs) -> None:
    """
    Process a single video through the deflection analysis pipeline.

    The function performs:
        1. Validation and loading of the speed telemetry data.
        2. Identification of the reference frame (lowest speed).
        3. Extraction of the wing edge from the reference image.
        4. Computation of deflection metrics across all frames.
        5. Generation of summary CSV, plots, and annotated analysis videos.

    Args:
        video_name (str): Name of the video (without file extension).
        masks_dir (str): Directory containing segmentation masks for each frame.
        configs: Configuration object containing project paths and parameters.

    Returns:
        None
    """
    # 1. Validate speed telemetry file
    speed_file_path = os.path.join(configs.speed_dir, f"{video_name}.csv")
    if not os.path.exists(speed_file_path):
        print(f"[ERROR] Missing speed file: {speed_file_path}")
        return
    
    print(f"[INFO] Loading speed data from: {speed_file_path}")
    speed_data = pd.read_csv(speed_file_path)
    # Ensure output directory exists
    os.makedirs(configs.deflection_dir, exist_ok=True)

    # 2. Reference image selection
    ref_image, minimum_speed_frame = reference_image(masks_dir, speed_data, speed_file_path)
    if ref_image is None or minimum_speed_frame is None:
        print("[ERROR] Failed to determine reference image.")
        return
    
    print(f"[INFO] Reference frame identified: {minimum_speed_frame}")

    # 3. Extract reference wing edge statistics
    print("[INFO] Computing wing edge metrics for reference frame...")
    ref_mean_top_y, ref_median_top_y, _, _ = get_wing_edge_points(ref_image, configs.exclude_center_fraction)
    if ref_mean_top_y is None or ref_median_top_y is None:
        print("[ERROR] Could not compute mean/median top edge for reference frame.")
        return
    
    print(f"[INFO] Reference mean top Y: {ref_mean_top_y:.2f}, median top Y: {ref_median_top_y:.2f}")

    # 4. Compute deflection metrics across frames
    print("[INFO] Computing deflection metrics across all frames...")
    results = computation_on_frames(masks_dir, video_name, ref_mean_top_y, ref_median_top_y, speed_data, configs)
    if results is None:
        print("[ERROR] Deflection computation failed.")
        return
    
    # 5. Export summary CSV and visualizations
    print("[INFO] Generating summary CSV and visualizations...")
    create_summary_csv(configs.deflection_dir, video_name, results)
    create_analysis_video(video_name, masks_dir, configs, "mean", ref_mean_top_y, speed_data)
    create_analysis_video(video_name, masks_dir, configs, "median", ref_median_top_y, speed_data)
    create_time_series_plot(video_name, results, configs.deflection_dir)
    
    # 6. Analyze deflection vs speed correlation
    df_results = pd.DataFrame(results)
    
    # mean deflection (mm) vs speed (km/h)
    analyze_deflection_speed(
        df=df_results,
        x_col="speed_kmh",
        y_col="mean_deflection_mm",
        save_dir=configs.deflection_dir,
        plot_title=f"{video_name} - Mean Deflection (mm) vs Speed",
    )
    
    # median deflection (mm) vs speed (km/h)
    analyze_deflection_speed(
        df=df_results,
        x_col="speed_kmh",
        y_col="median_deflection_mm",
        save_dir=configs.deflection_dir,
        plot_title=f"{video_name} - Median Deflection (mm) vs Speed",
    )
