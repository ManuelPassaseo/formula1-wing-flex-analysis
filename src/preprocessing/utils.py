import os
import cv2
import shutil
import pandas as pd
import fastf1
import json
from datetime import timedelta
from PIL import Image, ImageDraw

def extract_frames(video_path: str, output_dir: str, frames_per_second_to_extract: int = 30, max_frames: int = 5400) -> bool:
    """
    Extract frames from a video file and save them as individual images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        frames_per_second_to_extract (int, optional): Target number of frames per second to extract. Defaults to 30.
        max_frames (int, optional): Maximum number of frames to save. Defaults to 5400 (Video duration capped to 3 minutes).

    Returns:
        bool: True if extraction completes successfully.
    """

    # If output_dir exists and is not empty, clean it
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"[INFO] Cleaning existing directory: '{output_dir}'")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file '{video_path}'")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)  # Original video frames per second
    if fps <= 0:
        print("[ERROR] Unable to retrieve FPS from video.")
        return False

    # Determine how many frames to skip between saves
    frame_interval = int(round(fps / frames_per_second_to_extract))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    print(f"[INFO] Extraction completed: {saved_count} frames saved to '{output_dir}'")
    return True


def speed_detection(video_path: str, start_time_hms: timedelta, configs, fps: int = 30) -> None:
    """
    Overlay car speed on pre-extracted video frames and rebuild annotated video.

    Args:
        video_path (str): Video file path.
        start_time_hms (timedelta): Start time offset of video within race session.
        configs: Configuration object with speed output directory.
        fps (int, optional): Frame rate used for the original extraction. Defaults to 30.
    """
    # Parse video metadata from video path
    video_name = os.path.basename(video_path).split('.')[0]
    _, driver, race, year = video_name.split('_')

    os.makedirs(configs.speed_dir, exist_ok=True)
    csv_path = os.path.join(configs.speed_dir, f"{video_name}.csv")
    output_video_path = os.path.join(configs.speed_dir, f"{video_name}_speed.mp4")

    # Initialize FastF1 session
    os.makedirs('cache', exist_ok=True)
    fastf1.Cache.enable_cache('cache')  # Create a local cache directory
    session = fastf1.get_session(int(year), race, 'R')
    session.load()

    laps = session.laps.pick_driver(driver[:3].upper())
    telemetry_data = pd.DataFrame()

    for i, (_, lap) in enumerate(laps.iterrows()):
        if i == 0:
            start_time_hms += lap['LapStartTime'].to_pytimedelta()

        tel = lap.get_telemetry()
        tel['SessionTime'] = tel['Time'] + lap['LapStartTime']
        tel['SessionTimeSeconds'] = tel['SessionTime'].dt.total_seconds()
        telemetry_data = pd.concat([telemetry_data, tel], ignore_index=True)

    # Compute absolute session time for start
    start_session_time = start_time_hms.total_seconds()

    # Get frame list and properties
    frame_files = sorted([f for f in os.listdir(configs.frames_dir) if f.endswith(".jpg")])
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {configs.frames_dir}")
    
    first_frame = cv2.imread(os.path.join(configs.frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop over frames
    rows = []
    for frame_number, frame_file in enumerate(frame_files):
        frame_path = os.path.join(configs.frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        # Compute absolute session time for this frame
        time_in_video = frame_number / fps
        time_in_session = start_session_time + time_in_video

        # Find closest telemetry speed
        idx = telemetry_data['SessionTimeSeconds'].sub(time_in_session).abs().idxmin()
        speed = telemetry_data.iloc[idx]['Speed']

        rows.append({"Frame": frame_number, "Speed_kmh": speed})

        # Write speed over frame
        text = f"{speed:.0f} km/h"
        cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(frame)
        
    out.release()

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"[INFO] CSV saved to {csv_path}")

    # Cleanup cache
    shutil.rmtree('cache', ignore_errors=True)


def labelme_json_to_mask(json_path: str, configs) -> str:
    """
    Convert a LabelMe JSON annotation file into a segmentation mask image.

    Args:
        json_path (str): Path to the LabelMe JSON file.
        configs: Configuration object with ground truth mask output directory.

    The function creates a color mask where each annotated object has a distinct RGB color.
    Background is set to light gray.
    """
    print(f"[INFO] Processing annotation: {os.path.basename(json_path)}")

    # Load JSON annotation
    with open(json_path, "r") as f:
        data = json.load(f)

    image_height = data.get("imageHeight")
    image_width = data.get("imageWidth")

    if image_height is None or image_width is None:
        raise ValueError(f"[ERROR] Missing image dimensions in {json_path}")
    
    # Initialize label-color mapping
    label_to_color = {}
    current_index = 1  # 0 = background (unused color)

    # Create blank mask (light gray background)
    mask = Image.new("RGB", (image_width, image_height), (192, 192, 192))
    draw = ImageDraw.Draw(mask)

    # Draw polygons for each annotated shape
    for shape in data.get("shapes", []):
        label = shape.get("label", "undefined")
        points = shape.get("points", [])

        if not points:
            print(f"[WARNING] Skipping empty shape for label '{label}'")
            continue

        # Assign color per label (same color for simplicity)
        if label not in label_to_color:
            label_to_color[label] = (95, 158, 160)  # cadet blue
            current_index += 1

        polygon = [(x, y) for x, y in points]
        draw.polygon(polygon, fill=label_to_color[label])

    # Ensure output directory exists
    gt_mask_path = os.path.join(configs.output_dir, "gt_mask.png")
    
    # --- Save mask image ---
    mask.save(gt_mask_path)
    print(f"[INFO] Mask saved: {gt_mask_path}")
    return gt_mask_path