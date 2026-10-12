import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import gc
import cv2
import torch


def save_mask(mask: np.ndarray, frame_id: str, configs) -> None:
    """
    Save a raw segmentation mask as an RGBA image (or grayscale if 2D).

    Args:
        mask: (H, W) binary or float mask (0–1 range) or (H, W, 4) RGBA mask.
        frame_id: Frame name or index.
        configs: Configuration object with mask output directory.
    """
    os.makedirs(configs.masks_dir, exist_ok=True)
    output_path = os.path.join(configs.masks_dir, f"{frame_id}_mask.png")

    # Handle 1-channel mask
    if mask.ndim == 2 or (mask.ndim == 3 and mask.shape[0] == 1):
        mask = np.squeeze(mask)
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.bitwise_not(mask)
        cv2.imwrite(output_path, mask)
        return

    # Handle RGBA-style mask
    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(mask, cv2.COLOR_RGBA2BGRA))
        return

    raise ValueError(f"Unexpected mask shape {mask.shape} in save_mask()")


def show_mask(image: np.ndarray, mask: np.ndarray, obj_id: int = 0, alpha: float = 0.6) -> np.ndarray:
    """
    Overlay a semi-transparent mask on a BGR image.

    Args:
        image_bgr: Original frame (H, W, 3).
        mask: Binary mask (H, W) or (1, H, W).
        obj_id: Object ID for color selection.
        alpha: Mask transparency.
    """
    if mask.ndim == 3:
        mask = mask.squeeze()

    cmap = plt.get_cmap("tab10")
    color = np.array(cmap(obj_id % 10)[:3]) * 255
    color = color.astype(np.uint8)

    # Apply color overlay
    overlay = image.copy()
    overlay[mask > 0.5] = color

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def show_points(image: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Draw positive (green) and negative (red) points on the image."""
    for (x, y), label in zip(coords, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(image, (int(x), int(y)), radius=8, color=color, thickness=-1)
        cv2.circle(image, (int(x), int(y)), radius=10, color=(255, 255, 255), thickness=2)
    return image

def show_box(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Draw a bounding box on the image."""
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return image

def show_masks(
    image: Image.Image,
    mask: np.ndarray,
    configs,
    obj_id: int = 0,
    frame_id: str = None,
    point_coords: np.ndarray = None,
    box_coords: np.ndarray = None,
    input_labels: np.ndarray = None,
) -> None:
    """Display and save image with overlayed mask, points, and boxes."""
    os.makedirs(configs.ann_frames_dir, exist_ok=True)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply mask
    annotated = show_mask(image, mask, obj_id=obj_id)

    # Save the standalone mask too
    if frame_id is not None:
        save_mask(mask, frame_id, configs)

    # Add points and box if provided
    if point_coords is not None and input_labels is not None:
        annotated = show_points(annotated, point_coords, input_labels)
    if box_coords is not None:
        annotated = show_box(annotated, box_coords)

    # Save annotated frame
    if frame_id is not None:
        output_path = os.path.join(configs.ann_frames_dir, f"{frame_id}_box.png")
    else:
        output_path = os.path.join(configs.output_dir, "frame_marks.png")

    cv2.imwrite(output_path, annotated)
    gc.collect()

def efficienttam_inference(predictor, configs) -> None:
    # scan all the JPEG frame names in this directory
    print("[INFO] Scanning frames directory...")
    frame_names = [
        p
        for p in os.listdir(configs.frames_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    print(f"[INFO] Found {len(frame_names)} frames in '{configs.frames_dir}'")

    print("[INFO] Initializing video inference state...")
    inference_state = predictor.init_state(video_path=configs.frames_dir)
    prompts = {}  # hold all the clicks we add for visualization

    ann_frame_idx = 0  # the frame index we interact with (first frame of the video)
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    input_box = np.array([425, 575, 1525, 1000], dtype=np.float32)
    input_points = np.array([[975, 725], [750, 950], [1200, 950]], dtype=np.float32)

    # for labels, `1` means positive click and `0` means negative click
    input_labels = np.array([1, 0, 0], np.int32)

    prompts[ann_obj_id] = input_points, input_labels,
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=input_points,
        labels=input_labels,
        box=input_box,
    )
    
    # show the results on the current (interacted) frame
    image_path = os.path.join(configs.frames_dir, frame_names[ann_frame_idx])
    with Image.open(image_path) as image:
        for i, out_obj_id in enumerate(out_obj_ids):
            show_masks(image, (out_mask_logits[i] > 0.0).cpu().numpy(), configs, obj_id=out_obj_id, point_coords=input_points, box_coords=input_box, input_labels=input_labels)
    print(f"[INFO] Initial frame ({frame_names[ann_frame_idx]}) processed and saved.")

    # run propagation throughout the video and collect the results in a dict
    print("[INFO] Propagating segmentation across video frames...")
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 1
    os.makedirs(configs.ann_frames_dir, exist_ok=True)
    
    print("[INFO] Saving annotated frames...")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        image_path = os.path.join(configs.frames_dir, frame_names[out_frame_idx])
        with Image.open(image_path) as image:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_masks(image, out_mask, configs, obj_id=out_obj_id, frame_id=frame_names[out_frame_idx].split('.')[0])
        
        if out_frame_idx % 10 == 0:
            print(f"Processed frame {out_frame_idx + 1}/{len(frame_names)}")
            torch.cuda.empty_cache()
            gc.collect()

    print(f"[INFO] Inference completed. Results saved to: {configs.ann_frames_dir}")