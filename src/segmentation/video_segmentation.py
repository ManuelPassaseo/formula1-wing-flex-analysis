# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb
# Example of Efficient Track Anything Image with Point/Box Prompts

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from efficient_track_anything.build_efficienttam import build_efficienttam

from efficient_track_anything.efficienttam_image_predictor import (
    EfficientTAMImagePredictor,
)
from PIL import Image, ImageEnhance

np.random.seed(5)
dpi = 80

# Video path
filepath = "./videos/Ferrari_Hamilton_Suzuka_2025.mp4"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability()
else:
    device = torch.device("cpu")
    capability = None
print(f"Device: {device}, capability: {capability}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Model
checkpoint = "../checkpoints/efficienttam_s.pt"
model_cfg = "configs/efficienttam/efficienttam_s.yaml"

efficienttam = build_efficienttam(model_cfg, checkpoint, device=device)

predictor = EfficientTAMImagePredictor(efficienttam, max_hole_area=2000, max_sprinkle_area=2000)

# Point prompts
def show_points(coords, labels, ax, marker_size=2000):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.5,
    )

# box prompt
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

def show_mask(mask, ax, basename, random_color=False, borders=True):
    fig = plt.figure(figsize=figsize)
    a = fig.add_axes([0, 0, 1, 1])
    a.axis("off")
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )    
    a.imshow(mask_image)
    fig.savefig(f"./masks/{basename}_mask.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    ax.imshow(mask_image)

def show_masks(
    image,
    basename,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(image)
        show_mask(mask, ax, basename, borders=borders)
        if point_coords is not None:
            # points
            assert input_labels is not None
            show_points(point_coords, input_labels, ax)
        if box_coords is not None:
            # boxes
            show_box(box_coords, ax)
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        if box_coords is not None:
            fig.savefig(f"./images/{basename}_box.png", bbox_inches="tight", pad_inches=0)
        else:
            fig.savefig(f"./images/{basename}_point_{i}.png", bbox_inches="tight", pad_inches=0)
        # Close current figure
        plt.close(fig)

def extract_frames(video_path, output_dir, fps_interval=1):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    frame_interval = int(fps * fps_interval)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

output_dir = f"./videos/{os.path.basename(filepath).split('.')[0]}"
extract_frames(filepath, output_dir)
for frame in os.listdir(output_dir):
    if os.path.splitext(frame)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        image = Image.open(os.path.join(output_dir, frame))
        basename = "".join([os.path.basename(filepath).split('.')[0], "_", os.path.splitext(frame)[0]])
        width, height = image.size
        figsize = width / float(dpi), height / float(dpi)

        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.8)  # >1 = higher contrast
        image = np.array(image.convert("RGB"))
        
        predictor.set_image(image)

        #[400, 600, 1500, 1000]
        input_box = np.array([425, 575, 1525, 1000])
        #[[950, 700], [725, 925], [1175, 925]]
        input_point = np.array([[975, 725], [750, 950], [1200, 950]])
        input_label = np.array([1, 0, 0])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            multimask_output=False,
        )

        # show mask for box prompt segmentation and points 
        show_masks(image, basename, masks, scores, point_coords=input_point, input_labels=input_label, box_coords=input_box)