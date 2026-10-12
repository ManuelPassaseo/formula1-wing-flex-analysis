import os
import cv2
import numpy as np
from skimage.segmentation import active_contour

def snake_segmentation(configs) -> None:
    """ Perform active contour (snake) segmentation on a single image and save the result. """
    
    print(f"[INFO] Snake segmentation on: {os.path.basename(configs.output_dir)}")

    # Get frame list and properties
    frame_files = sorted([f for f in os.listdir(configs.frames_dir) if f.endswith(".jpg")])
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {configs.frames_dir}")

    # Loop over frames
    for frame_file in frame_files:
        frame_path = os.path.join(configs.frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[WARNING] Unable to load image: {frame_path}")
            continue

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        contrast_img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)

        rows, cols = gray.shape

        # Initial contour (ellipse)
        dy = 100
        s = np.linspace(0, 2 * np.pi, 300)
        rx = min(rows, cols) * 0.45
        ry = min(rows, cols) * 0.13
        x = cols / 2 + rx * np.cos(s)
        y = rows / 2 + dy + ry * np.sin(s)
        init = np.array([y, x]).T

        # --- Active contour ---
        snake = active_contour(
            gray, init,
            alpha=0.037, beta=0.16, gamma=0.005,
            w_line=0.15, w_edge=0.95
        )

        # --- Flatten top section ---
        snake_fixed = np.copy(snake)
        y_threshold = np.percentile(snake[:, 0], 22.5)
        snake_fixed[snake_fixed[:, 0] <= y_threshold, 0] = y_threshold

        # Draw contours with OpenCV
        image_draw = frame.copy()

        # Draw initial contour (yellow dashed)
        for i in range(len(init) - 1):
            if i % 2 == 0:  # simple dashed effect
                pt1 = (int(init[i, 1]), int(init[i, 0]))
                pt2 = (int(init[i + 1, 1]), int(init[i + 1, 0]))
                cv2.line(image_draw, pt1, pt2, (0, 255, 255), 1)  # BGR yellow

        # Draw final contour (blue solid)
        for i in range(len(snake_fixed) - 1):
            pt1 = (int(snake_fixed[i, 1]), int(snake_fixed[i, 0]))
            pt2 = (int(snake_fixed[i + 1, 1]), int(snake_fixed[i + 1, 0]))
            cv2.line(image_draw, pt1, pt2, (255, 0, 0), 2)  # BGR blue
            
        os.makedirs(configs.snake_dir, exist_ok=True)
        output_path = os.path.join(configs.snake_dir, f"{frame_file.split('.')[0]}_snake.png")
        cv2.imwrite(output_path, image_draw)

    print(f"[INFO] Saved snake segmentation to: {configs.snake_dir}")