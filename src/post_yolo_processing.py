import glob
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import input_folder, output_folder


def optimize_and_fill_book_page(
    image_path: str,
    output_path: str,
    max_rotation: float = 15,
    rotation_step: float = 1,
) -> None:
    """
    Rotate a book page image to best fit a bounding box, crop transparent space,
    and fill remaining transparency with the mean background color.

    Args:
        image_path: Path to input PNG image with transparent background
        output_path: Path to save the processed image
        max_rotation: Maximum rotation angle in degrees (default 40)
        rotation_step: Step size for rotation search in degrees (default 0.5)
    """

    # Read image with alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    if img.shape[2] != 4:
        raise ValueError("Image must have an alpha channel (RGBA)")

    # Downscale for faster rotation search
    scale_factor = 0.2
    h, w = img.shape[:2]
    img_small = cv2.resize(
        img,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_AREA,
    )

    # Step 1: Find best rotation on downscaled image
    best_angle = _find_best_rotation(img_small, max_rotation, rotation_step)

    # Step 2: Rotate image
    rotated_img = _rotate_image(img, best_angle)

    # Step 3: Crop transparent borders
    cropped_img = _crop_transparent_borders(rotated_img)

    # Step 4: Calculate mean color and fill in one pass
    alpha = cropped_img[:, :, 3]
    mask = alpha > 0

    mean_color = tuple(int(np.mean(cropped_img[:, :, i][mask])) for i in range(3)) + (
        255,
    )

    filled_img = cropped_img.copy()
    filled_img[alpha == 0] = mean_color

    # Save result
    cv2.imwrite(output_path, filled_img)


def get_bbox_area(angle: float, alpha) -> float:
    """Calculate bounding box area for a given rotation angle."""
    rotated_alpha = _rotate_image(alpha, angle)
    _, thresh = cv2.threshold(rotated_alpha, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        return w * h
    return float("inf")


def _find_best_rotation(img: np.ndarray, max_rotation: float, step: float) -> float:
    """Find rotation angle that minimizes bounding box area using gradient descent."""

    alpha = img[:, :, 3]
    current_angle = 0.0
    current_area = get_bbox_area(current_angle, alpha)

    # Try positive direction first
    direction = step
    while abs(current_angle) < max_rotation:
        next_angle = current_angle + direction
        next_area = get_bbox_area(next_angle, alpha)

        if next_area < current_area:
            current_angle = next_angle
            current_area = next_area
        else:
            # If positive direction didn't work, try negative
            if direction > 0:
                direction = -step
                next_angle = current_angle + direction
                next_area = get_bbox_area(next_angle, alpha)

                if next_area < current_area:
                    current_angle = next_angle
                    current_area = next_area
                else:
                    break  # No improvement in either direction
            else:
                break  # Already tried both directions

    # print(f"Best rotation angle: {current_angle}°")
    return current_angle


def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle (in degrees) around its center."""

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Handle both grayscale and color images
    if len(img.shape) == 2:
        rotated = cv2.warpAffine(
            img, M, (new_w, new_h), borderValue=0, flags=cv2.INTER_LINEAR
        )
    else:
        rotated = cv2.warpAffine(
            img, M, (new_w, new_h), borderValue=(0, 0, 0, 0), flags=cv2.INTER_LINEAR
        )

    return rotated


def _crop_transparent_borders(img: np.ndarray) -> np.ndarray:
    """Crop transparent borders from image."""

    alpha = img[:, :, 3]

    # Find non-transparent pixels
    coords = cv2.findNonZero(alpha)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    # Crop the image
    cropped = img[y : y + h, x : x + w]

    return cropped


def _get_mean_background_color(img: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculate mean color of non-transparent pixels (BGR format for OpenCV)."""

    alpha = img[:, :, 3]
    mask = alpha > 0

    if not mask.any():
        return (255, 255, 255, 255)  # Default to white

    # Calculate mean for each channel
    mean_color = []
    for i in range(3):  # BGR channels
        channel_mean = int(np.mean(img[:, :, i][mask]))
        mean_color.append(channel_mean)

    mean_color.append(255)  # Alpha channel

    return tuple(mean_color)


def _fill_transparency(
    img: np.ndarray, fill_color: Tuple[int, int, int, int]
) -> np.ndarray:
    """Fill transparent areas with specified color."""

    alpha = img[:, :, 3]
    mask = alpha == 0

    # Create output image (convert to BGR if needed for final output)
    result = img.copy()

    # Fill transparent pixels
    result[mask] = fill_color

    return result


def run_post_yolo():
    images = glob.glob(f"{input_folder}/yolo/*.png")
    for image in tqdm(images):
        optimize_and_fill_book_page(
            image, f"{input_folder}/ocr_ready/{image.split('/')[-1].split('.')[0]}.png"
        )


if __name__ == "__main__":
    run_post_yolo()
