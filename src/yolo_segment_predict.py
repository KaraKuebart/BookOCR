import glob
import os
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import input_folder


def downscale_image(image, scale_factor: int):
    """Scale down an image by the given factor (default: half)."""
    scaled = cv2.resize(
        image, (image.shape[1] // scale_factor, image.shape[0] // scale_factor)
    )
    return scaled


def normalize_image(input_path, output_path, downscale_factor: int):
    """
    Use OpenCV to normalize an image, lighting too dark images and darkening too light ones, then save it.

    Args:
        input_path: Path to the input image file
        output_path: Path to save the processed image
    """
    # Load the image
    os.makedirs(f"{input_folder}/norm", exist_ok=True)
    img = Image.open(input_path).convert("RGB")
    img = np.array(img)
    img = downscale_image(img, downscale_factor)

    stretched = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Save the image
    img = Image.fromarray(stretched)
    img.save(output_path)


def predict_book(scale_factor):
    # Load the original image
    original_img = cv2.imread(img_path)
    original_height, original_width = original_img.shape[:2]

    # 1. Downscale the image before inference

    downscaled_width = original_width // scale_factor
    downscaled_height = original_height // scale_factor
    downscaled_img = cv2.resize(original_img, (downscaled_width, downscaled_height))

    # Run inference on downscaled image
    results = model.predict(downscaled_img, verbose=False)

    # Process results
    # Filter for book class and find the most confident detection
    book_detections = []
    for result in results:
        if result.masks is None:
            out = backup_detection(downscaled_img, book_detections)
            if out:
                return out
    if not book_detections:
        extract_books(book_detections, results)

    if not book_detections:
        out = backup_detection(downscaled_img, book_detections)
        if out:
            return out

    # Select the most confident book detection
    best_detection = max(book_detections, key=lambda x: x["conf"])

    # Get mask and upscale to original size
    mask_points = best_detection["mask"]
    mask = np.zeros(
        (original_height // scale_factor, original_width // scale_factor),
        dtype=np.uint8,
    )
    cv2.drawContours(mask, [mask_points.astype(np.int32)], 0, 255, -1)

    # Upscale mask to original image dimensions
    upscaled_mask = cv2.resize(
        mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR
    )

    # Apply binary thresholding
    _, binary_mask = cv2.threshold(upscaled_mask, 127, 255, cv2.THRESH_BINARY)

    # Create output filename
    conf_value = best_detection["conf"]
    class_name = best_detection["class_name"]
    output_path = f"{input_folder}/yolo/{img_path.split('/')[-1]}_{class_name}_{conf_value:.2f}.png"

    # Save as PNG with transparent background
    # Create 4-channel image (BGR + Alpha)
    bgra_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)
    bgra_img[:, :, 3] = binary_mask  # Set alpha channel to mask

    cv2.imwrite(str(output_path), bgra_img)
    # print(f"Saved: {output_path}")


def extract_books(
    book_detections: list[Any],
    results: Iterator[Results | Tensor] | list[Results] | list[Tensor],
):
    for result in results:
        for mask, box, cls, conf in zip(
            result.masks.xy, result.boxes.xyxy, result.boxes.cls, result.boxes.conf
        ):
            if int(cls) == 73:
                book_detections.append(
                    {
                        "mask": mask,
                        "box": box,
                        "cls": cls,
                        "conf": conf.item(),
                        "class_name": result.names[int(cls)],
                    }
                )


def backup_detection(downscaled_img=None, book_detections=None):
    results = model2.predict(downscaled_img, verbose=False)
    for result in results:
        if result.masks is None:
            print(f"No masks found in {img_path} on the SECOND attempt")
            return img_path
    extract_books(book_detections, results)
    if not book_detections:
        print(f"No books found in {img_path} on the SECOND attempt")
        return img_path


def run_yolo(scale_factor=2):
    global img_path, model, model2
    # make sure necessary paths for the whole pipeline exist
    os.makedirs(f"{input_folder}/yolo", exist_ok=True)
    os.makedirs(f"{input_folder}/ocr_ready", exist_ok=True)
    model = YOLO("yolo26n-seg.pt")
    model2 = YOLO("yolo26s-seg.pt")
    images = glob.glob(f"{input_folder}/norm/*.jpg")
    to_correct_manually = []
    for img_path in tqdm(images):
        error = predict_book(scale_factor)
        if error:
            to_correct_manually.append(img_path)
    if to_correct_manually:
        print(
            f"ATTENTION: NOT ALL PAGES COULD BE DETECTED: MODELS {model.model_name} and {model2.model_name} DID NOT FIND BOOK PAGES IN: {to_correct_manually}"
        )
    else:
        print(f"YOLO models recognized all book pages.")


def preprocess(downscale_factor=2):
    if not os.path.exists(f"{input_folder}/norm"):
        os.mkdir(f"{input_folder}/norm")
    images = glob.glob(f"{input_folder}/*.jpg")
    for pre_path in tqdm(images):
        normalize_image(
            pre_path,
            f"{input_folder}/norm/{pre_path.split('/')[-1]}",
            downscale_factor=downscale_factor,
        )
    print("preprocessing done")


if __name__ == "__main__":
    preprocess(downscale_factor=2)
    run_yolo()
