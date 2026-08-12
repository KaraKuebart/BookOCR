"""Run the Pero OCR engine on the prepared page images."""

import configparser
import os
from pathlib import Path

import cv2
import numpy as np
from pero_ocr.core.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser
from tqdm import tqdm

import config as config

input_folder = os.path.expanduser(config.input_folder)
output_folder = os.path.expanduser(config.output_folder)


def run_pero(
    input_folder: str = f"{input_folder}/ocr_ready",
    output_folder: str = f"{output_folder}/xmls_pero",
    config_path: str = "./pero_eu_cz_print_newspapers_2022-09-26/config_cpu.ini",
    save_pagexml: bool = True,
    save_altoxml: bool = False,
    save_rendered: bool = False,
    save_cropped_lines: bool = False,
) -> None:
    """Process all images in a folder using Pero OCR.

    Args:
        input_folder: Path to folder containing image files
        output_folder: Path to folder where results will be saved
        config_path: Path to the OCR engine config file (.ini)
        save_pagexml: Whether to save results as Page XML (default: True)
        save_altoxml: Whether to save results as ALTO XML (default: False)
        save_rendered: Whether to save images with the detected text regions
            rendered into them (default: False)
        save_cropped_lines: Whether to save individual cropped text lines
            (default: False)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read config file
    config_pero = configparser.ConfigParser()
    config_pero.read(config_path)

    # Init the OCR pipeline
    page_parser = PageParser(config_pero, config_path=os.path.dirname(config_path))

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Get all image files from the input folder
    image_files = [
        f
        for f in os.listdir(input_folder)
        if Path(f).suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for image_file in tqdm(image_files):
        input_image_path = os.path.join(input_folder, image_file)
        file_base_name = Path(image_file).stem

        try:
            # Read the document page image
            image = cv2.imread(input_image_path, 1)
            if image is None:
                print("  ✗ Failed to read image")
                continue

            # Init empty page content
            page_layout = PageLayout(
                id=input_image_path, page_size=(image.shape[0], image.shape[1])
            )

            # Process the image by the OCR pipeline
            page_layout = page_parser.process_page(image, page_layout)

            # Save results as Page XML
            if save_pagexml:
                pagexml_output_path = os.path.join(
                    output_folder, f"{file_base_name}_page.xml"
                )
                page_layout.to_pagexml(pagexml_output_path)

            # Save results as ALTO XML
            if save_altoxml:
                altoxml_output_path = os.path.join(
                    output_folder, f"{file_base_name}_ALTO.xml"
                )
                page_layout.to_altoxml(altoxml_output_path)

            # Render detected text regions and text lines into the image
            if save_rendered:
                rendered_image = page_layout.render_to_image(image)
                rendered_output_path = os.path.join(
                    output_folder, f"{file_base_name}_render.jpg"
                )
                cv2.imwrite(rendered_output_path, rendered_image)

            # Save each cropped text line in a separate .jpg file
            if save_cropped_lines:
                for region in page_layout.regions:
                    for line in region.lines:
                        line_output_path = os.path.join(
                            output_folder, f"{file_base_name}-{line.id}.jpg"
                        )
                        cv2.imwrite(line_output_path, line.crop.astype(np.uint8))

        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {str(e)}")
            continue

    print("\nOCR complete!")


# Example usage
if __name__ == "__main__":
    run_pero()
