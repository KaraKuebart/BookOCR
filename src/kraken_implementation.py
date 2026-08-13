"""Run the Kraken OCR engine in parallel over the prepared page images."""

import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from src.import_paths import import_paths


def process_image(args):
    """Run the kraken CLI on a single image and write its PAGE XML."""
    img_file, output_dir = args
    output_file = output_dir / f"{img_file.stem}.xml"
    cmd = [
        "../.venv/bin/kraken",
        "-i",
        str(img_file),
        str(output_file),
        "-x",
        "segment",
        "-bl",
        "ocr",
        "-m",
        "tiny.safetensors",
    ]
    subprocess.run(cmd)


def run_kraken():
    """OCR all prepared page images with kraken, one process per CPU core."""
    input_folder, output_folder = import_paths()
    os.makedirs(f"{output_folder}/xmls_kraken", exist_ok=True)
    input_dir = Path(f"{input_folder}/ocr_ready")
    output_dir = Path(f"{output_folder}/xmls_kraken")
    output_dir.mkdir(exist_ok=True)

    # Get number of CPU cores (or set manually, e.g., num_cores=4)
    num_cores = os.cpu_count()

    # Create list of image files
    img_files = list(input_dir.glob("*"))

    # Process in parallel
    with Pool(processes=num_cores) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    process_image, [(f, output_dir) for f in img_files]
                ),
                total=len(img_files),
            )
        )
    print("\nOCR complete!")
