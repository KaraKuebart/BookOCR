import argparse
from src.yolo_segment_predict import run_yolo, preprocess
from src.post_yolo_processing import run_post_yolo
from src.pero_implementation import run_pero
from src.pdf_export import run_export
from datetime import datetime
import config as config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR pipeline")
    parser.add_argument("--input", type=str, help="Input folder path")
    parser.add_argument("--output", type=str, help="Output folder path")
    args = parser.parse_args()

    # Save to user_config.py if arguments were given
    if args.input or args.output:
        if args.input:
            config.input_folder = args.input
        if args.output:
            config.output_folder = args.output
        # write user config to temporary config file
        with open("user_config.py", "w") as f:
            f.write(f'input_folder = "{config.input_folder}"\n')
            f.write(f'output_folder = "{config.output_folder}"\n')

    a = datetime.now()
    print(a, "Beginning OCR with Pero OCR. Code: https://github.com/DCGM/pero-ocr Model: https://nextcloud.fit.vutbr.cz/s/NtAbHTNkZFpapdJ?opendetails=")
    preprocess(downscale_factor=2)
    run_yolo(scale_factor=2)  # the higher the scale factor, the lower the page detection resolution. Keep at 1 for low-res images, choose higher values for higher resolutions.
    run_post_yolo()
    run_pero(config_path="./pero_eu_cz_print_newspapers_2022-09-26/config_cpu.ini")  # or config.ini, if a GPU can be used.
    # to work, the pero model repository has to be downloaded from: https://nextcloud.fit.vutbr.cz/s/NtAbHTNkZFpapdJ?opendetails= and unpacked in the "src" folder
    run_export("pero_local")
    b = datetime.now()
    print(b, f"FINISHED. It took {b - a} seconds")
