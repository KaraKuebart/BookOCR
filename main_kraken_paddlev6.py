import argparse
import os
from src.yolo_segment_predict import run_yolo, preprocess
from src.post_yolo_processing import run_post_yolo
from src.kraken_implementation import run_kraken
from src.pdf_export import run_export
import datetime
import config as config
os.chdir("./src")

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
            f.write("""config file containing arguments parsed via argparse.""")
            f.write(f'input_folder = "{config.input_folder}"\n')
            f.write(f'output_folder = "{config.output_folder}"\n')

    a = datetime.datetime.now()
    print(a, "Beginning OCR using Kraken with PaddleV6 tiny: https://zenodo.org/records/21788403")
    preprocess(downscale_factor=2)
    run_yolo(scale_factor=2)  # the higher the scale factor, the lower the detection resolution. Keep at 1 for low-res images, choose higher values for higher resolutions.
    run_post_yolo()
    run_kraken()  # to run, the following model is needed: tiny.safetensors. It can be downloaded here: https://zenodo.org/records/21788403
    run_export("kraken")
    b = datetime.datetime.now()
    print(b, f"FINISHED. It took {b - a} seconds")
