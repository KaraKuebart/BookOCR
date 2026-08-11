from src.yolo_segment_predict import run_yolo, preprocess
from src.post_yolo_processing import run_post_yolo
from src.kraken_implementation import run_kraken
from src.pdf_export import run_export
import datetime

if __name__ == "__main__":
    a = datetime.datetime.now()
    print(a, "Beginning OCR using Kraken with PaddleV6 tiny: https://zenodo.org/records/21788403")
    preprocess(downscale_factor=2)
    run_yolo(scale_factor=2)  # the higher the scale factor, the lower the detection resolution. Keep at 1 for low-res images, choose higher values for higher resolutions.
    run_post_yolo()
    run_kraken()  # to run, the following model is needed: tiny.safetensors. It can be downloaded here: https://zenodo.org/records/21788403
    run_export("kraken")
    b = datetime.datetime.now()
    print(b, f"FINISHED. It took {b-a} seconds")
