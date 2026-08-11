from src.yolo_segment_predict import run_yolo, preprocess
from src.post_yolo_processing import run_post_yolo
from src.pero_implementation import run_pero
from src.pdf_export import run_export
from datetime import datetime

if __name__ == "__main__":
    a = datetime.now()
    print(a, "Beginning OCR with Pero OCR. Code: https://github.com/DCGM/pero-ocr Model: https://nextcloud.fit.vutbr.cz/s/NtAbHTNkZFpapdJ?opendetails=")
    preprocess(downscale_factor=2)
    run_yolo(scale_factor=2)  # the higher the scale factor, the lower the page detection resolution. Keep at 1 for low-res images, choose higher values for higher resolutions.
    run_post_yolo()
    run_pero(config_path="./pero_eu_cz_print_newspapers_2022-09-26/config_cpu.ini")  # or config.ini, if a GPU can be used.
    # to work, the pero model repository has to be downloaded from: https://nextcloud.fit.vutbr.cz/s/NtAbHTNkZFpapdJ?opendetails= and unpacked in the "src" folder
    run_export("pero_local")
    b = datetime.now()
    print(b, f"FINISHED. It took {b-a} seconds")