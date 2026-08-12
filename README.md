# BookOCR

BookOCR is a small OCR pipeline for photographed book pages. It takes ordinary photos of a book, automatically finds and extracts the page from the photo, straightens and cleans
it, runs OCR on it and finally exports a searchable PDF containing the page images with an invisible text layer.

Two OCR back ends are supported:

* **Kraken** with the PaddleV6 `tiny` model
* **Pero OCR** with a local Pero model

## Installation

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

### Models

The YOLO page-detection models (`yolo26n-seg.pt`, `yolo26s-seg.pt`) will be automatically downloaded to the `src` folder when yolo runs. The OCR models have to be downloaded
manually:

* **Kraken:** download `tiny.safetensors` from <https://zenodo.org/records/21788403> and place it in the `src` folder.
* **Pero:** download the model archive from
  <https://nextcloud.fit.vutbr.cz/s/NtAbHTNkZFpapdJ?opendetails=> and unpack it into the `src`
  folder (should be named `src/pero_eu_cz_print_newspapers_2022-09-26`). It contains the `config.ini` (GPU) and
  `config_cpu.ini` (CPU) files referenced by the pipeline. Pero OCR code: <https://github.com/DCGM/pero-ocr>

## Run Configuration

### Option 1: Manual Configuration in `config.py`
The path of the input folder with the images and the output folder with the XML and PDF output can be set in the `config.py`:

```python
os.chdir("./src")
input_folder = "../input"
output_folder = "../output"
```

* `input_folder` – the directory containing your input photos (`.jpg`).
    * The pipeline creates the subfolders `norm`, `yolo` and `ocr_ready` inside it for the intermediate results.
* `output_folder` – the directory where the XML files (`xmls_kraken` / `xmls_pero`) and the final PDF are written.

Both may point anywhere on your machine; the defaults are the `input` and `output` folders next to the project. Note that `config.py` also changes the working directory to `src`,
so relative paths are resolved from there.

### Option 2: Command-Line Arguments

Pass folder paths as bash arguments when running the pipeline:

```bash
python main_pero.py --input ~/Desktop/input --output ~/Desktop/output
```
##### or
```bash
python main_kraken_paddlev6.py --input ~/Desktop/input --output ~/Desktop/output
```

When you provide arguments this way, they are automatically saved to `src/user_config.py` for persistence. **You only need to provide the arguments once** — on subsequent runs, the saved configuration will be used unless you override it again with new arguments or delete the src/user_config.py. Paths in config.py are used as fallback if your paths are invalid.

Tilde (`~`) expansion is supported.

Both paths may point anywhere on your machine; the defaults are the `input` and `output` folders next to the project.
## Usage

Put your page photos (`.jpg`) into the configured input folder and run one of the entry points, either from an IDE or from the command line:

```bash
python main_kraken_paddlev6.py
# or
python main_pero.py
```

Both scripts print the start and end time and the total duration. If the YOLO models fail to detect a page on some images, the affected files are listed at the end of the detection
step so they can be cropped manually.

Useful parameters in the main scripts:

* `preprocess(downscale_factor=2)` – how strongly the input images are downscaled before processing.
* `run_yolo(scale_factor=2)` – the higher the factor, the lower the resolution used for page detection. Keep it at `1` for low-resolution images, use higher values for
  high-resolution photos.
* `run_pero(config_path=...)` – use `config_cpu.ini` for CPU-only machines and `config.ini` if a GPU is available.

## Output

* `<output_folder>/xmls_kraken/*.xml` or `<output_folder>/xmls_pero/*_page.xml` – PAGE XML with the recognized text lines.
* `<output_folder>/output_kraken.pdf` or `<output_folder>/output_pero.pdf` – the merged, searchable PDF.

## Pipeline

Both entry points share the same steps:

1. **Preprocessing** (`src/yolo_segment_predict.py`, `preprocess`) – images are downscaled and their contrast is normalized, so that too dark or too bright photos become
   comparable.
2. **Page detection** (`src/yolo_segment_predict.py`, `run_yolo`) – a YOLO segmentation model (`yolo26n-seg.pt`, with `yolo26s-seg.pt` as a fallback) detects the book page and
   everything outside the page mask is made transparent.
3. **Post-processing** (`src/post_yolo_processing.py`, `run_post_yolo`) – the extracted page is rotated to the angle that minimizes its bounding box (deskewing), transparent
   borders are cropped and remaining transparent areas are filled with the mean page colour.
4. **OCR** – either `src/kraken_implementation.py` (`run_kraken`, runs the `kraken` CLI in parallel over all CPU cores) or `src/pero_implementation.py` (`run_pero`). Both produce
   PAGE XML files.
5. **PDF export** (`src/pdf_export.py`, `run_export`) – the page images are merged into one PDF with the recognized text placed invisibly on the baselines, so the resulting PDF is
   searchable and the text can be copied.
