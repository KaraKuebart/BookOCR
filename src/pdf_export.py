import glob
import io
import json
import math
import os
import unicodedata

from lxml import etree
from PIL import Image
from PyPDF2 import PdfMerger
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from tqdm import tqdm

from config import input_folder, output_folder


def make_pdf_from_pero_local(
    png_path: str = f"{input_folder}/ocr_ready",
    xml_path: str = f"{output_folder}/xmls_pero",
):
    files = glob.glob(f"{png_path}/*.png")
    files = sorted(files)
    merger = PdfMerger()
    page_num = 0
    for page in tqdm(files):
        page_num += 1
        jpg_file = page
        page_name = page.split("/")[-1].split(".")[0]
        xml_file = f"{xml_path}/{page_name}_page.xml"
        if not os.path.exists(xml_file):
            continue

        img = Image.open(jpg_file)
        img_width, img_height = img.size

        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=(img_width, img_height))

        # Draw image
        c.drawImage(jpg_file, 0, 0, width=img_width, height=img_height)

        # Parse PAGE XML and overlay text
        xml = etree.parse(xml_file)
        c.setFillAlpha(0)  # Invisible text

        # Define PAGE XML namespace
        ns = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}

        # Iterate through all TextLines in all TextRegions
        for textline in xml.xpath("//page:TextLine", namespaces=ns):
            # Get text content
            text_elem = textline.find(".//page:TextEquiv/page:Unicode", namespaces=ns)
            if text_elem is None or text_elem.text is None:
                continue

            text = text_elem.text

            # Get baseline points (required for proper positioning)
            baseline = textline.find(".//page:Baseline", namespaces=ns)
            if baseline is None:
                continue

            points_str = baseline.get("points")
            if not points_str:
                continue

            # Parse baseline points
            points = [tuple(map(float, p.split(","))) for p in points_str.split()]

            # Calculate the midpoint for X position (more stable than first point)
            x = (points[0][0] + points[-1][0]) / 2
            y = img_height - (points[0][1] + points[-1][1]) / 2

            # Calculate the baseline angle for proper text rotation
            dx = points[-1][0] - points[0][0]
            dy = points[-1][1] - points[0][1]
            angle = math.atan2(dy, dx)  # in radians
            # Calculate font size from line height in custom metadata
            custom = textline.get("custom", "{}")
            try:
                line_height = float(
                    custom.split(":[")[1].split(",")[0]
                )  # First height is the text height
                font_size = max(
                    6, min(100, int(line_height * 1.3))
                )  # Scale to 80% of height, cap at 6-24pt
            except Exception as e:
                font_size = 36
                print(f"font size detection failure: {e}")

            c.saveState()
            c.translate(x, y)
            c.rotate(-math.degrees(angle))
            c.setFont("Helvetica", font_size)
            # Get text width and offset back by half
            text_width = c.stringWidth(text, "Helvetica", font_size)
            c.drawString(-text_width / 2, 0, text)  # Center horizontally
            c.restoreState()

        c.save()
        pdf_buffer.seek(0)
        merger.append(pdf_buffer)

    merger.write(f"{xml_path}/../output_pero.pdf")
    merger.close()


def make_pdf_from_pero_web(path: str):
    files = glob.glob(f"{path}/*.JPG")
    files = sorted(files)
    merger = PdfMerger()
    page_num = 0
    for page in tqdm(files):
        page_num += 1
        jpg_file = page
        page_name = page.split("/")[-1].split(".")[0]
        xml_file = f"{path}/pages/{page_name}.xml"

        img = Image.open(jpg_file)
        img_width, img_height = img.size

        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=(img_width, img_height))

        # Draw image
        c.drawImage(jpg_file, 0, 0, width=img_width, height=img_height)

        # Parse PAGE XML and overlay text
        xml = etree.parse(xml_file)
        c.setFillAlpha(0)  # Invisible text

        # Define PAGE XML namespace
        ns = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}

        # Iterate through all TextLines in all TextRegions
        for textline in xml.xpath("//page:TextLine", namespaces=ns):
            # Get text content
            text_elem = textline.find(".//page:TextEquiv/page:Unicode", namespaces=ns)
            if text_elem is None or text_elem.text is None:
                continue

            text = text_elem.text

            # Get baseline points (required for proper positioning)
            baseline = textline.find(".//page:Baseline", namespaces=ns)
            if baseline is None:
                continue

            points_str = baseline.get("points")
            if not points_str:
                continue

            # Parse baseline points
            points = [tuple(map(float, p.split(","))) for p in points_str.split()]

            # Calculate the midpoint for X position (more stable than first point)
            x = (points[0][0] + points[-1][0]) / 2
            y = img_height - (points[0][1] + points[-1][1]) / 2

            # Calculate the baseline angle for proper text rotation
            dx = points[-1][0] - points[0][0]
            dy = points[-1][1] - points[0][1]
            angle = math.atan2(dy, dx)  # in radians
            # Calculate font size from line height in custom metadata
            custom = textline.get("custom", "{}")
            try:
                custom_data = json.loads(custom)
                heights = custom_data.get("heights", [])
                if not heights:
                    font_size = 36
                else:
                    line_height = heights[0]  # First height is the text height
                    font_size = max(
                        6, min(100, int(line_height * 1.15))
                    )  # Scale to 80% of height, cap at 6-24pt
            except:
                font_size = 36
            c.saveState()
            c.translate(x, y)
            c.rotate(-math.degrees(angle))
            c.setFont("Helvetica", font_size)
            # Get text width and offset back by half
            text_width = c.stringWidth(text, "Helvetica", font_size)
            c.drawString(-text_width / 2, 0, text)  # Center horizontally
            c.restoreState()
        c.save()
        pdf_buffer.seek(0)
        merger.append(pdf_buffer)

    merger.write(f"{path}/output.pdf")
    merger.close()


def make_pdf_from_kraken(jpg_path: str, xml_path: str):
    """
    Create a PDF from JPG images with invisible searchable text overlay from Kraken XML.

    Args:
        jpg_path: Path to folder containing JPG/PNG images
        xml_path: Path to folder containing corresponding Kraken PAGE XML files
    """

    try:
        canvas.registerFont(
            TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        )
        font_to_use = "DejaVu"
    except:
        font_to_use = "Helvetica"

    # Get all image files
    files = glob.glob(f"{jpg_path}/*.JPG") + glob.glob(f"{jpg_path}/*.png")
    files = sorted(files)

    merger = PdfMerger()

    for image_file in tqdm(files):
        # Extract base filename
        page_name = image_file.split("/")[-1].split(".")[0]
        xml_file = f"{xml_path}/{page_name}.xml"

        # Check if corresponding XML exists
        if not os.path.exists(xml_file):
            print(f"Warning: No XML found for {page_name}, skipping")
            continue

        # Open and get image dimensions
        img = Image.open(image_file)
        img_width, img_height = img.size

        # Create PDF
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=(img_width, img_height))

        # Draw image
        c.drawImage(image_file, 0, 0, width=img_width, height=img_height)

        # Parse Kraken PAGE XML
        xml = etree.parse(xml_file)
        c.setFillAlpha(0)  # Invisible text

        # Define PAGE XML namespace
        ns = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}

        # Iterate through all TextLines
        for textline in xml.xpath("//page:TextLine", namespaces=ns):
            # Get full text from TextLine level (not Word or Glyph)
            text_elems = textline.findall(
                ".//page:TextEquiv/page:Unicode", namespaces=ns
            )
            if not text_elems:
                continue
            text_elem = text_elems[-1]
            if text_elem is None or text_elem.text is None:
                continue

            text = text_elem.text
            text = unicodedata.normalize("NFC", text)

            # Get baseline points
            baseline = textline.find(".//page:Baseline", namespaces=ns)
            if baseline is None:
                continue

            points_str = baseline.get("points")
            if not points_str:
                continue

            # Parse baseline points
            points = [tuple(map(float, p.split(","))) for p in points_str.split()]

            # Calculate the midpoint for X position (more stable than first point)
            x = (points[0][0] + points[-1][0]) / 2
            y = img_height - (points[0][1] + points[-1][1]) / 2

            # Calculate the baseline angle for proper text rotation
            dx = points[-1][0] - points[0][0]
            dy = points[-1][1] - points[0][1]
            angle = math.atan2(dy, dx)  # in radians

            # Estimate font size from Coords bounding box
            coords = textline.find(".//page:Coords", namespaces=ns)
            baseline = textline.find(".//page:Baseline", namespaces=ns)
            if coords is not None:
                coords_str = coords.get("points")
                if baseline is not None:
                    baseline_str = baseline.get("points")
                    baseline_points = [
                        tuple(map(float, p.split(","))) for p in baseline_str.split()
                    ]
                    baseline_y = [p[1] for p in baseline_points]
                    baseline_tilt = max(baseline_y) - min(baseline_y)
                else:
                    baseline_tilt = 0
                if coords_str:
                    coord_points = [
                        tuple(map(float, p.split(","))) for p in coords_str.split()
                    ]
                    # Get min/max Y coordinates to estimate height
                    y_coords = [p[1] for p in coord_points]
                    line_height = max(y_coords) - min(y_coords) - baseline_tilt
                    font_size = max(
                        6, min(72, int(line_height * 0.8))
                    )  # Scale to 90% of height
                else:
                    font_size = 10
            else:
                font_size = 10

            c.setFont(font_to_use, font_size)
            c.saveState()
            c.translate(x, y)
            c.rotate(-math.degrees(angle))
            # Get text width and offset back by half
            text_width = c.stringWidth(text, "Helvetica", font_size)
            c.drawString(-text_width / 2, 0, text)  # Center horizontally
            c.restoreState()

        c.save()
        pdf_buffer.seek(0)
        merger.append(pdf_buffer)

    merger.write(f"{xml_path}/../output_kraken.pdf")
    merger.close()


def run_export(xml_source, pero_path: str = None):
    if xml_source == "kraken":
        make_pdf_from_kraken(
            jpg_path=f"{input_folder}/ocr_ready",
            xml_path=f"{output_folder}/xmls_kraken",
        )
    elif xml_source == "pero_local":
        make_pdf_from_pero_local()
    elif xml_source == "pero_web":
        if pero_path:
            make_pdf_from_pero_web(pero_path)
        else:
            raise ValueError("No pero path specified")
    else:
        raise ValueError(
            "invalid xml-source. XML Source must be 'kraken' or 'pero_web' or 'pero_local'"
        )


if __name__ == "__main__":
    run_export("pero_local")
