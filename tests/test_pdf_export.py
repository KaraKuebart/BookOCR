"""Tests for the searchable PDF export."""

import numpy as np
import pytest
from PIL import Image
from PyPDF2 import PdfReader

from src import pdf_export

KRAKEN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="page.png" imageWidth="400" imageHeight="200">
    <TextRegion id="r1">
      <Coords points="10,40 390,40 390,90 10,90"/>
      <TextLine id="l1">
        <Coords points="10,40 390,40 390,90 10,90"/>
        <Baseline points="10,80 390,80"/>
        <TextEquiv><Unicode>Gutenberg</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""


def test_make_pdf_from_kraken_creates_searchable_pdf(tmp_path):
    """The exported PDF contains the recognized text of the XML."""
    image_folder = tmp_path / "ocr_ready"
    xml_folder = tmp_path / "xmls_kraken"
    image_folder.mkdir()
    xml_folder.mkdir()
    Image.fromarray(np.full((200, 400, 3), 255, dtype=np.uint8)).save(
        image_folder / "page.png"
    )
    (xml_folder / "page.xml").write_text(KRAKEN_XML, encoding="utf-8")

    pdf_export.make_pdf_from_kraken(str(image_folder), str(xml_folder))

    pdf_file = tmp_path / "output_kraken.pdf"
    assert pdf_file.exists()
    reader = PdfReader(str(pdf_file))
    assert len(reader.pages) == 1
    assert "Gutenberg" in reader.pages[0].extract_text()


def test_make_pdf_from_kraken_skips_images_without_xml(tmp_path):
    """Images without a matching XML file do not end up in the PDF."""
    image_folder = tmp_path / "ocr_ready"
    xml_folder = tmp_path / "xmls_kraken"
    image_folder.mkdir()
    xml_folder.mkdir()
    Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8)).save(
        image_folder / "lonely.png"
    )

    pdf_export.make_pdf_from_kraken(str(image_folder), str(xml_folder))

    assert len(PdfReader(str(tmp_path / "output_kraken.pdf")).pages) == 0


def test_run_export_rejects_unknown_source():
    """An unsupported XML source is reported instead of silently ignored."""
    with pytest.raises(ValueError):
        pdf_export.run_export("tesseract")
