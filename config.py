import os

# config.py
"""
This file should be used to configure where the OCR pipeline takes images from and where it saves output:
Set input_folder to the directory containing your input images.
Set output_folder to the directory where you want XML and PDF files saved.

After that, and after downloading models as described in the README.md, you can run either main_kraken_paddlev6.py or main_pero.py either in an IDE or from the command line.
"""

input_folder = "../input"
output_folder = "../output"

# Load folder paths from command line arguments, if they were given
try:
    from src import user_config

    input_folder = user_config.input_folder
    output_folder = user_config.output_folder
except ImportError:
    pass  # Use folders sepecified here
