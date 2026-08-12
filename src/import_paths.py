"""Define functions for importing paths."""

import os


def import_paths():
    """Import given arguments which are saved to the user_config."""
    from config import input_folder, output_folder

    input_folder = os.path.expanduser(input_folder)
    output_folder = os.path.expanduser(output_folder)
    return input_folder, output_folder
