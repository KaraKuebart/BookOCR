"""Shared fixtures for the BookOCR tests."""

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_FOLDER = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def project_root_cwd():
    """Run every test from the project root.

    Importing ``config`` changes the working directory to ``src``, which would
    otherwise leak into the following tests.

    Yields:
        The project root path.
    """
    previous = os.getcwd()
    os.chdir(PROJECT_ROOT)
    yield PROJECT_ROOT
    os.chdir(previous)


@pytest.fixture
def page_photo():
    """Return the path of a real page photo used as test input."""
    return FIXTURE_FOLDER / "test_page_1.jpg"
