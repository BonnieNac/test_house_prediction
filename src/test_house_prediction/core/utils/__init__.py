"""Core utilities for path management and project structure."""

from .log import setup_file_logging
from .paths import DATA_DIR, LOGS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, ensure_dirs_exist
from .version import get_project_version

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "ensure_dirs_exist",
    "get_project_version",
    "setup_file_logging",
]
