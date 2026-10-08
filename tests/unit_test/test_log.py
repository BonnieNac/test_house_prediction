"""Unit tests for the log files."""

from pathlib import Path

import pytest
from loguru import logger

from test_house_prediction.core.utils import log, paths


def test_setup_file_logging_writes_messages(tmp_path: Path) -> None:
    """The messages logged after setup land in <directory>/<name>_<date>.log."""
    handler_id = log.setup_file_logging("unit", directory=tmp_path / "logs")
    try:
        logger.info("hello from the test")
    finally:
        logger.remove(handler_id)

    files = list((tmp_path / "logs").glob("unit_*.log"))
    assert len(files) == 1
    assert "hello from the test" in files[0].read_text(encoding="utf-8")


def test_setup_file_logging_defaults_to_logs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a directory, the file goes to LOGS_DIR."""
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "default_logs")
    handler_id = log.setup_file_logging("api")
    try:
        logger.info("api started")
    finally:
        logger.remove(handler_id)

    assert len(list((tmp_path / "default_logs").glob("api_*.log"))) == 1
