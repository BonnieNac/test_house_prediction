"""Log files: copy the loguru messages of a run into logs/."""

from pathlib import Path

from loguru import logger

from test_house_prediction.core.utils import paths


def setup_file_logging(name: str, directory: Path | None = None, level: str = "INFO") -> int:
    """Also write the logs to ``logs/<name>_<date>.log`` (the terminal output is kept).

    One file per day and per entry point: a new run of the same day is appended to it. Files
    over 10 MB start a new one, files older than 30 days are deleted.

    Args:
        name: Entry point name, used as the file prefix (e.g. ``train``).
        directory: Destination folder (defaults to logs/).
        level: Minimum level written to the file.

    Returns:
        int: The loguru handler id (``logger.remove(id)`` stops writing to the file).

    """
    directory = directory or paths.LOGS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{{time:YYYY-MM-DD}}.log"
    return logger.add(path, level=level, rotation="10 MB", retention="30 days", encoding="utf-8")
