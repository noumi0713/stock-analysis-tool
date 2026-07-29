from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def configure_logging(log_dir: Path, verbose: bool = False) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline-{datetime.now():%Y%m%d}.log"
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return log_path
