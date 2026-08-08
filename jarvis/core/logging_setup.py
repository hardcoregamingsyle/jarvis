"""Logging configuration: coloured console output plus a rotating file log."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

from .platform_utils import IS_WINDOWS, data_dir, ensure_dir

_CONFIGURED = False

RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[48;5;203m\033[38;5;231m",
}


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if IS_WINDOWS:
        # Windows Terminal / VS Code set WT_SESSION or TERM; legacy conhost needs
        # ANSI enabled explicitly, which we attempt below.
        return _enable_windows_ansi()
    return True


def _enable_windows_ansi() -> bool:
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # -11 = STD_OUTPUT_HANDLE, 0x4 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x4))
    except Exception:  # noqa: BLE001
        return False


class ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: Optional[str] = None, color: bool = True) -> None:
        super().__init__(fmt, datefmt)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.color:
            prefix = COLORS.get(record.levelname, "")
            if prefix:
                return f"{prefix}{text}{RESET}"
        return text


def setup_logging(
    level: str = "INFO",
    *,
    log_file: Optional[Path] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger once.  Safe to call repeatedly."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)

    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(numeric)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric)
    console.setFormatter(
        ColorFormatter(
            "%(asctime)s %(levelname)-8s %(name)-22s %(message)s",
            datefmt="%H:%M:%S",
            color=_supports_color(sys.stderr),
        )
    )
    root.addHandler(console)

    try:
        target = log_file or (ensure_dir(data_dir() / "logs") / "jarvis.log")
        file_handler = logging.handlers.RotatingFileHandler(
            target, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)-28s %(funcName)s:%(lineno)d %(message)s"
            )
        )
        root.addHandler(file_handler)
    except OSError:
        # A read-only home directory must not prevent JARVIS from starting.
        root.warning("file logging disabled (could not open log file)")

    # Third-party libraries are chatty at DEBUG.
    for noisy in ("urllib3", "httpx", "httpcore", "asyncio", "numba", "filelock",
                  "huggingface_hub", "transformers", "torch", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger", "ColorFormatter"]
