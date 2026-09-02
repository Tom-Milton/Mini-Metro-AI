import ctypes
import sys


def enable_dpi_awareness() -> None:
    if sys.platform == "win32":
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
