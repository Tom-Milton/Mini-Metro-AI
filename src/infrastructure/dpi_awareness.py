import ctypes


def enable_dpi_awareness() -> None:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
