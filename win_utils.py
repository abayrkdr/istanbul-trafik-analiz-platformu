"""
win_utils.py — Windows-specific utilities for window management & capture.
Uses pywin32 + ctypes for PrintWindow (background window capture).
"""

import ctypes
from ctypes import wintypes

import cv2
import numpy as np
import win32gui
import win32ui
import win32con

user32 = ctypes.windll.user32

# ── Constants ────────────────────────────────────────────────────────────
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
PW_RENDERFULLCONTENT = 0x00000002
SW_RESTORE = 9


# ── Window enumeration ──────────────────────────────────────────────────

def get_visible_windows() -> list:
    """Return [(hwnd, title), ...] for all visible windows with titles."""
    results = []
    skip = [
        "Program Manager", "MSCTFIME UI", "Default IME",
        "Setup", "Windows Input Experience",
    ]

    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and len(title.strip()) >= 3:
                if not any(s in title for s in skip):
                    results.append((int(hwnd), title.strip()))
        return True

    win32gui.EnumWindows(_cb, None)
    return results


# ── Window pinning ───────────────────────────────────────────────────────

def pin_window(hwnd: int):
    """Make a window always-on-top."""
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)


def unpin_window(hwnd: int):
    """Remove always-on-top flag."""
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)


def is_pinned(hwnd: int) -> bool:
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    return bool(style & WS_EX_TOPMOST)


# ── Window manipulation ─────────────────────────────────────────────────

def bring_to_front(hwnd: int):
    """Bring a window to foreground and restore if minimized."""
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)


def get_window_rect(hwnd: int) -> tuple:
    """Return (left, top, width, height) of window."""
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return left, top, right - left, bottom - top


def is_window_valid(hwnd: int) -> bool:
    """Check if window handle is still valid."""
    return bool(win32gui.IsWindow(hwnd))


# ── Background window capture ───────────────────────────────────────────

def capture_window(hwnd: int) -> np.ndarray | None:
    """
    Capture a window's content using Win32 PrintWindow API.
    Works even when the window is behind other windows.

    Returns BGR numpy array (full window including title bar) or None.
    """
    if not win32gui.IsWindow(hwnd):
        return None

    # Don't capture minimized windows
    if win32gui.IsIconic(hwnd):
        return None

    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return None

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        result = ctypes.windll.user32.PrintWindow(
            hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT
        )

        bmp_info = bitmap.GetInfo()
        bmp_bits = bitmap.GetBitmapBits(True)

        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        if result != 1:
            return None

        img = np.frombuffer(bmp_bits, dtype=np.uint8)
        img = img.reshape((bmp_info["bmHeight"], bmp_info["bmWidth"], 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    except Exception:
        return None
