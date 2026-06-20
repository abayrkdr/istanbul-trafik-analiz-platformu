"""
app.py — Entry point for the Traffic Analysis Desktop Application.
Run:  python app.py
"""

import sys

# QtWebEngine, QApplication oluşturulmadan ÖNCE import edilmelidir (harita seçici için)
try:
    from PyQt6 import QtWebEngineWidgets  # noqa: F401
except ImportError:
    pass

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from ui import MainWindow


def main():
    app = QApplication(sys.argv)

    # Set application-wide default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Application metadata
    app.setApplicationName("Trafik Analiz Sistemi")
    app.setOrganizationName("TrafficAI")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
