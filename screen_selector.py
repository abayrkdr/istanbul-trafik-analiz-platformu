"""
screen_selector.py — Transparent overlay for screen region selection
and a persistent capture frame with close/resize controls.
"""

from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QGuiApplication
from PyQt6.QtWidgets import QWidget, QPushButton


class ScreenSelector(QWidget):
    """Full-screen overlay — draw a rectangle to pick capture region."""

    region_selected = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ekran Bölgesi Seç")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)

        geo = QGuiApplication.primaryScreen().virtualGeometry()
        self.setGeometry(geo)

        self._origin = QPoint()
        self._current = QPoint()
        self._drawing = False

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, 120))

        if self._drawing and not self._origin.isNull():
            rect = QRect(self._origin, self._current).normalized()
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(rect, Qt.GlobalColor.transparent)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setPen(QPen(QColor("#ff6b6b"), 2))
            p.drawRect(rect)
            p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            p.setPen(QColor("#ff6b6b"))
            p.drawText(rect.x() + 6, rect.y() - 8, f"{rect.width()} × {rect.height()}")

        if not self._drawing:
            p.setPen(QColor("#ffffff"))
            p.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "🎯  Yakalamak istediğiniz bölgeyi fare ile çizin\n"
                "ESC ile iptal edin"
            )
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._origin = e.pos()
            self._current = e.pos()
            self._drawing = True
            self.update()

    def mouseMoveEvent(self, e):
        if self._drawing:
            self._current = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            rect = QRect(self._origin, e.pos()).normalized()
            if rect.width() > 20 and rect.height() > 20:
                self.region_selected.emit(rect)
            self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()


class CaptureOverlay(QWidget):
    """
    Persistent on-screen frame showing the capture region.
    Features:
      - Draggable (click & drag anywhere)
      - Close button (❌) in top-right corner
      - Re-select button (🎯) next to close
      - Displays size label
    """

    region_changed = pyqtSignal(QRect)
    closed = pyqtSignal()            # emitted when user clicks ❌
    reselect = pyqtSignal()          # emitted when user clicks 🎯

    def __init__(self, rect: QRect, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Yakalama Bölgesi")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(rect)
        self.setMinimumSize(120, 80)

        self._drag_pos = None

        # ── Close button ─────────────────────────────────────────────────
        self._btn_close = QPushButton("✕", self)
        self._btn_close.setFixedSize(26, 26)
        self._btn_close.setStyleSheet(
            "QPushButton {"
            "  background-color: #cc2222; color: white; border: none;"
            "  border-radius: 13px; font-size: 14px; font-weight: bold;"
            "}"
            "QPushButton:hover { background-color: #ff3333; }"
        )
        self._btn_close.clicked.connect(self._on_close)
        self._btn_close.setCursor(Qt.CursorShape.PointingHandCursor)

        # ── Re-select button ─────────────────────────────────────────────
        self._btn_reselect = QPushButton("🎯", self)
        self._btn_reselect.setFixedSize(26, 26)
        self._btn_reselect.setStyleSheet(
            "QPushButton {"
            "  background-color: #2266cc; color: white; border: none;"
            "  border-radius: 13px; font-size: 13px;"
            "}"
            "QPushButton:hover { background-color: #3388ff; }"
        )
        self._btn_reselect.clicked.connect(self._on_reselect)
        self._btn_reselect.setCursor(Qt.CursorShape.PointingHandCursor)

        self._position_buttons()

    def _position_buttons(self):
        self._btn_close.move(self.width() - 32, 6)
        self._btn_reselect.move(self.width() - 62, 6)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_buttons()

    def _on_close(self):
        self.closed.emit()
        self.close()

    def _on_reselect(self):
        self.reselect.emit()
        self.close()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Transparent interior
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.fillRect(self.rect(), QColor(0, 0, 0, 1))

        # Red dashed border
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.setPen(QPen(QColor("#ff6b6b"), 3, Qt.PenStyle.DashLine))
        p.drawRect(self.rect().adjusted(2, 2, -2, -2))

        # Size label
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.setPen(QColor("#ff6b6b"))
        p.drawText(8, 18, f"📷 {self.width()}×{self.height()}")

        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_pos and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            self.region_changed.emit(self.geometry())

    def mouseReleaseEvent(self, e):
        self._drag_pos = None
        self.region_changed.emit(self.geometry())
