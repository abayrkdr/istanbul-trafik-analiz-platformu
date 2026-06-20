"""
ui.py — Modern dark-themed PyQt6 main window.
Features: multi-polygon zones, heatmap, CSV export, window pinning,
window capture (Alt-Tab safe), tabbed sources.
"""

import os
import time as _time
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLineEdit, QComboBox, QSlider,
    QTextEdit, QFileDialog, QFrame, QGroupBox, QSizePolicy,
    QSpacerItem, QScrollArea, QTabWidget, QCheckBox, QDialog,
    QListWidget, QListWidgetItem, QDialogButtonBox,
    QMessageBox, QInputDialog, QTableWidget, QTableWidgetItem, QHeaderView,
)

from video_thread import (
    VideoThread, SOURCE_FILE, SOURCE_URL, SOURCE_SCREEN, SOURCE_WINDOW,
)
from screen_selector import ScreenSelector, CaptureOverlay
from analytics import ZONE_COLORS_HEX, ZONE_COLORS_BGR


# ─────────────────────────────────────────────────────────────────────────
DARK_QSS = """
* { margin:0; padding:0; }
QWidget {
    background-color: #0c0c10; color: #d4d4dc;
    font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px;
}
QMainWindow { background-color: #0c0c10; }

QTabWidget::pane { border:1px solid #22222e; border-radius:8px; background:#13131a; top:-1px; }
QTabBar::tab {
    background:#13131a; border:1px solid #22222e; border-bottom:none;
    border-top-left-radius:8px; border-top-right-radius:8px;
    padding:7px 16px; margin-right:2px;
    color:#707090; font-weight:600; font-size:12px;
}
QTabBar::tab:selected { background:#1a1a24; color:#c0c0ff; border-color:#3838a0; }
QTabBar::tab:hover:!selected { background:#18181f; color:#a0a0c0; }

QScrollArea { background:transparent; border:none; }
QScrollBar:vertical { background:#0c0c10; width:7px; border-radius:3px; }
QScrollBar::handle:vertical { background:#28283a; border-radius:3px; min-height:30px; }
QScrollBar::handle:vertical:hover { background:#3a3a50; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; border:none; }

QGroupBox {
    background-color:#13131a; border:1px solid #22222e; border-radius:10px;
    margin-top:14px; padding:14px 10px 10px 10px;
    font-weight:600; font-size:12px; color:#8080a0;
}
QGroupBox::title {
    subcontrol-origin:margin; subcontrol-position:top left;
    padding:2px 12px; background-color:#1a1a24;
    border:1px solid #22222e; border-radius:6px; color:#b0b0ff;
}

QPushButton {
    background-color:#1a1a24; border:1px solid #28283a; border-radius:8px;
    padding:8px 16px; color:#d0d0e0; font-weight:600; font-size:12px;
}
QPushButton:hover { background-color:#24243a; border-color:#5050ff; }
QPushButton:pressed { background-color:#3030a0; }
QPushButton:disabled { background-color:#0f0f14; color:#3a3a48; border-color:#18181f; }
QPushButton#btn_start {
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #166534,stop:1 #1a8a4a);
    border-color:#22aa60; color:#d0ffd0; font-size:13px;
}
QPushButton#btn_stop {
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6b1a1a,stop:1 #882222);
    border-color:#bb3030; color:#ffe0e0;
}
QPushButton#btn_pause {
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #5a4b10,stop:1 #7a6a18);
    border-color:#aa9030; color:#fff5d0;
}
QPushButton#btn_screen {
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #4a1a6b,stop:1 #6a2a8a);
    border-color:#9040cc; color:#f0d0ff;
}

QLineEdit {
    background-color:#0f0f16; border:1px solid #22222e; border-radius:7px;
    padding:7px 12px; color:#e0e0f0; font-size:12px;
}
QLineEdit:focus { border-color:#5050ff; }

QComboBox {
    background-color:#0f0f16; border:1px solid #22222e; border-radius:7px;
    padding:6px 12px; color:#e0e0f0;
}
QComboBox::drop-down { border:none; width:28px; }
QComboBox QAbstractItemView {
    background-color:#13131a; border:1px solid #28283a;
    selection-background-color:#3030a0; color:#d0d0e0;
}

QSlider::groove:horizontal { height:5px; background:#1a1a24; border-radius:2px; }
QSlider::handle:horizontal { background:#5858ff; width:15px; height:15px; margin:-5px 0; border-radius:7px; }
QSlider::sub-page:horizontal {
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #3030c0,stop:1 #5858ff); border-radius:2px;
}
QSlider#timeline_slider::groove:horizontal { height:4px; background:#18181f; border-radius:2px; }
QSlider#timeline_slider::handle:horizontal { background:#ff5c5c; width:13px; height:13px; margin:-5px 0; border-radius:6px; }
QSlider#timeline_slider::sub-page:horizontal {
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #cc2222,stop:1 #ff5c5c); border-radius:2px;
}
QSlider#timeline_slider:disabled::handle:horizontal { background:#28283a; }
QSlider#timeline_slider:disabled::sub-page:horizontal { background:#18181f; }

QTextEdit {
    background-color:#08080c; border:1px solid #18181f; border-radius:8px;
    padding:6px; font-family:'Cascadia Code','Consolas',monospace; font-size:11px; color:#909098;
}
QLabel { background:transparent; color:#d0d0e0; }
QLabel#video_placeholder {
    background-color:#06060a; border:2px dashed #1a1a24;
    border-radius:14px; color:#303050; font-size:15px;
}
QLabel#time_label { color:#707088; font-size:11px; font-family:'Cascadia Code','Consolas',monospace; }
QFrame#metric_card { background-color:#13131a; border:1px solid #22222e; border-radius:10px; padding:6px; }
QCheckBox { color:#b0b0c8; font-size:12px; spacing:6px; }
QCheckBox::indicator { width:16px; height:16px; border:1px solid #28283a; border-radius:4px; background:#0f0f16; }
QCheckBox::indicator:checked { background:#5858ff; border-color:#7070ff; }
QDialog { background-color:#0c0c10; }
QListWidget { background-color:#0f0f16; border:1px solid #22222e; border-radius:6px; color:#d0d0e0; }
QListWidget::item { padding:6px; }
QListWidget::item:selected { background-color:#3030a0; }
"""


# ─────────────────────────────────────────────────────────────────────────
#  WIDGETS
# ─────────────────────────────────────────────────────────────────────────

class MetricCard(QFrame):
    def __init__(self, title, initial="—", accent="#5858ff"):
        super().__init__()
        self.setObjectName("metric_card")
        self.setFixedHeight(74)
        l = QVBoxLayout(self); l.setContentsMargins(10, 6, 10, 6); l.setSpacing(1)
        self._t = QLabel(title)
        self._t.setStyleSheet("font-size:10px;color:#606080;font-weight:600;")
        self._v = QLabel(initial)
        self._v.setStyleSheet(f"font-size:22px;font-weight:700;color:{accent};")
        l.addWidget(self._t); l.addWidget(self._v)

    def set_value(self, t): self._v.setText(t)
    def set_accent(self, c): self._v.setStyleSheet(f"font-size:22px;font-weight:700;color:{c};")
    def set_title(self, t): self._t.setText(t)


class DensityBadge(QFrame):
    C = {
        "Düşük": ("#0d3320", "#22c55e", "●  Düşük"),
        "Orta":  ("#332b0d", "#eab308", "●  Orta"),
        "Yüksek":("#330d0d", "#ef4444", "●  Yüksek"),
    }
    def __init__(self):
        super().__init__()
        self.setFixedHeight(34)
        self.setStyleSheet(self._s("Düşük"))
        l = QHBoxLayout(self); l.setContentsMargins(12, 4, 12, 4)
        self._l = QLabel("●  Düşük")
        self._l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._l.setStyleSheet("font-weight:700;font-size:12px;")
        l.addWidget(self._l)

    def update_level(self, lv):
        bg, fg, tx = self.C.get(lv, self.C["Düşük"])
        self.setStyleSheet(self._s(lv))
        self._l.setText(tx)
        self._l.setStyleSheet(f"font-weight:700;font-size:12px;color:{fg};")

    @staticmethod
    def _s(lv):
        bg, fg, _ = DensityBadge.C.get(lv, DensityBadge.C["Düşük"])
        return f"background-color:{bg};border:1px solid {fg};border-radius:8px;"


class ClickableVideoLabel(QLabel):
    def __init__(self, mw):
        super().__init__()
        self._mw = mw

    def mousePressEvent(self, ev: QMouseEvent):
        if self._mw._calib_mode and ev.button() == Qt.MouseButton.LeftButton:
            self._mw._on_calib_click(ev.pos())
        elif self._mw._poly_drawing and ev.button() == Qt.MouseButton.LeftButton:
            self._mw._on_video_click(ev.pos())
        elif self._mw._poly_drawing and ev.button() == Qt.MouseButton.RightButton:
            self._mw._finish_polygon()
        else:
            super().mousePressEvent(ev)


class WindowPickerDialog(QDialog):
    """Dialog to select a window for pinning."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📌 Pencere Seç — Üste Sabitle")
        self.setFixedSize(450, 400)
        self.setStyleSheet(DARK_QSS)
        l = QVBoxLayout(self)
        l.addWidget(QLabel("Üste sabitlemek istediğiniz pencereyi seçin:"))
        self._list = QListWidget()
        l.addWidget(self._list)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        l.addWidget(btns)
        self.selected_hwnd = None
        self._populate()

    def _populate(self):
        from win_utils import get_visible_windows
        for hwnd, title in get_visible_windows():
            if len(title) < 3:
                continue
            item = QListWidgetItem(f"  {title}")
            item.setData(Qt.ItemDataRole.UserRole, hwnd)
            self._list.addItem(item)

    def accept(self):
        item = self._list.currentItem()
        if item:
            self.selected_hwnd = item.data(Qt.ItemDataRole.UserRole)
        super().accept()


# ─────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trafik Analiz Sistemi — YOLOv11")
        self.setMinimumSize(1280, 720)
        self.resize(1500, 900)
        self.setStyleSheet(DARK_QSS)

        self._thread = VideoThread()
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.progress_update.connect(self._on_progress)
        self._thread.log_message.connect(self._on_log)
        self._thread.finished_signal.connect(self._on_finished)

        self._source_path = ""
        self._source_mode = SOURCE_FILE
        self._is_seekable = False
        self._is_paused = False
        self._slider_dragging = False
        self._total_frames = 0
        self._capture_overlay = None
        self._screen_rect = QRect()

        # Window capture state
        self._win_hwnd: int = 0
        self._win_title: str = ""
        self._win_crop: tuple = (0, 0, 0, 0)  # (x, y, w, h) in window coords

        # Multi-polygon state
        self._poly_drawing = False
        self._all_polygons: list = []
        self._current_poly: list = []

        # Hız kalibrasyonu (2 nokta + gerçek mesafe → metre/piksel)
        self._calib_mode = False
        self._calib_pts: list = []
        self._m_per_px = 0.0

        # Canlı rapor önizleme durumu
        self._last_report_row_count = -1
        self._frame_size = (0, 0)

        self._build_ui()
        self._on_log("🚀 Uygulama hazır. Kaynak seçin → ▶ BAŞLAT")

    # ─────────────────────────────────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QHBoxLayout(c)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ═══ LEFT — Video ═══════════════════════════════════════════════
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        self._video_label = ClickableVideoLabel(self)
        self._video_label.setObjectName("video_placeholder")
        self._video_label.setText("Video Önizleme\n\nKaynak seçin ve BAŞLAT'a basın")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._video_label.setMinimumSize(640, 360)
        ll.addWidget(self._video_label, stretch=1)

        # Timeline
        tl = QFrame()
        tl.setStyleSheet(
            "background:#0f0f16;border:1px solid #1a1a24;border-radius:10px;"
        )
        tll = QVBoxLayout(tl)
        tll.setContentsMargins(14, 8, 14, 8)
        tll.setSpacing(4)

        self._timeline = QSlider(Qt.Orientation.Horizontal)
        self._timeline.setObjectName("timeline_slider")
        self._timeline.setRange(0, 1000)
        self._timeline.setEnabled(False)
        self._timeline.sliderPressed.connect(
            lambda: setattr(self, "_slider_dragging", True)
        )
        self._timeline.sliderReleased.connect(self._on_slider_release)
        tll.addWidget(self._timeline)

        cr = QHBoxLayout()
        cr.setSpacing(6)
        self._lbl_time = QLabel("00:00 / 00:00")
        self._lbl_time.setObjectName("time_label")
        self._lbl_time.setFixedWidth(115)
        cr.addWidget(self._lbl_time)
        self._mode_badge = QLabel("")
        self._mode_badge.setVisible(False)
        self._mode_badge.setFixedHeight(22)
        cr.addWidget(self._mode_badge)
        cr.addStretch()

        self._btn_back = QPushButton("⏪ 10s")
        self._btn_back.setFixedWidth(58)
        self._btn_back.setEnabled(False)
        self._btn_back.clicked.connect(lambda: self._skip(-10))
        cr.addWidget(self._btn_back)

        self._btn_pause = QPushButton("⏸  Duraklat")
        self._btn_pause.setObjectName("btn_pause")
        self._btn_pause.setFixedWidth(110)
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._toggle_pause)
        cr.addWidget(self._btn_pause)

        self._btn_fwd = QPushButton("10s ⏩")
        self._btn_fwd.setFixedWidth(58)
        self._btn_fwd.setEnabled(False)
        self._btn_fwd.clicked.connect(lambda: self._skip(10))
        cr.addWidget(self._btn_fwd)

        tll.addLayout(cr)
        ll.addWidget(tl)
        root.addWidget(left, stretch=3)

        # ═══ RIGHT — Controls ══════════════════════════════════════════
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(380)
        rp = QWidget()
        rl = QVBoxLayout(rp)
        rl.setContentsMargins(4, 0, 4, 0)
        rl.setSpacing(5)

        # ── Source Tabs ──────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.setFixedHeight(160)

        # Tab: File
        t1 = QWidget()
        t1l = QVBoxLayout(t1)
        t1l.setContentsMargins(8, 10, 8, 8)
        t1l.setSpacing(6)
        b1 = QPushButton("📂  GÖZAT — Video Dosyası")
        b1.clicked.connect(self._browse)
        t1l.addWidget(b1)
        self._file_lbl = QLabel("Seçilen: —")
        self._file_lbl.setStyleSheet("font-size:11px;color:#606080;")
        self._file_lbl.setWordWrap(True)
        t1l.addWidget(self._file_lbl)
        t1l.addStretch()
        tabs.addTab(t1, "📄  Dosya")

        # Tab: URL
        t2 = QWidget()
        t2l = QVBoxLayout(t2)
        t2l.setContentsMargins(8, 10, 8, 8)
        t2l.setSpacing(6)
        t2l.addWidget(QLabel("YouTube / RTSP / Web URL:"))
        ur = QHBoxLayout()
        self._url_inp = QLineEdit()
        self._url_inp.setPlaceholderText("https://youtube.com/... veya rtsp://...")
        b2 = QPushButton("Ayarla")
        b2.setFixedWidth(64)
        b2.clicked.connect(self._set_url)
        ur.addWidget(self._url_inp)
        ur.addWidget(b2)
        t2l.addLayout(ur)
        self._url_lbl = QLabel("Kaynak: —")
        self._url_lbl.setStyleSheet("font-size:11px;color:#606080;")
        self._url_lbl.setWordWrap(True)
        t2l.addWidget(self._url_lbl)
        t2l.addStretch()
        tabs.addTab(t2, "🌐  URL / Canlı")

        # Tab: İBB Trafik Kameraları
        t3 = QWidget()
        t3l = QVBoxLayout(t3)
        t3l.setContentsMargins(8, 10, 8, 8)
        t3l.setSpacing(6)
        b3 = QPushButton("🗺️  İBB Trafik Haritasını Aç")
        b3.clicked.connect(self._open_ibb_map)
        t3l.addWidget(b3)
        hint = QLabel("Haritadaki 📹 kamera ikonuna tıklayın; canlı yayın otomatik yakalanır.")
        hint.setStyleSheet("font-size:10px;color:#505070;")
        hint.setWordWrap(True)
        t3l.addWidget(hint)
        self._ibb_lbl = QLabel("Yayın: —")
        self._ibb_lbl.setStyleSheet("font-size:11px;color:#606080;")
        self._ibb_lbl.setWordWrap(True)
        t3l.addWidget(self._ibb_lbl)
        t3l.addStretch()
        tabs.addTab(t3, "🚦  İBB Kamera")

        tabs.currentChanged.connect(self._on_tab)
        rl.addWidget(tabs)

        # ── Model ────────────────────────────────────────────────────────
        gm = QGroupBox("🧠  Model & Cihaz")
        ml = QGridLayout(gm)
        ml.setSpacing(6)
        ml.addWidget(QLabel("Model:"), 0, 0)
        self._cb_model = QComboBox()
        self._cb_model.addItems([
            "Nano (yolo11n)", "Small (yolo11s)",
            "Medium (yolo11m)", "Large (yolo11l)",
        ])
        ml.addWidget(self._cb_model, 0, 1)
        ml.addWidget(QLabel("Cihaz:"), 1, 0)
        self._cb_device = QComboBox()
        self._cb_device.addItems(["CUDA (GPU)", "CPU", "MPS (Apple)"])
        ml.addWidget(self._cb_device, 1, 1)
        rl.addWidget(gm)

        # ── Settings ─────────────────────────────────────────────────────
        gc = QGroupBox("⚙️  Ayarlar")
        cl = QVBoxLayout(gc)
        cl.setSpacing(5)
        cl.addWidget(QLabel("Güven Eşiği:"))
        cr2 = QHBoxLayout()
        self._sl_conf = QSlider(Qt.Orientation.Horizontal)
        self._sl_conf.setRange(10, 90)
        self._sl_conf.setValue(35)
        self._lbl_conf = QLabel("0.35")
        self._lbl_conf.setFixedWidth(36)
        self._sl_conf.valueChanged.connect(
            lambda v: self._lbl_conf.setText(f"{v / 100:.2f}")
        )
        cr2.addWidget(self._sl_conf)
        cr2.addWidget(self._lbl_conf)
        cl.addLayout(cr2)

        cl.addWidget(QLabel("Sayım Çizgisi (Y %):"))
        lr = QHBoxLayout()
        self._sl_line = QSlider(Qt.Orientation.Horizontal)
        self._sl_line.setRange(10, 90)
        self._sl_line.setValue(50)
        self._lbl_line = QLabel("50%")
        self._lbl_line.setFixedWidth(36)
        self._sl_line.valueChanged.connect(lambda v: self._lbl_line.setText(f"{v}%"))
        lr.addWidget(self._sl_line)
        lr.addWidget(self._lbl_line)
        cl.addLayout(lr)

        # 2. sayım çizgisi (opsiyonel — çift yön/ayrı yol için)
        self._chk_line2 = QCheckBox("➕ 2. Sayım Çizgisi")
        cl.addWidget(self._chk_line2)
        lr2 = QHBoxLayout()
        self._sl_line2 = QSlider(Qt.Orientation.Horizontal)
        self._sl_line2.setRange(10, 90)
        self._sl_line2.setValue(30)
        self._sl_line2.setEnabled(False)
        self._lbl_line2 = QLabel("30%")
        self._lbl_line2.setFixedWidth(36)
        self._sl_line2.valueChanged.connect(lambda v: self._lbl_line2.setText(f"{v}%"))
        self._chk_line2.stateChanged.connect(
            lambda st: self._sl_line2.setEnabled(st == Qt.CheckState.Checked.value))
        lr2.addWidget(self._sl_line2)
        lr2.addWidget(self._lbl_line2)
        cl.addLayout(lr2)

        # Gece modu + video kaydı
        self._chk_night = QCheckBox("🌙 Gece Modu (kontrast iyileştirme)")
        cl.addWidget(self._chk_night)
        self._chk_record = QCheckBox("🎥 İşlenmiş Videoyu Kaydet (MP4)")
        cl.addWidget(self._chk_record)
        rl.addWidget(gc)

        # ── Tools ────────────────────────────────────────────────────────
        gt = QGroupBox("🛠  Araçlar")
        tl2 = QVBoxLayout(gt)
        tl2.setSpacing(6)

        # Multi-polygon
        pr = QHBoxLayout()
        self._btn_poly = QPushButton("🔷 Yeni Bölge Ekle")
        self._btn_poly.clicked.connect(self._start_new_polygon)
        pr.addWidget(self._btn_poly)
        self._btn_poly_clear = QPushButton("🗑 Tümünü Sil")
        self._btn_poly_clear.clicked.connect(self._clear_all_polygons)
        pr.addWidget(self._btn_poly_clear)
        tl2.addLayout(pr)

        self._poly_hint = QLabel("Sol tık: nokta | Sağ tık: tamamla")
        self._poly_hint.setStyleSheet("font-size:10px;color:#505070;")
        self._poly_hint.setVisible(False)
        tl2.addWidget(self._poly_hint)

        self._poly_info = QLabel("Bölgeler: Yok")
        self._poly_info.setStyleSheet("font-size:11px;color:#707088;")
        tl2.addWidget(self._poly_info)

        # Heatmap
        self._chk_hm = QCheckBox("🌡 Isı Haritası (Heatmap)")
        self._chk_hm.stateChanged.connect(self._toggle_hm)
        tl2.addWidget(self._chk_hm)

        # Pin window
        self._btn_pin = QPushButton("📌 Pencereyi Üste Sabitle")
        self._btn_pin.clicked.connect(self._pin_window)
        tl2.addWidget(self._btn_pin)

        # Location (Lat/Lon for dataset-compatible export)
        loc_grp = QHBoxLayout()
        loc_grp.setSpacing(4)
        loc_grp.addWidget(QLabel("📍 Enlem:"))
        self._inp_lat = QLineEdit()
        self._inp_lat.setPlaceholderText("41.015")
        self._inp_lat.setFixedWidth(80)
        loc_grp.addWidget(self._inp_lat)
        loc_grp.addWidget(QLabel("Boylam:"))
        self._inp_lon = QLineEdit()
        self._inp_lon.setPlaceholderText("29.010")
        self._inp_lon.setFixedWidth(80)
        loc_grp.addWidget(self._inp_lon)
        tl2.addLayout(loc_grp)

        # İnteraktif CARTO haritasından konum seçimi (yoğunluk noktalı)
        self._btn_map = QPushButton("🗺️ Haritadan Konum Seç")
        self._btn_map.clicked.connect(self._pick_from_map)
        tl2.addWidget(self._btn_map)

        self._lbl_geohash = QLabel("Geohash: (otomatik hesaplanır)")
        self._lbl_geohash.setStyleSheet("font-size:10px;color:#505070;")
        tl2.addWidget(self._lbl_geohash)

        # Hız kalibrasyonu: görüntüde bilinen mesafeyi işaretle → km/s
        self._btn_calib = QPushButton("📏 Hız Kalibrasyonu (2 nokta işaretle)")
        self._btn_calib.clicked.connect(self._start_calibration)
        tl2.addWidget(self._btn_calib)
        self._lbl_calib = QLabel("Kalibrasyon: yok (hızlar px/s)")
        self._lbl_calib.setStyleSheet("font-size:10px;color:#505070;")
        tl2.addWidget(self._lbl_calib)

        # CSV
        csv_r = QHBoxLayout()
        csv_r.addWidget(QLabel("📊 Rapor:"))
        self._cb_interval = QComboBox()
        self._cb_interval.addItems(["1 dk", "5 dk", "10 dk", "30 dk", "1 saat"])
        self._cb_interval.setCurrentIndex(1)
        csv_r.addWidget(self._cb_interval)
        tl2.addLayout(csv_r)
        self._chk_hourly = QCheckBox("🕐 Saatlik Topla (İBB uyumlu)")
        self._chk_hourly.setToolTip(
            "İşaretliyse CSV'de periyot satırları saat başlarına toplanır:\n"
            "araç sayıları toplanır, hızlar ortalanır → İBB veri seti gibi\n"
            "'10 Haziran 2026 saat 05:00' başına tek satır.")
        tl2.addWidget(self._chk_hourly)
        self._btn_csv = QPushButton("💾 CSV Rapor Kaydet (Dataset Uyumlu)")
        self._btn_csv.clicked.connect(self._export_csv)
        tl2.addWidget(self._btn_csv)

        # Canlı rapor önizleme: birikmiş periyot satırları
        tl2.addWidget(QLabel("📋 Birikmiş Rapor Satırları:"))
        self._tbl_report = QTableWidget(0, 3)
        self._tbl_report.setHorizontalHeaderLabels(["Saat", "Araç", "Ort. Hız"])
        self._tbl_report.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._tbl_report.verticalHeader().setVisible(False)
        self._tbl_report.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tbl_report.setFixedHeight(110)
        self._tbl_report.setStyleSheet(
            "QTableWidget { background:#0f0f16; border:1px solid #22222e; border-radius:6px;"
            " font-size:11px; } QHeaderView::section { background:#1a1a24; color:#8080a0;"
            " border:none; padding:3px; font-size:10px; }")
        tl2.addWidget(self._tbl_report)
        rl.addWidget(gt)

        # ── Başlat / Durdur (duraklat) / Devam Et + Bitir ────────────────
        bs = QHBoxLayout()
        self._btn_start = QPushButton("▶  BAŞLAT")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.clicked.connect(self._start)
        self._btn_pause_panel = QPushButton("⏸  DURDUR")
        self._btn_pause_panel.setEnabled(False)
        self._btn_pause_panel.clicked.connect(self._pause_from_panel)
        self._btn_resume_panel = QPushButton("⏯  DEVAM ET")
        self._btn_resume_panel.setEnabled(False)
        self._btn_resume_panel.clicked.connect(self._resume_from_panel)
        bs.addWidget(self._btn_start)
        bs.addWidget(self._btn_pause_panel)
        bs.addWidget(self._btn_resume_panel)
        rl.addLayout(bs)

        # Tam durdurma: analizi bitirir (yeni BAŞLAT sıfırdan başlar)
        self._btn_stop = QPushButton("■  BİTİR (Analizi Sonlandır)")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        rl.addWidget(self._btn_stop)

        # ── Density badge ────────────────────────────────────────────────
        self._density = DensityBadge()
        rl.addWidget(self._density)

        # ── Metrics ──────────────────────────────────────────────────────
        mg = QGridLayout()
        mg.setSpacing(5)
        self._c_act = MetricCard("Anlık Araç", "0", "#5858ff")
        self._c_fps = MetricCard("FPS", "0", "#22c55e")
        self._c_in  = MetricCard("Giren ↓", "0", "#38bdf8")
        self._c_out = MetricCard("Çıkan ↑", "0", "#f472b6")
        self._c_avg = MetricCard("30s Ort.", "0", "#a78bfa")
        self._c_spd = MetricCard("Anlık Hız (px/s)", "0", "#fb923c")
        self._c_park = MetricCard("🅿 Park Halinde", "0", "#94a3b8")
        self._c_sesspd = MetricCard("Oturum Ort. Hız", "0", "#22d3ee")
        mg.addWidget(self._c_act, 0, 0); mg.addWidget(self._c_fps, 0, 1)
        mg.addWidget(self._c_in, 1, 0);  mg.addWidget(self._c_out, 1, 1)
        mg.addWidget(self._c_avg, 2, 0); mg.addWidget(self._c_spd, 2, 1)
        mg.addWidget(self._c_park, 3, 0); mg.addWidget(self._c_sesspd, 3, 1)
        rl.addLayout(mg)

        self._lbl_types = QLabel("Araç Dağılımı: —")
        self._lbl_types.setStyleSheet("font-size:11px;color:#707088;padding:4px;")
        self._lbl_types.setWordWrap(True)
        rl.addWidget(self._lbl_types)

        # ── Log ──────────────────────────────────────────────────────────
        gl = QGroupBox("📋  Günlük")
        gll = QVBoxLayout(gl)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(100)
        gll.addWidget(self._log)
        rl.addWidget(gl)

        rl.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        scroll.setWidget(rp)
        root.addWidget(scroll, stretch=0)

    # ─────────────────────────────────────────────────────────────────────
    #  SOURCE — tabs
    # ─────────────────────────────────────────────────────────────────────

    def _on_tab(self, i):
        modes = [SOURCE_FILE, SOURCE_URL, SOURCE_URL]  # İBB Kamera sekmesi de URL kaynağıdır
        self._source_mode = modes[min(i, len(modes) - 1)]

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Video Seç", "",
            "Video (*.mp4 *.avi *.mkv *.mov *.ts *.wmv);;Tüm (*)"
        )
        if p:
            self._source_path = p
            n = p.replace("\\", "/").split("/")[-1]
            self._file_lbl.setText(f"📄 {n}")
            self._source_mode = SOURCE_FILE
            self._on_log(f"📂 {n}")

    def _set_url(self):
        u = self._url_inp.text().strip()
        if u:
            self._source_path = u
            short = f"🌐 {u[:45]}…" if len(u) > 45 else f"🌐 {u}"
            self._url_lbl.setText(short)
            self._source_mode = SOURCE_URL
            self._on_log(f"🌐 URL ayarlandı")

    def _open_ibb_map(self):
        """İBB trafik haritasını aç; kameraya tıklanınca yakalanan HLS yayınını kaynak yap."""
        try:
            from ibb_map import IBBMapDialog, WEBENGINE_OK
        except ImportError:
            self._on_log("⚠ ibb_map.py bulunamadı.")
            return
        if not WEBENGINE_OK:
            self._on_log("⚠ Harita için 'PyQt6-WebEngine' gerekli: pip install PyQt6-WebEngine")
            return
        dlg = IBBMapDialog(self)
        if dlg.exec() and dlg.stream_url:
            u = dlg.stream_url
            self._source_path = u
            self._source_mode = SOURCE_URL
            short = u if len(u) <= 48 else u[:45] + "…"
            self._ibb_lbl.setText(f"📹 {short}")
            # İBB HLS sunucusu için referer/UA başlıkları (FFMPEG açılışında okunur)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "user_agent;Mozilla/5.0"
                "|referer;https://uym.ibb.gov.tr/"
                "|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
            )
            self._on_log("🚦 İBB kamera yayını ayarlandı — ▶ BAŞLAT'a basın")

    # ─────────────────────────────────────────────────────────────────────
    #  WINDOW CAPTURE
    # ─────────────────────────────────────────────────────────────────────

    def _refresh_windows(self):
        """Refresh the window dropdown list."""
        from win_utils import get_visible_windows
        self._cb_windows.clear()
        my_title = self.windowTitle()
        for hwnd, title in get_visible_windows():
            # Skip our own window
            if title == my_title:
                continue
            self._cb_windows.addItem(f"{title}", hwnd)

    def _select_window_region(self):
        """User selects a sub-region within the chosen window."""
        idx = self._cb_windows.currentIndex()
        if idx < 0:
            self._on_log("⚠️ Pencere seçin!")
            return

        hwnd = self._cb_windows.currentData()
        title = self._cb_windows.currentText()

        from win_utils import bring_to_front, is_window_valid, get_window_rect

        if not is_window_valid(hwnd):
            self._on_log("❌ Pencere geçerli değil, yenileyin")
            self._refresh_windows()
            return

        self._win_hwnd = hwnd
        self._win_title = title

        # Store window position at selection time
        wl, wt, ww, wh = get_window_rect(hwnd)
        self._win_screen_pos = (wl, wt, ww, wh)

        self._on_log(f"🪟 Pencere: {title[:40]}… öne getiriliyor")

        # Bring target window to front
        bring_to_front(hwnd)
        _time.sleep(0.4)

        # Hide ourselves to avoid capturing our own window
        self.hide()
        _time.sleep(0.3)

        # Show the fullscreen region selector
        self._selector = ScreenSelector()
        self._selector.region_selected.connect(self._on_window_region)
        self._selector.destroyed.connect(self._show_after_select)
        self._selector.show()

    def _show_after_select(self):
        if not self.isVisible():
            self.show()
            self.activateWindow()

    def _on_window_region(self, screen_rect: QRect):
        """Convert screen coordinates to window-relative coordinates."""
        self.show()
        self.activateWindow()

        wl, wt, ww, wh = self._win_screen_pos

        # Calculate window-relative coordinates
        crop_x = screen_rect.x() - wl
        crop_y = screen_rect.y() - wt
        crop_w = screen_rect.width()
        crop_h = screen_rect.height()

        # Clamp to window bounds
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        crop_w = min(crop_w, ww - crop_x)
        crop_h = min(crop_h, wh - crop_y)

        if crop_w <= 10 or crop_h <= 10:
            self._on_log("⚠️ Seçilen bölge çok küçük!")
            return

        self._win_crop = (crop_x, crop_y, crop_w, crop_h)
        self._source_mode = SOURCE_WINDOW

        self._win_lbl.setText(
            f"🪟 {self._win_title[:25]}…\n"
            f"Bölge: {crop_w}×{crop_h} @ ({crop_x},{crop_y})"
        )
        self._on_log(
            f"✅ Pencere bölgesi: {crop_w}×{crop_h} "
            f"(Alt-Tab güvenli)"
        )

    # ─────────────────────────────────────────────────────────────────────
    #  MULTI-POLYGON
    # ─────────────────────────────────────────────────────────────────────

    def _start_new_polygon(self):
        if self._poly_drawing:
            self._finish_polygon()
            return

        self._poly_drawing = True
        self._current_poly = []
        zone_idx = len(self._all_polygons)
        color = ZONE_COLORS_HEX[zone_idx % len(ZONE_COLORS_HEX)]

        self._btn_poly.setText("✅ Çizimi Bitir (Sağ Tık)")
        self._btn_poly.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #cc4400,stop:1 #ff6622);"
            "border-color:#ff8844;color:#fff0e0;border-radius:8px;"
            "padding:8px 16px;font-weight:600;font-size:12px;"
        )
        self._poly_hint.setVisible(True)
        self._video_label.setCursor(Qt.CursorShape.CrossCursor)
        self._on_log(f"🔷 Bölge {zone_idx + 1} çiziliyor ({color})")

    def _map_to_frame(self, pos: QPoint):
        """Video etiketi koordinatını gerçek kare koordinatına çevirir."""
        pix = self._video_label.pixmap()
        if pix is None or self._frame_size == (0, 0):
            return None
        lw, lh = self._video_label.width(), self._video_label.height()
        pw, ph = pix.width(), pix.height()
        xo, yo = (lw - pw) / 2, (lh - ph) / 2
        px, py = pos.x() - xo, pos.y() - yo
        if px < 0 or py < 0 or px > pw or py > ph:
            return None
        fw, fh = self._frame_size
        return int(px / pw * fw), int(py / ph * fh)

    def _on_video_click(self, pos: QPoint):
        mapped = self._map_to_frame(pos)
        if mapped is None:
            return
        fx, fy = mapped
        self._current_poly.append((fx, fy))
        self._on_log(f"   📍 Nokta {len(self._current_poly)}: ({fx},{fy})")
        preview = self._all_polygons + (
            [self._current_poly] if len(self._current_poly) >= 2 else []
        )
        self._thread.set_polygons(preview)

    def _finish_polygon(self):
        self._poly_drawing = False
        self._btn_poly.setText("🔷 Yeni Bölge Ekle")
        self._btn_poly.setStyleSheet("")
        self._poly_hint.setVisible(False)
        self._video_label.setCursor(Qt.CursorShape.ArrowCursor)

        if len(self._current_poly) >= 3:
            self._all_polygons.append(self._current_poly)
            self._thread.set_polygons(self._all_polygons)
            self._on_log(
                f"✅ Bölge {len(self._all_polygons)} tamamlandı "
                f"({len(self._current_poly)} nokta)"
            )
        else:
            self._on_log("⚠️ En az 3 nokta gerekli")
            self._thread.set_polygons(self._all_polygons)

        self._current_poly = []
        self._update_poly_info()

    # ─────────────────────────────────────────────────────────────────────
    #  HIZ KALİBRASYONU (px/s → km/s)
    # ─────────────────────────────────────────────────────────────────────

    def _start_calibration(self):
        """Görüntüde gerçek mesafesi bilinen 2 nokta işaretletir (örn. şerit çizgileri)."""
        if self._video_label.pixmap() is None:
            self._on_log("⚠ Önce analizi başlatın — görüntü üzerinde işaretleme yapılır.")
            return
        self._calib_mode = True
        self._calib_pts = []
        self._btn_calib.setText("📏 2 noktaya tıklayın… (1/2)")
        self._video_label.setCursor(Qt.CursorShape.CrossCursor)
        self._on_log("📏 Kalibrasyon: gerçek mesafesini bildiğiniz 2 noktaya tıklayın "
                     "(örn. ardışık şerit çizgileri ≈ 3.5 m, direkler arası vb.)")

    def _on_calib_click(self, pos: QPoint):
        mapped = self._map_to_frame(pos)
        if mapped is None:
            return
        self._calib_pts.append(mapped)
        if len(self._calib_pts) == 1:
            self._btn_calib.setText("📏 2 noktaya tıklayın… (2/2)")
            return
        # İki nokta tamam → piksel mesafesi + gerçek mesafe sor
        self._calib_mode = False
        self._video_label.setCursor(Qt.CursorShape.ArrowCursor)
        self._btn_calib.setText("📏 Hız Kalibrasyonu (2 nokta işaretle)")
        (x1, y1), (x2, y2) = self._calib_pts
        px_dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if px_dist < 5:
            self._on_log("⚠ Noktalar çok yakın — kalibrasyon iptal.")
            return
        meters, ok = QInputDialog.getDouble(
            self, "📏 Hız Kalibrasyonu",
            f"İşaretlenen iki nokta arasındaki GERÇEK mesafe kaç metre?\n"
            f"(Piksel mesafesi: {px_dist:.0f} px)",
            3.5, 0.1, 1000.0, 2)
        if not ok:
            self._on_log("📏 Kalibrasyon iptal edildi.")
            return
        self._m_per_px = meters / px_dist
        self._thread.analytics.set_calibration(self._m_per_px)
        self._lbl_calib.setText(
            f"Kalibrasyon: 1 px = {self._m_per_px:.4f} m → hızlar km/s")
        self._lbl_calib.setStyleSheet("font-size:10px;color:#22c55e;")
        self._on_log(f"✅ Kalibrasyon: {meters} m / {px_dist:.0f} px → "
                     f"hızlar artık km/s cinsinden (İBB ile karşılaştırılabilir)")

    def _clear_all_polygons(self):
        self._all_polygons = []
        self._current_poly = []
        self._poly_drawing = False
        self._btn_poly.setText("🔷 Yeni Bölge Ekle")
        self._btn_poly.setStyleSheet("")
        self._poly_hint.setVisible(False)
        self._video_label.setCursor(Qt.CursorShape.ArrowCursor)
        self._thread.set_polygons([])
        self._update_poly_info()
        self._on_log("🗑 Tüm bölgeler silindi")

    def _update_poly_info(self):
        n = len(self._all_polygons)
        if n == 0:
            self._poly_info.setText("Bölgeler: Yok")
        else:
            parts = []
            for i in range(n):
                c = ZONE_COLORS_HEX[i % len(ZONE_COLORS_HEX)]
                parts.append(f'<span style="color:{c}">● Bölge {i + 1}</span>')
            self._poly_info.setText("Bölgeler: " + "  ".join(parts))

    # ─────────────────────────────────────────────────────────────────────
    #  HEATMAP & PIN & CSV
    # ─────────────────────────────────────────────────────────────────────

    def _toggle_hm(self, st):
        en = st == Qt.CheckState.Checked.value
        self._thread.set_heatmap(en)
        self._on_log("🌡 Isı haritası " + ("AÇ" if en else "KAPALI"))

    def _pin_window(self):
        dlg = WindowPickerDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_hwnd:
            from win_utils import pin_window
            pin_window(dlg.selected_hwnd)
            self._on_log(f"📌 Pencere üste sabitlendi")

    def _pick_from_map(self):
        """CARTO haritasından tıklayarak enlem/boylam seç (CSV raporu bu konumla kaydedilir)."""
        try:
            from map_picker import MapPickerDialog, WEBENGINE_OK
        except ImportError:
            self._on_log("⚠ map_picker.py bulunamadı.")
            return
        if not WEBENGINE_OK:
            self._on_log("⚠ Harita için 'PyQt6-WebEngine' gerekli: pip install PyQt6-WebEngine")
            return
        dlg = MapPickerDialog(self)
        if dlg.exec() and dlg.picked_lat is not None:
            lat, lon = dlg.picked_lat, dlg.picked_lon
            self._inp_lat.setText(f"{lat:.5f}")
            self._inp_lon.setText(f"{lon:.5f}")
            gh = ""
            try:
                import pygeohash as pgh
                gh = pgh.encode(lat, lon, precision=6)
                self._lbl_geohash.setText(f"Geohash: {gh}")
            except ImportError:
                pass
            # Analiz çalışıyorsa konumu anında uygula (sonraki rapor satırları bu konumu kullanır)
            if self._thread.isRunning():
                self._thread.analytics.set_location(lat, lon)
            self._on_log(f"🗺️ Haritadan konum seçildi: {lat:.5f}, {lon:.5f}" + (f" → {gh}" if gh else ""))

    def _export_csv(self):
        from datetime import datetime as _dt
        default_name = f"traffic_density_{_dt.now().strftime('%Y_%m')}.csv"
        p, _ = QFileDialog.getSaveFileName(
            self, "CSV Kaydet", default_name, "CSV (*.csv);;Tüm (*)"
        )
        if p:
            try:
                hourly = self._chk_hourly.isChecked()
                self._thread.analytics.export_csv(p, hourly=hourly)
                self._on_log(f"💾 Rapor: {p}" + (" (saatlik toplandı — İBB uyumlu)" if hourly else ""))
            except Exception as e:
                self._on_log(f"❌ Rapor hatası: {e}")

    def _get_interval(self):
        return {0: 60, 1: 300, 2: 600, 3: 1800, 4: 3600}.get(
            self._cb_interval.currentIndex(), 300
        )

    # ─────────────────────────────────────────────────────────────────────
    #  ANALYSIS START / STOP
    # ─────────────────────────────────────────────────────────────────────

    def _start(self):
        if self._source_mode == SOURCE_SCREEN:
            if self._screen_rect.isEmpty():
                self._on_log("⚠️ Önce bölge seçin!")
                return
        elif not self._source_path:
            self._on_log("⚠️ Kaynak seçin!")
            return

        # Önceki oturumdan kaydedilmemiş rapor satırları varsa kullanıcıya sor
        # (BAŞLAT her zaman sıfırdan başlar; eski veriler sessizce silinmesin)
        prev_rows = len(self._thread.analytics._report_rows)
        if prev_rows > 0:
            ret = QMessageBox.question(
                self, "Önceki Oturum Verisi",
                f"Önceki oturumdan kaydedilmemiş {prev_rows} rapor satırı var.\n"
                f"Yeni analiz başlayınca bunlar silinecek.\n\n"
                f"Önce CSV olarak kaydetmek ister misiniz?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save)
            if ret == QMessageBox.StandardButton.Cancel:
                return
            if ret == QMessageBox.StandardButton.Save:
                self._export_csv()
            self._on_log(f"🗑 Önceki oturumun {prev_rows} satırı temizlendi.")

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_pause.setEnabled(True)
        self._btn_pause_panel.setEnabled(True)
        self._btn_resume_panel.setEnabled(False)
        self._is_paused = False
        self._btn_pause.setText("⏸  Duraklat")
        self._set_ctrls(False)

        t = self._thread
        t.source = self._source_path
        t.source_mode = self._source_mode
        t.screen_rect = QRect(self._screen_rect)
        t.window_hwnd = self._win_hwnd
        t.window_crop = self._win_crop
        t.model_variant = [
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"
        ][self._cb_model.currentIndex()]
        t.device = ["cuda", "cpu", "mps"][self._cb_device.currentIndex()]
        t.confidence = self._sl_conf.value() / 100.0
        t.line_ratio = self._sl_line.value() / 100.0
        t.line_ratio2 = (self._sl_line2.value() / 100.0
                         if self._chk_line2.isChecked() else -1.0)
        t.night_mode = self._chk_night.isChecked()
        t.record_video = self._chk_record.isChecked()
        t.report_interval = self._get_interval()
        t.set_polygons(self._all_polygons)
        t.set_heatmap(self._chk_hm.isChecked())
        # Set location from UI inputs
        try:
            lat = float(self._inp_lat.text().strip()) if self._inp_lat.text().strip() else 0.0
            lon = float(self._inp_lon.text().strip()) if self._inp_lon.text().strip() else 0.0
            t.analytics.set_location(lat, lon)
            if lat != 0.0 and lon != 0.0:
                gh = t.analytics._geohash
                self._lbl_geohash.setText(f"Geohash: {gh}" if gh else "Geohash: (pygeohash yüklenmedi)")
                self._on_log(f"📍 Konum: {lat}, {lon} → Geohash: {gh}")
            else:
                self._lbl_geohash.setText("Geohash: (konum girilmedi)")
        except ValueError:
            self._on_log("⚠️ Enlem/Boylam geçersiz! Örn: 41.015 / 29.010")
            self._lbl_geohash.setText("Geohash: (çevrim hatası)")

        # Kalibrasyon yeni oturuma taşınır (reset hız birimini sıfırlamaz)
        t.analytics.set_calibration(self._m_per_px)
        self._last_report_row_count = -1
        self._tbl_report.setRowCount(0)

        t.start()

    def _stop(self):
        self._thread.stop()
        self._on_log("⏳ Durduruluyor…")

    def _toggle_pause(self):
        if self._is_paused:
            self._thread.resume()
            self._is_paused = False
            self._btn_pause.setText("⏸  Duraklat")
            self._on_log("▶ Devam")
        else:
            self._thread.pause()
            self._is_paused = True
            self._btn_pause.setText("▶  Devam Et")
            self._on_log("⏸ Duraklat")
        # Sağ paneldeki Durdur/Devam Et butonlarını senkronize et
        self._btn_pause_panel.setEnabled(not self._is_paused)
        self._btn_resume_panel.setEnabled(self._is_paused)

    def _pause_from_panel(self):
        """⏸ DURDUR — analizi duraklatır, tüm sayaçlar ve video pozisyonu korunur."""
        if not self._is_paused:
            self._toggle_pause()

    def _resume_from_panel(self):
        """⏯ DEVAM ET — kaldığı yerden sürdürür."""
        if self._is_paused:
            self._toggle_pause()

    def _skip(self, s):
        if not self._is_seekable or self._total_frames <= 0:
            return
        tt = self._pt(self._lbl_time.text().split("/")[-1])
        if tt <= 0:
            return
        d = int(s * self._total_frames / tt)
        tgt = max(0, min(self._timeline.value() + d, self._total_frames))
        self._thread.seek_to(tgt)
        self._timeline.setValue(tgt)

    @staticmethod
    def _pt(s):
        p = s.strip().split(":")
        try:
            if len(p) == 3: return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
            if len(p) == 2: return int(p[0]) * 60 + int(p[1])
        except Exception:
            pass
        return 1.0

    def _on_finished(self):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._btn_pause_panel.setEnabled(False)
        self._btn_resume_panel.setEnabled(False)
        self._btn_back.setEnabled(False)
        self._btn_fwd.setEnabled(False)
        self._timeline.setEnabled(False)
        self._is_paused = False
        self._btn_pause.setText("⏸  Duraklat")
        self._mode_badge.setVisible(False)
        self._set_ctrls(True)

    def _set_ctrls(self, on):
        for w in (self._cb_model, self._cb_device, self._sl_conf, self._sl_line):
            w.setEnabled(on)

    def _on_slider_release(self):
        self._slider_dragging = False
        if self._is_seekable:
            self._thread.seek_to(self._timeline.value())

    # ─────────────────────────────────────────────────────────────────────
    #  SIGNAL HANDLERS
    # ─────────────────────────────────────────────────────────────────────

    def _on_progress(self, cf, tf, ct, tt, seekable):
        self._is_seekable = seekable
        self._total_frames = tf
        if seekable and tf > 0:
            self._timeline.setEnabled(True)
            self._btn_back.setEnabled(True)
            self._btn_fwd.setEnabled(True)
            self._timeline.setRange(0, tf)
            if not self._slider_dragging:
                self._timeline.setValue(cf)
            self._lbl_time.setText(f"{ct} / {tt}")
            self._badge("📄 DOSYA", "#0d2833", "#38bdf8")
        else:
            self._timeline.setEnabled(False)
            self._btn_back.setEnabled(False)
            self._btn_fwd.setEnabled(False)
            if self._source_mode == SOURCE_WINDOW:
                self._lbl_time.setText("PENCERE")
                self._badge("🪟 PENCERE", "#1a2d0d", "#84cc40")
            elif self._source_mode == SOURCE_SCREEN:
                self._lbl_time.setText("EKRAN")
                self._badge("📷 EKRAN", "#2d0d33", "#c084fc")
            else:
                self._lbl_time.setText("CANLI")
                self._badge("📡 CANLI", "#330d0d", "#ef4444")

    def _badge(self, t, bg, fg):
        self._mode_badge.setText(t)
        self._mode_badge.setVisible(True)
        self._mode_badge.setStyleSheet(
            f"background-color:{bg};border:1px solid {fg};border-radius:6px;"
            f"color:{fg};font-size:11px;font-weight:700;padding:2px 10px;"
        )

    def _on_frame(self, q_img: QImage, metrics: dict):
        self._frame_size = (q_img.width(), q_img.height())
        pix = QPixmap.fromImage(q_img)
        scaled = pix.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

        self._c_act.set_value(str(metrics.get("active_count", 0)))
        self._c_fps.set_value(str(metrics.get("fps", 0)))
        self._c_in.set_value(str(metrics.get("count_in", 0)))
        self._c_out.set_value(str(metrics.get("count_out", 0)))
        self._c_avg.set_value(str(metrics.get("avg_30s", 0)))
        self._c_spd.set_value(str(metrics.get("avg_speed_px_s", 0)))
        self._c_park.set_value(str(metrics.get("parked_count", 0)))
        self._c_sesspd.set_value(str(metrics.get("session_avg_speed", 0)))

        # Hız birimi (kalibrasyon yapıldıysa km/s)
        unit = metrics.get("speed_unit", "px/s")
        self._c_spd.set_title(f"Anlık Hız ({unit})")
        self._c_sesspd.set_title(f"Oturum Ort. Hız ({unit})")

        # Canlı rapor önizleme: yeni periyot satırı geldiyse tabloyu yenile
        rows = self._thread.analytics._report_rows
        if len(rows) != self._last_report_row_count:
            self._last_report_row_count = len(rows)
            show = rows[-5:]
            self._tbl_report.setRowCount(len(show))
            for i, r in enumerate(show):
                self._tbl_report.setItem(i, 0, QTableWidgetItem(r["date_time"][11:16]))
                self._tbl_report.setItem(i, 1, QTableWidgetItem(str(r["vehicle_count"])))
                self._tbl_report.setItem(i, 2, QTableWidgetItem(f"{r['avg_speed']:.1f}"))

        dl = metrics.get("density_label", "Düşük")
        self._density.update_level(dl)
        accent = {"Düşük": "#22c55e", "Orta": "#eab308", "Yüksek": "#ef4444"}
        self._c_act.set_accent(accent.get(dl, "#5858ff"))

        td = metrics.get("type_distribution", {})
        if td:
            self._lbl_types.setText(
                "Araç Dağılımı:  " + "  |  ".join(f"{k}: {v}" for k, v in td.items())
            )

    def _on_log(self, msg):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        if self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(3000)
        if self._capture_overlay:
            self._capture_overlay.close()
        event.accept()
