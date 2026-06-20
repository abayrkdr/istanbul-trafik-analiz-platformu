"""
app_data_manager.py — Kalıcı Veri Yöneticisi (PyQt6)
=====================================================
Parquet dosyalarını görüntüle, CSV ekle (append), veri sil, yedekle, doğrula.
Kullanım:  python app_data_manager.py
"""

import sys
import os
import shutil
import pandas as pd
import duckdb
import numpy as np
from datetime import datetime

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QTableView, QPushButton, QLabel, QComboBox,
    QFileDialog, QGroupBox, QGridLayout, QLineEdit, QTextEdit,
    QMessageBox, QProgressBar, QHeaderView, QSpinBox, QDateEdit,
    QSplitter, QFrame, QStatusBar, QScrollArea,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_parquet")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")

PARQUET_FILES = [
    "traffic_all.parquet",
    "summary_hourly.parquet",
    "summary_monthly.parquet",
    "summary_geo_hourly.parquet",
]

# Her parquet dosyası için beklenen zorunlu sütunlar ve tarih sütunu adı
# date_col: None → gün filtresi bu dosyaya uygulanamaz
PARQUET_SCHEMAS = {
    "traffic_all.parquet": {
        "required": {"date_time", "lat", "lon", "geohash", "avg_speed",
                     "vehicle_count", "year", "month", "hour", "day_of_week"},
        "optional": {"min_speed", "max_speed", "period_minutes"},
        "date_col": "date_time",
    },
    "summary_hourly.parquet": {
        "required": {"hour_ts", "hour", "total_vehicles", "avg_speed"},
        "optional": set(),
        "date_col": "hour_ts",
    },
    "summary_monthly.parquet": {
        "required": {"year", "month", "total_vehicles", "avg_speed"},
        "optional": set(),
        "date_col": None,
    },
    "summary_geo_hourly.parquet": {
        "required": {"lat", "lon", "geohash", "hour", "total_vehicles", "avg_speed"},
        "optional": set(),
        "date_col": None,
    },
}

# analytics.py CSV'sindeki sütun adını hedef parquet sütun adına çevirir
# (örn. masaüstü uygulaması vehicle_count çıkarır; summary dosyaları total_vehicles bekler)
COLUMN_RENAMES = {
    "summary_hourly.parquet":     {"vehicle_count": "total_vehicles"},
    "summary_monthly.parquet":    {"vehicle_count": "total_vehicles"},
    "summary_geo_hourly.parquet": {"vehicle_count": "total_vehicles"},
}

DARK_QSS = """
* { margin:0; padding:0; }
QWidget {
    background-color: #0e0e14; color: #d4d4dc;
    font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px;
}
QMainWindow { background-color: #0e0e14; }

QTabWidget::pane { border:1px solid #22222e; border-radius:8px; background:#13131a; top:-1px; }
QTabBar::tab {
    background:#13131a; border:1px solid #22222e; border-bottom:none;
    border-top-left-radius:8px; border-top-right-radius:8px;
    padding:8px 18px; margin-right:2px;
    color:#707090; font-weight:600; font-size:12px;
}
QTabBar::tab:selected { background:#1a1a24; color:#c0c0ff; border-color:#3838a0; }
QTabBar::tab:hover:!selected { background:#18181f; color:#a0a0c0; }

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
QPushButton#btn_danger {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6b1a1a,stop:1 #882222);
    border-color:#bb3030; color:#ffe0e0;
}
QPushButton#btn_success {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #166534,stop:1 #1a8a4a);
    border-color:#22aa60; color:#d0ffd0;
}

QTableView {
    background-color:#10101a; gridline-color:#1e1e2e;
    selection-background-color:#2a2a5a; selection-color:#ffffff;
    border:1px solid #22222e; border-radius:6px;
    font-size:11px;
}
QTableView::item { padding:4px; }
QHeaderView::section {
    background-color:#16161f; color:#a0a0cc; border:1px solid #22222e;
    padding:6px; font-weight:600; font-size:11px;
}

QLineEdit, QComboBox, QSpinBox, QDateEdit {
    background-color:#14141e; border:1px solid #28283a; border-radius:6px;
    padding:6px 10px; color:#d0d0e0; font-size:12px;
}
QLineEdit:focus, QComboBox:focus { border-color:#5050ff; }

QTextEdit {
    background-color:#0c0c12; border:1px solid #22222e; border-radius:6px;
    color:#a0a0b8; font-family:'Consolas','Courier New',monospace; font-size:11px;
}

QProgressBar {
    background-color:#14141e; border:1px solid #22222e; border-radius:6px;
    text-align:center; color:#ffffff; font-weight:600;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #667eea,stop:1 #764ba2);
    border-radius:5px;
}

QScrollBar:vertical { background:#0e0e14; width:8px; border-radius:4px; }
QScrollBar::handle:vertical { background:#28283a; border-radius:4px; min-height:30px; }
QScrollBar::handle:vertical:hover { background:#3a3a50; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background:none; border:none; }

QStatusBar { background-color:#0a0a10; color:#707090; font-size:11px; border-top:1px solid #1a1a2e; }
"""


# ═══════════════════════════════════════════════════════════════════════════
#  PANDAS TABLE MODEL
# ═══════════════════════════════════════════════════════════════════════════
class PandasModel(QAbstractTableModel):
    def __init__(self, df=None, page_size=500):
        super().__init__()
        self._full_df = df if df is not None else pd.DataFrame()
        self._page_size = page_size
        self._page = 0
        self._update_view()

    def _update_view(self):
        start = self._page * self._page_size
        end = start + self._page_size
        self._df = self._full_df.iloc[start:end].reset_index(drop=True)

    def set_dataframe(self, df):
        self.beginResetModel()
        self._full_df = df
        self._page = 0
        self._update_view()
        self.endResetModel()

    def set_page(self, page):
        total = self.total_pages()
        self.beginResetModel()
        self._page = max(0, min(page, total - 1))
        self._update_view()
        self.endResetModel()

    def total_pages(self):
        return max(1, (len(self._full_df) + self._page_size - 1) // self._page_size)

    @property
    def current_page(self):
        return self._page

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        val = self._df.iloc[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(val, float):
                return f"{val:.4f}" if abs(val) < 1000 else f"{val:,.1f}"
            return str(val)
        if role == Qt.ItemDataRole.ForegroundRole:
            return QColor("#d0d0e0")
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1 + self._page * self._page_size)


# ═══════════════════════════════════════════════════════════════════════════
#  BACKGROUND WORKER
# ═══════════════════════════════════════════════════════════════════════════
class DataWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task == "load":
                self._load()
            elif self.task == "append":
                self._append()
            elif self.task == "delete":
                self._delete()
        except Exception as e:
            self.finished.emit(False, f"Hata: {e}")

    def _load(self):
        filename = self.kwargs["filename"]
        where_clause = self.kwargs.get("where_clause", "")
        p = os.path.join(DATA_DIR, filename).replace("\\", "/")
        self.progress.emit(10, f"Yükleniyor: {filename}...")
        db = duckdb.connect()
        query = f"SELECT * FROM '{p}' {where_clause} LIMIT 50000"
        df = db.execute(query).df()
        db.close()
        self.progress.emit(100, f"Yüklendi: {len(df)} satır")
        self.kwargs["result_df"] = df
        self.finished.emit(True, f"✅ {filename}: {len(df)} satır yüklendi (max 50K)")

    def _append(self):
        filename = self.kwargs["filename"]
        csv_path = self.kwargs["csv_path"]
        p = os.path.join(DATA_DIR, filename)

        self.progress.emit(10, "CSV okunuyor...")
        new_df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # Sütun yeniden adlandırma (örn. vehicle_count → total_vehicles)
        renames = COLUMN_RENAMES.get(filename, {})
        if renames:
            new_df = new_df.rename(columns=renames)

        # Hedef şemada tanımlanmayan sütunları kaldır (period_minutes gibi)
        schema = PARQUET_SCHEMAS.get(filename, {})
        allowed = schema.get("required", set()) | schema.get("optional", set())
        if allowed:
            drop_cols = [c for c in new_df.columns if c not in allowed]
            if drop_cols:
                new_df = new_df.drop(columns=drop_cols)

        self.progress.emit(30, "Yedek alınıyor...")
        self._backup(filename)

        self.progress.emit(50, "Mevcut veri okunuyor...")
        db = duckdb.connect()
        existing = db.execute(f"SELECT * FROM '{p.replace(chr(92), '/')}'").df()
        db.close()

        # Yalnızca mevcut parquet'te bulunan sütunları tut (şema uyumu)
        shared_cols = [c for c in existing.columns if c in new_df.columns]
        new_df = new_df[shared_cols]

        self.progress.emit(70, "Birleştiriliyor...")
        combined = pd.concat([existing, new_df], ignore_index=True)

        self.progress.emit(90, "Kaydediliyor...")
        combined.to_parquet(p, index=False)

        self.progress.emit(100, "Tamamlandı!")
        self.finished.emit(True, f"✅ {len(new_df)} satır eklendi → Toplam: {len(combined)}")

    def _delete(self):
        filename = self.kwargs["filename"]
        filter_col = self.kwargs["filter_col"]
        filter_val = self.kwargs["filter_val"]
        p = os.path.join(DATA_DIR, filename)

        self.progress.emit(10, "Yedek alınıyor...")
        self._backup(filename)

        self.progress.emit(30, "Veri okunuyor...")
        db = duckdb.connect()
        df = db.execute(f"SELECT * FROM '{p.replace(chr(92), '/')}'").df()
        db.close()

        before = len(df)
        self.progress.emit(50, "Filtreleniyor...")
        if filter_col in df.columns:
            df = df[df[filter_col].astype(str) != str(filter_val)]
        after = len(df)

        self.progress.emit(80, "Kaydediliyor...")
        df.to_parquet(p, index=False)

        self.progress.emit(100, "Tamamlandı!")
        self.finished.emit(True, f"✅ {before - after} satır silindi. Kalan: {after}")

    def _backup(self, filename):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        src = os.path.join(DATA_DIR, filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(BACKUP_DIR, f"{filename}.{ts}.bak")
        if os.path.exists(src):
            shutil.copy2(src, dst)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════
class DataManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 İstanbul Trafik — Veri Yöneticisi")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(DARK_QSS)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 8, 12, 4)
        main_layout.setSpacing(8)

        # Header
        header = QLabel("📊 İstanbul Trafik Veri Yöneticisi")
        header.setStyleSheet("""
            font-size: 22px; font-weight: 800; padding: 8px;
            color: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #667eea,stop:1 #764ba2);
        """)
        main_layout.addWidget(header)

        # Tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # ── Tab 1: Görüntüle ─────────────────────────────────────────────
        tab_view = QWidget()
        vl = QVBoxLayout(tab_view)
        vl.setSpacing(8)

        # File selector
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("📁 Dosya:"))
        self._cb_file = QComboBox()
        self._cb_file.addItems(PARQUET_FILES)
        self._cb_file.setMinimumWidth(280)
        top_row.addWidget(self._cb_file)
        
        # Date Filters
        top_row.addWidget(QLabel("📅 Yıl:"))
        self._cb_filter_year = QComboBox()
        self._cb_filter_year.addItems(["Tümü", "2020", "2021", "2022", "2023", "2024", "2025"])
        top_row.addWidget(self._cb_filter_year)
        
        top_row.addWidget(QLabel("Ay:"))
        self._cb_filter_month = QComboBox()
        months_list = ["Tümü", "Ocak (1)", "Şubat (2)", "Mart (3)", "Nisan (4)", "Mayıs (5)", 
                       "Haziran (6)", "Temmuz (7)", "Ağustos (8)", "Eylül (9)", "Ekim (10)", 
                       "Kasım (11)", "Aralık (12)"]
        self._cb_filter_month.addItems(months_list)
        top_row.addWidget(self._cb_filter_month)
        
        top_row.addWidget(QLabel("Gün:"))
        self._cb_filter_day = QComboBox()
        days_list = ["Tümü"] + [str(i) for i in range(1, 32)]
        self._cb_filter_day.addItems(days_list)
        top_row.addWidget(self._cb_filter_day)
        
        btn_load = QPushButton("📥 Yükle")
        btn_load.setObjectName("btn_success")
        btn_load.clicked.connect(self._load_data)
        top_row.addWidget(btn_load)
        top_row.addStretch()

        self._lbl_info = QLabel("Dosya ve filtreleri seçip 'Yükle'ye basın")
        self._lbl_info.setStyleSheet("color:#8080a0; font-size:12px;")
        top_row.addWidget(self._lbl_info)
        vl.addLayout(top_row)

        # Table
        self._table_model = PandasModel()
        self._table_view = QTableView()
        self._table_view.setModel(self._table_model)
        self._table_view.horizontalHeader().setStretchLastSection(True)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.verticalHeader().setDefaultSectionSize(26)
        vl.addWidget(self._table_view, stretch=1)

        # Pagination
        page_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Önceki")
        self._btn_prev.clicked.connect(lambda: self._change_page(-1))
        self._btn_next = QPushButton("Sonraki ▶")
        self._btn_next.clicked.connect(lambda: self._change_page(1))
        self._lbl_page = QLabel("Sayfa: 1/1")
        self._lbl_page.setStyleSheet("color:#a0a0cc; font-weight:600;")
        page_row.addWidget(self._btn_prev)
        page_row.addWidget(self._lbl_page)
        page_row.addWidget(self._btn_next)
        page_row.addStretch()
        vl.addLayout(page_row)

        tabs.addTab(tab_view, "👁️  Görüntüle")

        # ── Tab 2: CSV İçe Aktar ─────────────────────────────────────────
        tab_import = QWidget()
        il = QVBoxLayout(tab_import)
        il.setSpacing(10)

        imp_grp = QGroupBox("📂 CSV Dosyası İçe Aktar (Append)")
        imp_gl = QVBoxLayout(imp_grp)

        imp_row1 = QHBoxLayout()
        imp_row1.addWidget(QLabel("Hedef:"))
        self._cb_target = QComboBox()
        self._cb_target.addItems(PARQUET_FILES)
        self._cb_target.currentTextChanged.connect(self._on_target_changed)
        imp_row1.addWidget(self._cb_target)
        imp_gl.addLayout(imp_row1)

        imp_row2 = QHBoxLayout()
        self._lbl_csv = QLabel("CSV: Seçilmedi")
        self._lbl_csv.setStyleSheet("color:#8080a0;")
        imp_row2.addWidget(self._lbl_csv, stretch=1)
        btn_browse = QPushButton("📂 CSV Gözat")
        btn_browse.clicked.connect(self._browse_csv)
        imp_row2.addWidget(btn_browse)
        imp_gl.addLayout(imp_row2)

        # Validation preview
        self._txt_validation = QTextEdit()
        self._txt_validation.setMaximumHeight(120)
        self._txt_validation.setReadOnly(True)
        self._txt_validation.setPlaceholderText("CSV seçildiğinde doğrulama sonuçları burada gösterilir...")
        imp_gl.addWidget(self._txt_validation)

        btn_append = QPushButton("✅ Veriye Ekle (Append + Yedekle)")
        btn_append.setObjectName("btn_success")
        btn_append.clicked.connect(self._append_csv)
        imp_gl.addWidget(btn_append)

        il.addWidget(imp_grp)

        # Progress
        self._progress = QProgressBar()
        self._progress.setValue(0)
        il.addWidget(self._progress)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color:#a0a0cc; font-size:12px; padding:4px;")
        il.addWidget(self._lbl_status)
        il.addStretch()

        tabs.addTab(tab_import, "📥  İçe Aktar")

        # ── Tab 3: Sil / Temizle ─────────────────────────────────────────
        tab_delete = QWidget()
        dl = QVBoxLayout(tab_delete)
        dl.setSpacing(10)

        del_grp = QGroupBox("🗑️ Veri Sil (Filtre ile)")
        del_gl = QVBoxLayout(del_grp)

        del_row1 = QHBoxLayout()
        del_row1.addWidget(QLabel("Dosya:"))
        self._cb_del_file = QComboBox()
        self._cb_del_file.addItems(PARQUET_FILES)
        del_row1.addWidget(self._cb_del_file)
        del_gl.addLayout(del_row1)

        del_row2 = QHBoxLayout()
        del_row2.addWidget(QLabel("Sütun:"))
        self._inp_del_col = QLineEdit()
        self._inp_del_col.setPlaceholderText("Örn: geohash, year, month")
        del_row2.addWidget(self._inp_del_col)
        del_row2.addWidget(QLabel("Değer:"))
        self._inp_del_val = QLineEdit()
        self._inp_del_val.setPlaceholderText("Örn: sxk9h5, 2025, 6")
        del_row2.addWidget(self._inp_del_val)
        del_gl.addLayout(del_row2)

        del_warn = QLabel("⚠️ Bu işlem geri alınabilir (yedek otomatik alınır). İlgili sütun ve değerdeki TÜM satırlar silinir.")
        del_warn.setStyleSheet("color:#f97316; font-size:11px;")
        del_warn.setWordWrap(True)
        del_gl.addWidget(del_warn)

        btn_delete = QPushButton("🗑️ Sil (Yedekle + Filtrele)")
        btn_delete.setObjectName("btn_danger")
        btn_delete.clicked.connect(self._delete_data)
        del_gl.addWidget(btn_delete)

        dl.addWidget(del_grp)

        # ── Yedekleri Göster ──────────────────────────────────────────────
        bak_grp = QGroupBox("💾 Yedekler (Geri Al)")
        bak_gl = QVBoxLayout(bak_grp)

        self._bak_list = QTextEdit()
        self._bak_list.setReadOnly(True)
        self._bak_list.setMaximumHeight(150)
        bak_gl.addWidget(self._bak_list)

        bak_row = QHBoxLayout()
        btn_refresh_bak = QPushButton("🔄 Yedekleri Yenile")
        btn_refresh_bak.clicked.connect(self._refresh_backups)
        bak_row.addWidget(btn_refresh_bak)

        btn_restore = QPushButton("♻️ Son Yedeği Geri Yükle")
        btn_restore.clicked.connect(self._restore_backup)
        bak_row.addWidget(btn_restore)
        bak_row.addStretch()
        bak_gl.addLayout(bak_row)

        dl.addWidget(bak_grp)

        self._progress_del = QProgressBar()
        self._progress_del.setValue(0)
        dl.addWidget(self._progress_del)

        self._lbl_del_status = QLabel("")
        self._lbl_del_status.setStyleSheet("color:#a0a0cc; font-size:12px; padding:4px;")
        dl.addWidget(self._lbl_del_status)
        dl.addStretch()

        tabs.addTab(tab_delete, "🗑️  Sil / Geri Al")

        # ── Tab 4: Genel Bilgi ───────────────────────────────────────────
        tab_info = QWidget()
        info_l = QVBoxLayout(tab_info)
        info_l.setSpacing(10)

        self._txt_info = QTextEdit()
        self._txt_info.setReadOnly(True)
        info_l.addWidget(self._txt_info)

        btn_scan = QPushButton("🔍 Tüm Dosyaları Tara")
        btn_scan.setObjectName("btn_success")
        btn_scan.clicked.connect(self._scan_files)
        info_l.addWidget(btn_scan)

        tabs.addTab(tab_info, "ℹ️  Genel Bilgi")

        # Status bar
        self.statusBar().showMessage("Hazır — Dosya seçip işlem yapabilirsiniz")

        # Internal state
        self._csv_path = None
        self._worker = None

        # Initial scan
        self._scan_files()
        self._refresh_backups()

    # ─────────────────────────────────────────────────────────────────────
    #  TAB 1: GÖRÜNTÜLE
    # ─────────────────────────────────────────────────────────────────────

    def _load_data(self):
        filename = self._cb_file.currentText()
        year_val = self._cb_filter_year.currentText()
        month_val = self._cb_filter_month.currentText()
        day_val = self._cb_filter_day.currentText()
        
        # Build dynamic where clause
        conditions = []
        if year_val != "Tümü":
            conditions.append(f"year = {int(year_val)}")
            
        if month_val != "Tümü":
            try:
                m_num = int(month_val.split("(")[1].replace(")", ""))
                conditions.append(f"month = {m_num}")
            except Exception:
                pass
                
        if day_val != "Tümü":
            d_num = int(day_val)
            date_col = PARQUET_SCHEMAS.get(filename, {}).get("date_col")
            if date_col:
                conditions.append(f"EXTRACT(DAY FROM {date_col}) = {d_num}")
            else:
                self._lbl_info.setText(
                    f"⚠️ '{filename}' için gün filtresi desteklenmiyor — görmezden gelindi"
                )
            
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
            
        self._lbl_info.setText("Yükleniyor...")
        self.statusBar().showMessage(f"Yükleniyor: {filename} {where_clause}")
        self._worker = DataWorker("load", filename=filename, where_clause=where_clause, result_df=None)
        self._worker.progress.connect(lambda v, m: self._lbl_info.setText(m))
        self._worker.finished.connect(self._on_load_finished)
        self._worker.start()

    def _on_load_finished(self, ok, msg):
        if ok and self._worker and hasattr(self._worker, 'kwargs'):
            df = self._worker.kwargs.get("result_df")
            if df is not None:
                self._table_model.set_dataframe(df)
                self._update_page_label()
        self._lbl_info.setText(msg)
        self.statusBar().showMessage(msg)

    def _change_page(self, delta):
        new_page = self._table_model.current_page + delta
        self._table_model.set_page(new_page)
        self._update_page_label()

    def _update_page_label(self):
        cur = self._table_model.current_page + 1
        total = self._table_model.total_pages()
        self._lbl_page.setText(f"Sayfa: {cur}/{total}")

    # ─────────────────────────────────────────────────────────────────────
    #  TAB 2: CSV İÇE AKTAR
    # ─────────────────────────────────────────────────────────────────────

    def _browse_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "CSV Seç", "", "CSV (*.csv);;Tüm (*)")
        if p:
            self._csv_path = p
            name = os.path.basename(p)
            self._lbl_csv.setText(f"📄 {name}")
            self._validate_csv(p)

    def _on_target_changed(self, _):
        """Hedef dosya değiştiğinde mevcut CSV'yi yeni şemaya göre yeniden doğrula."""
        if self._csv_path:
            self._validate_csv(self._csv_path)

    def _validate_csv(self, path):
        """CSV dosyasını hedef parquet şemasına göre doğrular."""
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", nrows=100)
            target = self._cb_target.currentText()
            schema = PARQUET_SCHEMAS.get(target, {})
            required = schema.get("required", set())
            optional = schema.get("optional", set())
            renames = COLUMN_RENAMES.get(target, {})

            csv_cols = set(df.columns)
            # Rename önizlemesi: analytics.py sütun adları hedef adlara çevrilmiş gibi hesapla
            effective_cols = {renames.get(c, c) for c in csv_cols}

            lines = []
            lines.append(f"📄 Dosya: {os.path.basename(path)}")
            lines.append(f"🎯 Hedef: {target}")
            lines.append(f"📊 CSV Sütunları: {', '.join(sorted(csv_cols))}")
            lines.append(f"📏 Satır Sayısı (ilk 100): {len(df)}")

            warnings = []
            errors = []

            # Zorunlu sütun kontrolü
            missing = required - effective_cols
            if missing:
                errors.append(f"❌ Eksik zorunlu sütunlar: {', '.join(sorted(missing))}")

            # Bilinmeyen ekstra sütunlar (zorunlu + optional dışında)
            known = required | optional | set(renames.values())
            extra = effective_cols - known
            if extra:
                warnings.append(f"⚠️ Bilinmeyen/ekstra sütunlar (append'de tutulacak): {', '.join(sorted(extra))}")

            # Sütun yeniden adlandırma bildirimi
            active_renames = {k: v for k, v in renames.items() if k in csv_cols}
            if active_renames:
                rename_info = ", ".join(f"{k}→{v}" for k, v in active_renames.items())
                lines.append(f"🔄 Otomatik yeniden adlandırma: {rename_info}")

            # period_minutes bilgisi
            if "period_minutes" in csv_cols and "period_minutes" not in (required | optional):
                warnings.append("ℹ️ 'period_minutes' bu dosyada desteklenmiyor — append sırasında kaldırılacak")

            # Değer aralığı kontrolleri
            spd_col = "avg_speed" if "avg_speed" in df.columns else None
            if spd_col:
                max_spd = df[spd_col].max()
                min_spd = df[spd_col].min()
                if max_spd > 200:
                    warnings.append(f"⚠️ Aşırı yüksek hız: {max_spd:.0f} km/s (beklenen max ~150)")
                if min_spd < 0:
                    warnings.append(f"⚠️ Negatif hız: {min_spd:.0f} km/s")

            for vc in ("vehicle_count", "total_vehicles"):
                if vc in df.columns:
                    max_v = df[vc].max()
                    if max_v > 50000:
                        warnings.append(f"⚠️ Aşırı yüksek araç sayısı: {max_v:,.0f}")
                    break

            if "lat" in df.columns:
                lat_min, lat_max = df["lat"].min(), df["lat"].max()
                if lat_min < 40 or lat_max > 42:
                    warnings.append(f"⚠️ Enlem İstanbul dışında: ({lat_min:.3f}, {lat_max:.3f})")
            if "lon" in df.columns:
                lon_min, lon_max = df["lon"].min(), df["lon"].max()
                if lon_min < 27 or lon_max > 30:
                    warnings.append(f"⚠️ Boylam İstanbul dışında: ({lon_min:.3f}, {lon_max:.3f})")

            if errors:
                lines.append("\n🔴 HATALAR (append engellenmez ama veri bozulabilir):")
                lines.extend(errors)
            if warnings:
                lines.append("\n🟡 UYARILAR:")
                lines.extend(warnings)
            if not errors and not warnings:
                lines.append("\n✅ Doğrulama: Sorun bulunamadı — şema uyumlu!")

            self._txt_validation.setText("\n".join(lines))
        except Exception as e:
            self._txt_validation.setText(f"❌ CSV okunamadı: {e}")

    def _append_csv(self):
        if not self._csv_path:
            QMessageBox.warning(self, "Uyarı", "Önce CSV dosyası seçin!")
            return

        filename = self._cb_target.currentText()
        reply = QMessageBox.question(
            self, "Onay",
            f"'{os.path.basename(self._csv_path)}' dosyası '{filename}' içine eklenecek.\n"
            f"Otomatik yedek alınacak.\n\nDevam etmek istiyor musunuz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._progress.setValue(0)
        self._lbl_status.setText("İşlem başlıyor...")
        self._worker = DataWorker("append", filename=filename, csv_path=self._csv_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_append_finished)
        self._worker.start()

    def _on_progress(self, value, msg):
        self._progress.setValue(value)
        self._lbl_status.setText(msg)
        self.statusBar().showMessage(msg)

    def _on_append_finished(self, ok, msg):
        self._progress.setValue(100 if ok else 0)
        self._lbl_status.setText(msg)
        self.statusBar().showMessage(msg)
        if ok:
            QMessageBox.information(self, "Başarılı", msg)
            self._refresh_backups()
            self._scan_files()

    # ─────────────────────────────────────────────────────────────────────
    #  TAB 3: SİL / GERİ AL
    # ─────────────────────────────────────────────────────────────────────

    def _delete_data(self):
        filename = self._cb_del_file.currentText()
        col = self._inp_del_col.text().strip()
        val = self._inp_del_val.text().strip()

        if not col or not val:
            QMessageBox.warning(self, "Uyarı", "Sütun ve değer girilmeli!")
            return

        reply = QMessageBox.question(
            self, "⚠️ DİKKAT — Veri Silme",
            f"'{filename}' dosyasından\n"
            f"Sütun: '{col}' = '{val}' olan TÜM satırlar silinecek.\n"
            f"Yedek otomatik alınacak.\n\nEmin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._progress_del.setValue(0)
        self._lbl_del_status.setText("Silme işlemi başlıyor...")
        self._worker = DataWorker("delete", filename=filename, filter_col=col, filter_val=val)
        self._worker.progress.connect(lambda v, m: (self._progress_del.setValue(v), self._lbl_del_status.setText(m)))
        self._worker.finished.connect(self._on_delete_finished)
        self._worker.start()

    def _on_delete_finished(self, ok, msg):
        self._progress_del.setValue(100 if ok else 0)
        self._lbl_del_status.setText(msg)
        self.statusBar().showMessage(msg)
        if ok:
            QMessageBox.information(self, "Tamamlandı", msg)
            self._refresh_backups()
            self._scan_files()

    def _refresh_backups(self):
        lines = []
        if os.path.exists(BACKUP_DIR):
            files = sorted(os.listdir(BACKUP_DIR), reverse=True)[:20]
            for f in files:
                fp = os.path.join(BACKUP_DIR, f)
                size = os.path.getsize(fp) / (1024 * 1024)
                lines.append(f"💾 {f}  ({size:.1f} MB)")
        if not lines:
            lines.append("Henüz yedek yok.")
        self._bak_list.setText("\n".join(lines))

    def _restore_backup(self):
        if not os.path.exists(BACKUP_DIR):
            QMessageBox.warning(self, "Uyarı", "Yedek klasörü bulunamadı!")
            return

        files = sorted(os.listdir(BACKUP_DIR), reverse=True)
        if not files:
            QMessageBox.warning(self, "Uyarı", "Yedek dosya bulunamadı!")
            return

        latest = files[0]
        # Extract original filename: "traffic_all.parquet.20260606_123456.bak" -> "traffic_all.parquet"
        parts = latest.rsplit(".", 3)
        if len(parts) >= 3:
            original_name = parts[0] + "." + parts[1]  # e.g. traffic_all.parquet
        else:
            QMessageBox.warning(self, "Hata", f"Yedek dosya adı tanınamadı: {latest}")
            return

        reply = QMessageBox.question(
            self, "Geri Yükleme",
            f"Son yedek: {latest}\n→ {original_name} olarak geri yüklenecek.\n\nDevam?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        src = os.path.join(BACKUP_DIR, latest)
        dst = os.path.join(DATA_DIR, original_name)
        shutil.copy2(src, dst)
        QMessageBox.information(self, "Başarılı", f"✅ {original_name} geri yüklendi!")
        self._scan_files()
        self.statusBar().showMessage(f"Geri yüklendi: {original_name}")

    # ─────────────────────────────────────────────────────────────────────
    #  TAB 4: GENEL BİLGİ
    # ─────────────────────────────────────────────────────────────────────

    def _scan_files(self):
        lines = ["═" * 60, "📊 PARQUET DOSYA RAPORU", "═" * 60, ""]
        total_size = 0
        total_rows = 0

        for f in PARQUET_FILES:
            fp = os.path.join(DATA_DIR, f)
            if os.path.exists(fp):
                size = os.path.getsize(fp) / (1024 * 1024)
                total_size += size
                try:
                    db = duckdb.connect()
                    count = db.execute(f"SELECT COUNT(*) FROM '{fp.replace(chr(92), '/')}'").fetchone()[0]
                    cols = db.execute(f"DESCRIBE SELECT * FROM '{fp.replace(chr(92), '/')}'").df()
                    db.close()
                    total_rows += count
                    lines.append(f"📁 {f}")
                    lines.append(f"   Boyut: {size:.1f} MB | Satır: {count:,}")
                    lines.append(f"   Sütunlar: {', '.join(cols['column_name'].tolist())}")
                    lines.append("")
                except Exception as e:
                    lines.append(f"📁 {f} — ❌ Okunamadı: {e}")
                    lines.append("")
            else:
                lines.append(f"📁 {f} — ⚠️ Dosya bulunamadı")
                lines.append("")

        lines.append("─" * 60)
        lines.append(f"📦 Toplam Boyut: {total_size:.1f} MB")
        lines.append(f"📊 Toplam Satır: {total_rows:,}")

        # Backup info
        if os.path.exists(BACKUP_DIR):
            bak_files = os.listdir(BACKUP_DIR)
            bak_size = sum(os.path.getsize(os.path.join(BACKUP_DIR, f)) for f in bak_files) / (1024 * 1024)
            lines.append(f"💾 Yedek Dosyaları: {len(bak_files)} adet ({bak_size:.1f} MB)")
        lines.append("═" * 60)

        self._txt_info.setText("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DataManagerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
