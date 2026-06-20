"""
ibb_map.py — İBB Trafik Yoğunluk Haritası (uym.ibb.gov.tr/yharita6) tarayıcı penceresi.
Haritadaki bir trafik kamerasına tıklanınca sayfanın açtığı HLS (m3u8) yayın
isteği ağ katmanında yakalanır ve analiz kaynağı olarak kullanılmak üzere döndürülür.
"""

from PyQt6.QtCore import QObject, pyqtSignal, QUrl
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import (
        QWebEngineUrlRequestInterceptor, QWebEngineProfile, QWebEnginePage,
    )
    WEBENGINE_OK = True
except ImportError:
    WEBENGINE_OK = False

IBB_MAP_URL = "https://uym.ibb.gov.tr/yharita6/"


if WEBENGINE_OK:

    class _StreamSniffer(QWebEngineUrlRequestInterceptor):
        """Sayfanın yaptığı tüm ağ isteklerini izler, HLS yayın adreslerini yakalar."""

        def __init__(self, emit_fn):
            super().__init__()
            self._emit = emit_fn  # thread-safe: sinyal emit (IO thread → GUI thread)

        def interceptRequest(self, info):
            url = info.requestUrl().toString()
            low = url.lower()
            if ".m3u8" in low or ".mpd" in low or ".flv" in low:
                self._emit(url)


class IBBMapDialog(QDialog):
    """İBB haritasını gömülü tarayıcıda açar; kamera yayını yakalanınca
    stream_url doldurulur ve '✔ Bu Yayını Kullan' aktifleşir."""

    _sig_stream = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🚦  İBB Trafik Kameraları — Yayın Yakalayıcı")
        self.resize(1180, 760)
        self.setStyleSheet("""
            QDialog { background:#0d0d14; }
            QLabel { color:#e2e8f0; font-size:13px; }
            QPushButton {
                background:#1b1b26; color:#e2e8f0; border:1px solid #2a2a3a;
                border-radius:8px; padding:8px 18px; font-weight:600;
            }
            QPushButton:hover { border-color:#6366f1; }
            QPushButton#btn_ok { background:#16321f; border-color:#22c55e; color:#22c55e; }
            QPushButton#btn_ok:disabled { background:#1b1b26; border-color:#2a2a3a; color:#505070; }
        """)

        self.stream_url = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Özel profil + ağ dinleyici (interceptor IO thread'inde çalışır,
        # sinyal emit'i Qt tarafından GUI thread'ine güvenle taşınır)
        self._view = QWebEngineView()
        self._profile = QWebEngineProfile(self)
        self._sniffer = _StreamSniffer(self._sig_stream.emit)
        self._profile.setUrlRequestInterceptor(self._sniffer)
        self._page = QWebEnginePage(self._profile, self)
        self._view.setPage(self._page)
        self._sig_stream.connect(self._on_stream)
        self._view.load(QUrl(IBB_MAP_URL))
        lay.addWidget(self._view, stretch=1)

        bottom = QHBoxLayout()
        self._lbl = QLabel("📡 Haritadan bir trafik kamerasına tıklayın — yayın otomatik yakalanır…")
        self._lbl.setWordWrap(True)
        bottom.addWidget(self._lbl, stretch=1)
        btn_cancel = QPushButton("✕ Kapat")
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        self._btn_ok = QPushButton("✔ Bu Yayını Kullan")
        self._btn_ok.setObjectName("btn_ok")
        self._btn_ok.setEnabled(False)
        self._btn_ok.clicked.connect(self.accept)
        bottom.addWidget(self._btn_ok)
        lay.addLayout(bottom)

    def _on_stream(self, url):
        # Master playlist'i tercih et: elimizde master varken chunklist ile ezme
        if self.stream_url:
            if url == self.stream_url:
                return
            if "chunklist" in url.lower() and "chunklist" not in self.stream_url.lower():
                return
        self.stream_url = url
        short = url if len(url) <= 90 else url[:87] + "…"
        self._lbl.setText(f"📹 Yayın yakalandı: {short}")
        self._btn_ok.setEnabled(True)
