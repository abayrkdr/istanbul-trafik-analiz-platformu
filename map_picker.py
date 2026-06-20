"""
map_picker.py — Masaüstü uygulaması için interaktif konum seçme penceresi.
CARTO Dark Matter altlığı + dataset'ten gelen trafik yoğunluk noktaları
(web arayüzündeki İstanbul Haritası ile aynı görünüm).
Haritaya tıklanan koordinat enlem/boylam alanlarına aktarılır ve
CSV raporu o konumla (geohash dahil) kaydedilir.
"""

import os
import json

from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal, QUrl
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebChannel import QWebChannel
    WEBENGINE_OK = True
except ImportError:
    WEBENGINE_OK = False

try:
    import pygeohash as pgh
    GEOHASH_OK = True
except ImportError:
    GEOHASH_OK = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_density_points(max_points=4000):
    """data_parquet'ten nokta başına ortalama araç yoğunluğunu yükler.
    Dönen liste: [[lat, lon, renk], ...] — yeşil/sarı/kırmızı."""
    pq = os.path.join(BASE_DIR, "data_parquet", "summary_geo_hourly.parquet")
    if not os.path.exists(pq):
        return []
    df = None
    try:
        import duckdb
        q = (f"SELECT lat, lon, AVG(total_vehicles) AS tv "
             f"FROM '{pq.replace(chr(92), '/')}' GROUP BY lat, lon "
             f"ORDER BY tv DESC LIMIT {int(max_points)}")
        df = duckdb.connect().execute(q).fetchdf()
    except Exception:
        try:
            import pandas as pd
            df = pd.read_parquet(pq, columns=["lat", "lon", "total_vehicles"])
            df = (df.groupby(["lat", "lon"], as_index=False)["total_vehicles"]
                    .mean().rename(columns={"total_vehicles": "tv"}))
            df = df.nlargest(max_points, "tv")
        except Exception:
            return []
    if df is None or df.empty:
        return []
    # Eşikler: sabit 1/3 yerine p33/p66 persentil — dağılım çarpık olsa da
    # her renk bandına yaklaşık eşit sayıda nokta düşer.
    p33 = float(df["tv"].quantile(0.33))
    p66 = float(df["tv"].quantile(0.66))
    pts = []
    for r in df.itertuples(index=False):
        tv = float(r.tv)
        color = "#22c55e" if tv < p33 else "#eab308" if tv < p66 else "#ef4444"
        pts.append([round(float(r.lat), 5), round(float(r.lon), 5), color])
    return pts


_MAP_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<style>
  html,body,#map { height:100%; margin:0; background:#0d0d14; }
  .coord-chip {
    position:absolute; bottom:14px; left:14px; z-index:1000;
    background:rgba(10,14,26,0.90); color:#e2e8f0; padding:9px 16px;
    border:1px solid rgba(99,102,241,0.55); border-radius:12px;
    font:600 13px 'Segoe UI',sans-serif; pointer-events:none;
    box-shadow:0 8px 24px rgba(0,0,0,0.5);
  }
  .legend {
    position:absolute; bottom:14px; right:14px; z-index:1000;
    background:rgba(10,14,26,0.90); color:#94a3b8; padding:8px 14px;
    border:1px solid rgba(99,102,241,0.35); border-radius:12px;
    font:600 11px 'Segoe UI',sans-serif; pointer-events:none;
  }
  .legend i { display:inline-block; width:9px; height:9px; border-radius:50%; margin:0 3px 0 8px; }
  .leaflet-container { cursor: crosshair; }
</style></head>
<body>
<div id="map"></div>
<div class="coord-chip" id="chip">🖱️ Video konumunu haritadan seçin</div>
<div class="legend">🚦 Yoğunluk:<i style="background:#22c55e"></i>Akıcı<i style="background:#eab308"></i>Orta<i style="background:#ef4444"></i>Yoğun</div>
<script>
  var map = L.map('map', {zoomControl:true}).setView([41.015, 29.01], 10);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution:'&copy; OpenStreetMap &copy; CARTO', maxZoom:19
  }).addTo(map);

  // Trafik yoğunluk noktaları (canvas renderer — binlerce nokta için hızlı)
  var pts = __POINTS__;
  var canvas = L.canvas({padding: 0.3});
  pts.forEach(function(p){
      L.circleMarker([p[0], p[1]], {
          renderer: canvas, radius: 2.5, weight: 0,
          fillColor: p[2], fillOpacity: 0.85
      }).addTo(map);
  });

  var bridge = null;
  new QWebChannel(qt.webChannelTransport, function(ch){ bridge = ch.objects.bridge; });

  // Seçim işaretçisi: turkuaz odak halkası (web arayüzündekiyle aynı stil)
  var sel = [];
  map.on('click', function(e){
      var lat = e.latlng.lat, lon = e.latlng.lng;
      sel.forEach(function(l){ map.removeLayer(l); });
      sel = [
        L.circle([lat, lon], {radius:1100, weight:0, fillColor:'#22d3ee', fillOpacity:0.12}).addTo(map),
        L.circle([lat, lon], {radius:650, color:'#22d3ee', weight:2, fill:false}).addTo(map),
        L.circleMarker([lat, lon], {radius:7, color:'#ffffff', weight:2,
                                    fillColor:'#22d3ee', fillOpacity:1}).addTo(map)
      ];
      document.getElementById('chip').innerHTML =
          '📍 <b>' + lat.toFixed(5) + '</b>, <b>' + lon.toFixed(5) + '</b>';
      if (bridge) bridge.on_map_click(lat, lon);
  });
</script>
</body></html>"""


class _Bridge(QObject):
    """JS → Python köprüsü: haritadaki tıklamayı yakalar."""
    coord_picked = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def on_map_click(self, lat, lon):
        self.coord_picked.emit(lat, lon)


class MapPickerDialog(QDialog):
    """CARTO haritasından nokta seçtirir; onaylanınca picked_lat/picked_lon dolu döner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🗺️  Video Konumunu Haritadan Seç — İstanbul")
        self.resize(900, 640)
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

        self.picked_lat = None
        self.picked_lon = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)

        self._view = QWebEngineView()
        self._bridge = _Bridge()
        self._bridge.coord_picked.connect(self._on_pick)
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)
        self._view.page().setWebChannel(self._channel)

        html = _MAP_HTML.replace("__POINTS__", json.dumps(_load_density_points()))
        self._view.setHtml(html, QUrl("https://local.map/"))
        lay.addWidget(self._view, stretch=1)

        bottom = QHBoxLayout()
        self._lbl = QLabel("🖱️ Videonun çekildiği noktaya haritada tıklayın…")
        bottom.addWidget(self._lbl, stretch=1)
        btn_cancel = QPushButton("✕ Vazgeç")
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        self._btn_ok = QPushButton("✔ Bu Konumu Kullan")
        self._btn_ok.setObjectName("btn_ok")
        self._btn_ok.setEnabled(False)
        self._btn_ok.clicked.connect(self.accept)
        bottom.addWidget(self._btn_ok)
        lay.addLayout(bottom)

    def _on_pick(self, lat, lon):
        self.picked_lat, self.picked_lon = lat, lon
        gh = pgh.encode(lat, lon, precision=6) if GEOHASH_OK else "—"
        self._lbl.setText(f"📍 Seçilen: {lat:.5f}, {lon:.5f}   |   Geohash: {gh}")
        self._btn_ok.setEnabled(True)
