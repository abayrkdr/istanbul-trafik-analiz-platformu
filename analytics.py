"""
analytics.py — In-memory analytics engine for traffic analysis.
Supports: multi-polygon ROI, heatmap, CSV export, crossing detection.
"""

import csv
import time
import datetime
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import pygeohash as pgh
    HAS_GEOHASH = True
except ImportError:
    HAS_GEOHASH = False


VEHICLE_CLASSES = {
    2: "Araba",
    3: "Motosiklet",
    5: "Otobüs",
    7: "Kamyon",
    1: "Bisiklet",
}
VEHICLE_CLASS_IDS = set(VEHICLE_CLASSES.keys())

# Colour palette for multiple polygon zones (BGR)
ZONE_COLORS_BGR = [
    (0, 220, 100),    # Green
    (255, 160, 40),   # Orange
    (100, 160, 255),  # Blue
    (255, 100, 255),  # Pink
    (0, 220, 220),    # Cyan
    (130, 100, 255),  # Purple
    (80, 255, 255),   # Yellow
    (255, 180, 180),  # Light pink
]

# Same palette in RGB hex for the UI
ZONE_COLORS_HEX = [
    "#64dc00", "#28a0ff", "#ff6428", "#ff64ff",
    "#dcdc00", "#ff6482", "#ffff50", "#b4b4ff",
]


# Park tespiti: bu süreden uzun sabit kalan araç "park halinde" sayılır
# (trafik ışığı beklemesi ~60-90 sn altında kalır, eşik bunun üstünde tutuldu)
PARK_SECONDS = 45.0
PARK_MOVE_PX = 18.0   # bu kadar piksel hareket sabitlik sayacını sıfırlar


@dataclass
class TrackState:
    class_id: int
    last_cy: float
    positions: list = field(default_factory=list)
    crossed: bool = False
    direction: Optional[str] = None
    anchor_pos: Optional[Tuple[float, float]] = None
    anchor_time: float = 0.0
    parked: bool = False
    crossed_lines: set = field(default_factory=set)


class TrafficAnalytics:
    def __init__(self):
        self.tracks: Dict[int, TrackState] = {}
        self.count_in = 0
        self.count_out = 0
        self.active_ids: set = set()
        self.type_counts: Dict[str, int] = defaultdict(int)

        self._window_30s: deque = deque()
        self._window_5m: deque = deque()
        self._window_1h: deque = deque()
        self._speed_samples: deque = deque(maxlen=200)

        self._frame_times: deque = deque(maxlen=60)
        self._last_frame_ts = 0.0

        # Heatmap
        self._heatmap: Optional[np.ndarray] = None
        self._heatmap_size: Tuple[int, int] = (0, 0)

        # CSV report (dataset-compatible)
        self._report_rows: List[Dict] = []
        self._report_interval = 300
        self._report_last_ts = 0.0
        self._report_period_types: Dict[str, int] = defaultdict(int)
        self._report_period_count = 0
        self._report_start_time = 0.0

        # Location (set from UI)
        self._lat: float = 0.0
        self._lon: float = 0.0
        self._geohash: str = ""

        # Speed tracking for min/max per period
        self._period_speeds: List[float] = []

        # Park halindeki araçlar + oturum geneli kesintisiz hız ortalaması
        self.parked_ids: set = set()
        self._session_speed_sum: float = 0.0
        self._session_speed_n: int = 0
        self._period_start_dt: Optional[datetime.datetime] = None

        # Hız kalibrasyonu: 1 piksel = kaç metre (0 = kalibre edilmedi → px/s)
        self._m_per_px: float = 0.0

    def set_calibration(self, m_per_px: float):
        """px/s hızları km/s'e çevirmek için metre/piksel oranını ayarlar."""
        self._m_per_px = max(0.0, float(m_per_px))

    @property
    def speed_unit(self) -> str:
        return "km/s" if self._m_per_px > 0 else "px/s"

    # ── FPS ──────────────────────────────────────────────────────────────

    def tick_fps(self):
        now = time.perf_counter()
        if self._last_frame_ts > 0:
            self._frame_times.append(now - self._last_frame_ts)
        self._last_frame_ts = now
        if self._report_start_time == 0.0:
            self._report_start_time = now
            self._report_last_ts = now
            self._period_start_dt = datetime.datetime.now()

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        return 1.0 / (sum(self._frame_times) / len(self._frame_times))

    # ── MAIN UPDATE ──────────────────────────────────────────────────────

    def update(self, detections: list, line_y: int, frame_h: int,
               frame_w: int = 0, polygons: list = None) -> Dict:
        """
        Parameters
        ----------
        polygons : list of list of (x,y)
            Multiple polygon zones. Each polygon is [(x1,y1), (x2,y2), ...].
            If any polygon is given, only vehicles inside at least one are counted.
        """
        import cv2
        now = time.perf_counter()
        self.active_ids.clear()

        # Tek çizgi (int) veya çoklu çizgi (liste) desteği
        line_ys = list(line_y) if isinstance(line_y, (list, tuple)) else [line_y]

        # Ensure heatmap buffer
        if frame_w > 0 and frame_h > 0:
            if self._heatmap is None or self._heatmap_size != (frame_h, frame_w):
                self._heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
                self._heatmap_size = (frame_h, frame_w)

        # Pre-build polygon arrays for fast checks
        poly_arrays = []
        if polygons:
            for poly in polygons:
                if len(poly) >= 3:
                    poly_arrays.append(np.array(poly, dtype=np.int32))

        for det in detections:
            tid = det["id"]
            cls = det["class_id"]
            cx = (det["x1"] + det["x2"]) / 2
            cy = (det["y1"] + det["y2"]) / 2

            # Polygon filter: must be inside at least one zone
            if poly_arrays:
                inside_any = False
                for pa in poly_arrays:
                    if cv2.pointPolygonTest(pa, (cx, cy), False) >= 0:
                        inside_any = True
                        break
                if not inside_any:
                    continue

            self.active_ids.add(tid)

            if tid not in self.tracks:
                self.tracks[tid] = TrackState(class_id=cls, last_cy=cy)

            ts = self.tracks[tid]
            prev_cy = ts.last_cy

            # Crossing detection (her çizgi için ayrı; araç her çizgiyi 1 kez sayılır)
            for li, ly_ in enumerate(line_ys):
                if li in ts.crossed_lines:
                    continue
                if prev_cy < ly_ <= cy:
                    ts.crossed_lines.add(li)
                    ts.crossed, ts.direction = True, "in"
                    self.count_in += 1
                    lbl = VEHICLE_CLASSES.get(cls, "Diğer")
                    self.type_counts[lbl] += 1
                    self._report_period_types[lbl] += 1
                    self._report_period_count += 1
                elif prev_cy > ly_ >= cy:
                    ts.crossed_lines.add(li)
                    ts.crossed, ts.direction = True, "out"
                    self.count_out += 1
                    lbl = VEHICLE_CLASSES.get(cls, "Diğer")
                    self.type_counts[lbl] += 1
                    self._report_period_types[lbl] += 1
                    self._report_period_count += 1

            # Park tespiti: anchor noktasından PARK_MOVE_PX'ten az hareketle
            # PARK_SECONDS'tan uzun kalan araç park kabul edilir
            if ts.anchor_pos is None:
                ts.anchor_pos, ts.anchor_time = (cx, cy), now
            else:
                moved = ((cx - ts.anchor_pos[0]) ** 2 + (cy - ts.anchor_pos[1]) ** 2) ** 0.5
                if moved > PARK_MOVE_PX:
                    ts.anchor_pos, ts.anchor_time = (cx, cy), now
                    ts.parked = False
                elif now - ts.anchor_time > PARK_SECONDS:
                    ts.parked = True
            if ts.parked:
                self.parked_ids.add(tid)
            else:
                self.parked_ids.discard(tid)

            # Speed — park halindeki araçlar hız denkleminden hariç tutulur;
            # ışıkta bekleyenler (kısa duraklamalar) dahil kalır
            ts.positions.append((cx, cy, now))
            if len(ts.positions) > 30:
                ts.positions = ts.positions[-30:]
            if len(ts.positions) >= 2 and not ts.parked:
                p0, p1 = ts.positions[0], ts.positions[-1]
                dt = p1[2] - p0[2]
                if dt > 0:
                    spd = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)**0.5 / dt
                    if self._m_per_px > 0:
                        spd *= self._m_per_px * 3.6  # px/s → km/s
                    self._speed_samples.append(spd)
                    self._period_speeds.append(spd)
                    self._session_speed_sum += spd
                    self._session_speed_n += 1

            ts.last_cy = cy

            # Heatmap blob
            if self._heatmap is not None:
                ix, iy = int(cx), int(cy)
                r = 20
                y0 = max(0, iy - r)
                y1 = min(self._heatmap_size[0], iy + r)
                x0 = max(0, ix - r)
                x1 = min(self._heatmap_size[1], ix + r)
                if y1 > y0 and x1 > x0:
                    self._heatmap[y0:y1, x0:x1] += 1.0

        # Purge stale
        stale = [
            tid for tid, ts in self.tracks.items()
            if tid not in self.active_ids and ts.positions
            and (now - ts.positions[-1][2]) > 2.0
        ]
        for tid in stale:
            del self.tracks[tid]
            self.parked_ids.discard(tid)

        # Windows (park halindeki araçlar trafik yoğunluğuna dahil edilmez)
        ac = len(self.active_ids - self.parked_ids)
        self._window_30s.append((now, ac))
        self._window_5m.append((now, ac))
        self._window_1h.append((now, ac))
        self._trim(self._window_30s, 30)
        self._trim(self._window_5m, 300)
        self._trim(self._window_1h, 3600)

        # Report
        if now - self._report_last_ts >= self._report_interval:
            self._flush_period(now)

        return self.snapshot(frame_h)

    # ── SNAPSHOT ─────────────────────────────────────────────────────────

    def snapshot(self, frame_h=720) -> Dict:
        ac = len(self.active_ids - self.parked_ids)
        ds = self._density_score(ac, frame_h)
        return {
            "active_count": ac,
            "parked_count": len(self.parked_ids),
            "count_in": self.count_in,
            "count_out": self.count_out,
            "count_total": self.count_in + self.count_out,
            "fps": round(self.fps, 1),
            "density_score": ds,
            "density_label": self._density_label(ds),
            "avg_speed_px_s": round(self._avg_speed(), 1),
            "session_avg_speed": round(self._session_avg_speed(), 1),
            "speed_unit": self.speed_unit,
            "avg_30s": self._window_avg(self._window_30s),
            "avg_5m": self._window_avg(self._window_5m),
            "avg_1h": self._window_avg(self._window_1h),
            "type_distribution": dict(self.type_counts),
        }

    # ── HEATMAP ──────────────────────────────────────────────────────────

    def get_heatmap_overlay(self, frame: np.ndarray, alpha=0.4) -> np.ndarray:
        import cv2
        if self._heatmap is None or self._heatmap.max() == 0:
            return frame
        h, w = frame.shape[:2]
        hm = cv2.GaussianBlur(self._heatmap.copy(), (51, 51), 0)
        mx = hm.max()
        if mx <= 0:
            return frame
        hm = (hm / mx * 255).astype(np.uint8)
        colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        if colored.shape[:2] != (h, w):
            colored = cv2.resize(colored, (w, h))
        mask = hm > 10
        mask3 = np.stack([mask]*3, axis=-1)
        out = frame.copy()
        out[mask3] = cv2.addWeighted(frame, 1-alpha, colored, alpha, 0)[mask3]
        return out

    # ── LOCATION ─────────────────────────────────────────────────────────

    def set_location(self, lat: float, lon: float):
        """Kameranın konumunu ayarlar. Geohash otomatik hesaplanır."""
        self._lat = lat
        self._lon = lon
        if HAS_GEOHASH and lat != 0.0 and lon != 0.0:
            self._geohash = pgh.encode(lat, lon, precision=6)
        else:
            self._geohash = ""

    # ── CSV (Dataset Uyumlu) ─────────────────────────────────────────────

    def set_report_interval(self, seconds: int):
        self._report_interval = max(10, seconds)

    def _flush_period(self, now: float):
        """Periyot verilerini dataset formatında (traffic_all.parquet uyumlu) kaydeder.

        Hız, son birkaç saniyenin anlık penceresi DEĞİL, periyot boyunca biriken
        TÜM örneklerin ortalamasıdır (duraklat/durdur anındaki 0'lar sonucu bozmaz).
        Periyotta hiç hız örneği yoksa son bilinen değere / oturum ortalamasına
        geri düşülür — CSV'ye asla yapay '0 hız' satırı yazılmaz.
        Tarih, periyodun gerçek başlangıç zamanıdır (dinamik)."""
        ts = self._period_start_dt or datetime.datetime.now()

        if self._period_speeds:
            avg_spd = round(sum(self._period_speeds) / len(self._period_speeds), 1)
            min_spd = round(min(self._period_speeds), 1)
            max_spd = round(max(self._period_speeds), 1)
        elif self._report_rows:
            avg_spd = self._report_rows[-1]["avg_speed"]
            min_spd = self._report_rows[-1]["min_speed"]
            max_spd = self._report_rows[-1]["max_speed"]
        else:
            s = round(self._session_avg_speed(), 1)
            avg_spd = min_spd = max_spd = s

        row = {
            "date_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "lat": self._lat,
            "lon": self._lon,
            "geohash": self._geohash,
            "min_speed": min_spd,
            "max_speed": max_spd,
            "avg_speed": avg_spd,
            "vehicle_count": self._report_period_count,
            "year": ts.year,
            "month": ts.month,
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            # Periyodun gerçek süresi (dk) — web'de saatlik orana çevirme için
            "period_minutes": round(max(now - self._report_last_ts, 30) / 60, 1),
        }
        self._report_rows.append(row)
        self._report_period_types.clear()
        self._report_period_count = 0
        self._period_speeds.clear()
        self._period_start_dt = datetime.datetime.now()
        self._report_last_ts = now

    def export_csv(self, filepath: str, hourly: bool = False):
        """CSV'yi traffic_all.parquet formatında dışa aktarır.
        hourly=True: periyot satırları saat başlarına toplanır (İBB veri seti
        gibi '... saat 05:00' başına tek satır — araçlar toplanır, hızlar ortalanır)."""
        # Yalnızca içinde gerçek veri olan açık periyodu kapat (boş satır eklenmez)
        if self._report_period_count > 0 or self._period_speeds:
            self._flush_period(time.perf_counter())
        if not self._report_rows:
            return
        rows = self._report_rows
        if hourly:
            rows = self._aggregate_hourly(rows)
        fieldnames = [
            "date_time", "lat", "lon", "geohash",
            "min_speed", "max_speed", "avg_speed", "vehicle_count",
            "year", "month", "hour", "day_of_week", "period_minutes",
        ]
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    @staticmethod
    def _aggregate_hourly(rows: List[Dict]) -> List[Dict]:
        """Periyot satırlarını İBB formatında saatlik satırlara toplar."""
        buckets: Dict[str, List[Dict]] = {}
        for r in rows:
            key = r["date_time"][:13]  # 'YYYY-MM-DD HH'
            buckets.setdefault(key, []).append(r)
        out = []
        for key in sorted(buckets):
            grp = buckets[key]
            first = grp[0]
            speeds = [g["avg_speed"] for g in grp if g["avg_speed"] > 0]
            out.append({
                "date_time": key + ":00:00",
                "lat": first["lat"],
                "lon": first["lon"],
                "geohash": first["geohash"],
                "min_speed": round(min((g["min_speed"] for g in grp if g["min_speed"] > 0), default=0), 1),
                "max_speed": round(max((g["max_speed"] for g in grp), default=0), 1),
                "avg_speed": round(sum(speeds) / len(speeds), 1) if speeds else 0,
                "vehicle_count": sum(g["vehicle_count"] for g in grp),
                "year": first["year"],
                "month": first["month"],
                "hour": int(key[11:13]),
                "day_of_week": first["day_of_week"],
                "period_minutes": round(sum(g.get("period_minutes", 0) for g in grp), 1),
            })
        return out

    # ── RESET ────────────────────────────────────────────────────────────

    def reset(self):
        self.tracks.clear()
        self.count_in = self.count_out = 0
        self.active_ids.clear()
        self.type_counts.clear()
        self._window_30s.clear()
        self._window_5m.clear()
        self._window_1h.clear()
        self._speed_samples.clear()
        self._frame_times.clear()
        self._last_frame_ts = 0.0
        self._heatmap = None
        self._heatmap_size = (0, 0)
        self._report_rows.clear()
        self._report_last_ts = 0.0
        self._report_period_types.clear()
        self._report_period_count = 0
        self._report_start_time = 0.0
        self._period_speeds.clear()
        self.parked_ids.clear()
        self._session_speed_sum = 0.0
        self._session_speed_n = 0
        self._period_start_dt = None

    # ── INTERNALS ────────────────────────────────────────────────────────

    @staticmethod
    def _trim(dq, secs):
        c = time.perf_counter() - secs
        while dq and dq[0][0] < c:
            dq.popleft()

    @staticmethod
    def _window_avg(dq):
        return round(sum(v for _, v in dq)/len(dq), 1) if dq else 0.0

    @staticmethod
    def _density_score(active, fh):
        return min(round((active / max(fh/36, 1)) * 100, 1), 100.0)

    @staticmethod
    def _density_label(s):
        if s < 30: return "Düşük"
        if s < 65: return "Orta"
        return "Yüksek"

    def _avg_speed(self):
        return sum(self._speed_samples)/len(self._speed_samples) if self._speed_samples else 0.0

    def _session_avg_speed(self):
        """Oturum başından beri biriken tüm (park hariç) hız örneklerinin ortalaması."""
        return self._session_speed_sum / self._session_speed_n if self._session_speed_n else 0.0
