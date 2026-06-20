"""
video_thread.py — Background worker thread for YOLO inference.
Four source modes:
  1. FILE    — OpenCV VideoCapture (seekable)
  2. URL     — yt-dlp direct extraction + OpenCV CAP_FFMPEG
  3. SCREEN  — mss screen capture (legacy)
  4. WINDOW  — PrintWindow background capture (Alt-Tab safe)
Multi-polygon ROI and heatmap support.
"""

import os
import time
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QRect
from PyQt6.QtGui import QImage

from analytics import (
    TrafficAnalytics, VEHICLE_CLASS_IDS, VEHICLE_CLASSES,
    ZONE_COLORS_BGR,
)

SOURCE_FILE = "file"
SOURCE_URL = "url"
SOURCE_SCREEN = "screen"
SOURCE_WINDOW = "window"


class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage, dict)
    progress_update = pyqtSignal(int, int, str, str, bool)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QMutex()
        self._running = False
        self._paused = False
        self._seek_frame = -1

        self.source = ""
        self.source_mode = SOURCE_FILE
        self.screen_rect = QRect()
        self.model_variant = "yolo11n.pt"
        self.device = "cuda"
        self.confidence = 0.35
        self.line_ratio = 0.50

        # Window capture mode
        self.window_hwnd: int = 0
        self.window_crop: tuple = (0, 0, 0, 0)  # (x, y, w, h) in window coords

        # Multi-polygon
        self.polygons: list = []
        self.heatmap_enabled = False
        self.report_interval = 300

        # Yeni özellikler
        self.night_mode = False          # CLAHE kontrast iyileştirme (gece)
        self.record_video = False        # işlenmiş görüntüyü MP4 kaydet
        self.line_ratio2 = -1.0          # 2. sayım çizgisi (negatif = kapalı)
        self._clahe = None
        self._writer = None
        self._writer_size = None
        self._writer_path = ""

        self.analytics = TrafficAnalytics()

    # ── Controls ─────────────────────────────────────────────────────────

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False

    def pause(self):
        with QMutexLocker(self._mutex):
            self._paused = True

    def resume(self):
        with QMutexLocker(self._mutex):
            self._paused = False

    def toggle_pause(self):
        with QMutexLocker(self._mutex):
            self._paused = not self._paused

    def seek_to(self, frame_number: int):
        with QMutexLocker(self._mutex):
            self._seek_frame = frame_number

    def update_screen_rect(self, rect: QRect):
        with QMutexLocker(self._mutex):
            self.screen_rect = QRect(rect)

    def set_polygons(self, polygons: list):
        with QMutexLocker(self._mutex):
            self.polygons = [list(p) for p in polygons]

    def set_heatmap(self, enabled: bool):
        with QMutexLocker(self._mutex):
            self.heatmap_enabled = enabled

    @property
    def running(self):
        with QMutexLocker(self._mutex):
            return self._running

    @property
    def paused(self):
        with QMutexLocker(self._mutex):
            return self._paused

    @staticmethod
    def _fmt(seconds):
        s = max(0, int(seconds))
        h, r = divmod(s, 3600)
        m, sec = divmod(r, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    # ── Main ─────────────────────────────────────────────────────────────

    def run(self):
        from ultralytics import YOLO

        self._running = True
        self._paused = False
        self._seek_frame = -1
        self.analytics.reset()
        self.analytics.set_report_interval(self.report_interval)

        self.log_message.emit(f"⏳ Model: {self.model_variant} ({self.device})")
        try:
            model = YOLO(self.model_variant)
            model.predict(
                np.zeros((640, 640, 3), dtype=np.uint8),
                device=self.device, verbose=False,
            )
            self.log_message.emit("✅ Model hazır")
        except Exception as e:
            self.log_message.emit(f"❌ Model yüklenemedi: {e}")
            self.finished_signal.emit()
            return

        try:
            if self.source_mode == SOURCE_FILE:
                self._run_file(model)
            elif self.source_mode == SOURCE_URL:
                self._run_url(model)
            elif self.source_mode == SOURCE_SCREEN:
                self._run_screen(model)
            elif self.source_mode == SOURCE_WINDOW:
                self._run_window(model)
        except Exception as e:
            self.log_message.emit(f"❌ Beklenmeyen hata: {e}")

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self.log_message.emit(f"🎥 İşlenmiş video kaydedildi: {self._writer_path}")

        self.log_message.emit("⏹ İşlem durduruldu")
        self.finished_signal.emit()

    # ═════════════════════════════════════════════════════════════════════
    #  FILE MODE
    # ═════════════════════════════════════════════════════════════════════

    def _run_file(self, model):
        self.log_message.emit(f"📹 Dosya: {self.source}")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.log_message.emit("❌ Dosya açılamadı!")
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ts = tf / fps if fps > 0 else 0
        ly = int(fh * self.line_ratio)

        self.log_message.emit(f"✅ 📄 DOSYA — {fw}×{fh} @ {fps:.0f} FPS | {self._fmt(ts)}")

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            with QMutexLocker(self._mutex):
                req = self._seek_frame
                self._seek_frame = -1
            if req >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, req)
                self.analytics.reset()

            ret, frame = cap.read()
            if not ret:
                self.log_message.emit("📄 Video sona erdi")
                break

            self.analytics.tick_fps()
            cf = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.progress_update.emit(cf, tf, self._fmt(cf / fps), self._fmt(ts), True)
            self._process(model, frame, fw, fh, ly)

        cap.release()

    # ═════════════════════════════════════════════════════════════════════
    #  URL MODE (yt-dlp + OpenCV FFMPEG)
    # ═════════════════════════════════════════════════════════════════════

    def _run_url(self, model):
        self.log_message.emit(f"🌐 URL çözümleniyor: {self.source[:60]}…")

        stream_url = self.source
        is_live = False

        is_web = any(d in self.source.lower() for d in [
            "youtube.com", "youtu.be", "twitch.tv", "dailymotion",
            "vimeo.com", "facebook.com",
        ])

        if is_web:
            try:
                import yt_dlp
                ydl_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "format": (
                        "best[ext=mp4][protocol!=m3u8][protocol!=m3u8_native][height<=1080]/"
                        "best[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]/"
                        "best[ext=mp4][height<=1080]/best[ext=mp4]/"
                        "best[height<=1080]/best"
                    ),
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source, download=False)
                    stream_url = info.get("url", "")
                    is_live = bool(info.get("is_live", False))
                    headers = info.get("http_headers", {})
                    ua = headers.get("User-Agent", "Mozilla/5.0")
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                        f"user_agent;{ua}"
                        f"|referer;https://www.youtube.com/"
                        f"|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
                    )
                    fmt = info.get("format", "?")
                    w = info.get("width", "?")
                    h = info.get("height", "?")

                if is_live:
                    self.log_message.emit(f"📡 Canlı yayın ({w}×{h})")
                else:
                    self.log_message.emit(f"🎬 Çözümlendi: {fmt} ({w}×{h})")
            except Exception as e:
                self.log_message.emit(f"⚠️ yt-dlp: {e}")
                self.log_message.emit("🔄 Doğrudan URL deneniyor…")

        self.log_message.emit("📹 Akış açılıyor…")
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            self.log_message.emit("❌ Video akışı açılamadı!")
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ly = int(fh * self.line_ratio)
        self.log_message.emit(f"✅ 📡 CANLI — {fw}×{fh}")

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            ret, frame = cap.read()
            if not ret:
                if is_live:
                    time.sleep(0.1)
                    continue
                self.log_message.emit("📡 Akış sona erdi")
                break
            self.analytics.tick_fps()
            self.progress_update.emit(0, 0, "", "", False)
            self._process(model, frame, fw, fh, ly)

        cap.release()
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

    # ═════════════════════════════════════════════════════════════════════
    #  SCREEN MODE (legacy mss)
    # ═════════════════════════════════════════════════════════════════════

    def _run_screen(self, model):
        import mss

        with QMutexLocker(self._mutex):
            r = self.screen_rect
        if r.isEmpty():
            self.log_message.emit("❌ Ekran bölgesi seçilmedi!")
            return

        mon = {"left": r.x(), "top": r.y(), "width": r.width(), "height": r.height()}
        fw, fh = r.width(), r.height()
        ly = int(fh * self.line_ratio)
        self.log_message.emit(f"✅ 📷 EKRAN — {fw}×{fh}")

        with mss.mss() as sct:
            while self.running:
                if self.paused:
                    time.sleep(0.05)
                    continue
                with QMutexLocker(self._mutex):
                    cr = self.screen_rect
                if not cr.isEmpty():
                    mon = {"left": cr.x(), "top": cr.y(),
                           "width": cr.width(), "height": cr.height()}
                    fw, fh = cr.width(), cr.height()
                    ly = int(fh * self.line_ratio)

                frame = np.array(sct.grab(mon))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self.analytics.tick_fps()
                self.progress_update.emit(0, 0, "", "", False)
                self._process(model, frame, fw, fh, ly)
                time.sleep(0.01)

    # ═════════════════════════════════════════════════════════════════════
    #  WINDOW MODE (PrintWindow — works behind other windows)
    # ═════════════════════════════════════════════════════════════════════

    def _run_window(self, model):
        from win_utils import capture_window, is_window_valid

        hwnd = self.window_hwnd
        cx, cy, cw, ch = self.window_crop

        if not hwnd or cw <= 0 or ch <= 0:
            self.log_message.emit("❌ Pencere veya bölge seçilmedi!")
            return

        if not is_window_valid(hwnd):
            self.log_message.emit("❌ Pencere artık geçerli değil!")
            return

        ly = int(ch * self.line_ratio)
        self.log_message.emit(f"✅ 🪟 PENCERE — Bölge {cw}×{ch} (Alt-Tab güvenli)")

        fail_count = 0
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            if not is_window_valid(hwnd):
                self.log_message.emit("⚠️ Pencere kapatıldı!")
                break

            full = capture_window(hwnd)
            if full is None:
                fail_count += 1
                if fail_count > 30:
                    self.log_message.emit("❌ Pencere yakalanamıyor (minimize mi?)")
                    break
                time.sleep(0.1)
                continue

            fail_count = 0

            # Crop the selected sub-region
            fh_full, fw_full = full.shape[:2]
            # Clamp crop to actual window size
            x1 = max(0, min(cx, fw_full - 1))
            y1 = max(0, min(cy, fh_full - 1))
            x2 = min(x1 + cw, fw_full)
            y2 = min(y1 + ch, fh_full)

            if x2 <= x1 or y2 <= y1:
                time.sleep(0.05)
                continue

            frame = full[y1:y2, x1:x2].copy()
            actual_w, actual_h = frame.shape[1], frame.shape[0]
            ly = int(actual_h * self.line_ratio)

            self.analytics.tick_fps()
            self.progress_update.emit(0, 0, "", "", False)
            self._process(model, frame, actual_w, actual_h, ly)
            time.sleep(0.01)  # ~30+ FPS cap

    # ═════════════════════════════════════════════════════════════════════
    #  VERİ PANELİ (video üzerine basılır — kayıtta da görünür)
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tr_ascii(s: str) -> str:
        """cv2 fontları Türkçe karakter çizemez — ASCII'ye çevir."""
        return (s.replace("ı", "i").replace("İ", "I").replace("ş", "s").replace("Ş", "S")
                 .replace("ğ", "g").replace("Ğ", "G").replace("ü", "u").replace("Ü", "U")
                 .replace("ö", "o").replace("Ö", "O").replace("ç", "c").replace("Ç", "C"))

    def _draw_stats_overlay(self, frame, metrics, fw, fh):
        unit = metrics.get("speed_unit", "px/s")
        td = metrics.get("type_distribution", {})
        top_types = sorted(td.items(), key=lambda kv: -kv[1])[:3]
        dist_txt = "  ".join(f"{self._tr_ascii(k)}:{v}" for k, v in top_types) or "-"

        lines = [
            ("TRAFIK ANALIZ", (90, 200, 255)),
            (f"Anlik Arac: {metrics.get('active_count', 0)}    Park: {metrics.get('parked_count', 0)}", (255, 255, 255)),
            (f"Anlik Hiz: {metrics.get('avg_speed_px_s', 0)} {unit}", (255, 255, 255)),
            (f"Oturum Ort. Hiz: {metrics.get('session_avg_speed', 0)} {unit}", (120, 255, 170)),
            (f"Giren: {metrics.get('count_in', 0)}   Cikan: {metrics.get('count_out', 0)}   Toplam: {metrics.get('count_total', 0)}", (255, 255, 255)),
            (f"Dagilim: {dist_txt}", (200, 200, 210)),
        ]

        # Panel boyutu: kare yüksekliğine göre ölçekle
        scale = max(0.42, min(0.62, fh / 1100))
        lh_px = int(26 * scale / 0.5)
        pad = 10
        tw_max = 0
        for txt, _c in lines:
            (tw, _th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            tw_max = max(tw_max, tw)
        pw = tw_max + pad * 2
        ph = lh_px * len(lines) + pad * 2
        x0 = max(0, fw - pw - 12)
        y0 = max(0, fh - ph - 12)

        # Yarı saydam koyu zemin + ince kenarlık
        roi = frame[y0:y0 + ph, x0:x0 + pw]
        if roi.size > 0:
            panel = roi.copy()
            panel[:] = (16, 12, 10)
            cv2.addWeighted(panel, 0.72, roi, 0.28, 0, roi)
            cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (140, 110, 80), 1, cv2.LINE_AA)
            cv2.line(frame, (x0, y0 + lh_px + pad // 2 + 2),
                     (x0 + pw, y0 + lh_px + pad // 2 + 2), (90, 70, 55), 1, cv2.LINE_AA)

        ty = y0 + pad + int(16 * scale / 0.5)
        for txt, color in lines:
            cv2.putText(frame, txt, (x0 + pad, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            ty += lh_px

    # ═════════════════════════════════════════════════════════════════════
    #  SHARED FRAME PROCESSING
    # ═════════════════════════════════════════════════════════════════════

    def _process(self, model, frame, fw, fh, ly):
        palette = {
            2: (0, 220, 120), 3: (255, 180, 0), 5: (60, 160, 255),
            7: (200, 80, 255), 1: (255, 255, 0),
        }
        dc = (180, 180, 180)

        with QMutexLocker(self._mutex):
            polys = [list(p) for p in self.polygons]
            show_hm = self.heatmap_enabled

        # Gece modu: CLAHE ile kontrast iyileştirme (karanlık görüntüde tespit artar)
        if self.night_mode:
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            frame = cv2.cvtColor(cv2.merge((self._clahe.apply(l_ch), a_ch, b_ch)),
                                 cv2.COLOR_LAB2BGR)

        # GPU'da daha büyük girdi (960) + FP16: küçük/uzak araçlar daha iyi yakalanır
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=self.confidence, iou=0.45,
            imgsz=960 if self.device == "cuda" else 640,
            half=(self.device == "cuda"),
            device=self.device,
            classes=list(VEHICLE_CLASS_IDS), verbose=False,
        )

        dets = []
        boxes = results[0].boxes if results and results[0].boxes is not None else []
        if hasattr(boxes, "id") and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = xyxy[i]
                # Hayalet tespitleri ele: çok küçük kutular araç değildir
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue
                dets.append({
                    "id": tid, "class_id": clss[i],
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "conf": confs[i],
                })

        # Sayım çizgileri (2. çizgi etkinse listeye eklenir)
        line_list = [ly]
        if self.line_ratio2 is not None and self.line_ratio2 >= 0:
            line_list.append(int(fh * self.line_ratio2))

        metrics = self.analytics.update(
            dets, line_list, fh, frame_w=fw,
            polygons=polys if polys else None,
        )

        # Heatmap
        if show_hm:
            frame = self.analytics.get_heatmap_overlay(frame, 0.45)

        # Draw polygon zones
        for idx, poly in enumerate(polys):
            if len(poly) < 3:
                continue
            color = ZONE_COLORS_BGR[idx % len(ZONE_COLORS_BGR)]
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
            zx = int(np.mean([p[0] for p in poly]))
            zy = int(np.mean([p[1] for p in poly]))
            cv2.putText(frame, f"Bolge {idx + 1}", (zx - 30, zy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Counting line(s)
        _line_cols = [(0, 255, 255), (255, 100, 255)]
        for _li, _ly in enumerate(line_list):
            _lc = _line_cols[_li % len(_line_cols)]
            cv2.line(frame, (0, _ly), (fw, _ly), _lc, 2)
            cv2.putText(frame, f"SAYIM CIZGISI {_li + 1}" if len(line_list) > 1 else "SAYIM CIZGISI",
                        (10, _ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _lc, 1, cv2.LINE_AA)

        # Boxes (park halindeki araçlar gri kutu + "PARK HALINDE" etiketiyle ayrılır)
        parked_ids = self.analytics.parked_ids
        for det in dets:
            x1, y1 = int(det["x1"]), int(det["y1"])
            x2, y2 = int(det["x2"]), int(det["y2"])
            cls = det["class_id"]
            is_parked = det["id"] in parked_ids
            col = (140, 140, 150) if is_parked else palette.get(cls, dc)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            if is_parked:
                txt = f"P PARK HALINDE #{det['id']}"
            else:
                lbl = f'{VEHICLE_CLASSES.get(cls, "?")} #{det["id"]}'
                txt = f"{lbl} {det['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), col, -1)
            cv2.putText(frame, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            ccx, ccy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (ccx, ccy), 4, col, -1)

        # Veri paneli: sağ alt köşeye canlı istatistikleri bas
        # (hem ekranda hem kaydedilen MP4'te görünür)
        self._draw_stats_overlay(frame, metrics, fw, fh)

        # İşlenmiş video kaydı (MP4)
        if self.record_video:
            if self._writer is None:
                rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kayitlar")
                os.makedirs(rec_dir, exist_ok=True)
                import datetime as _dt
                self._writer_path = os.path.join(
                    rec_dir, f"islenmis_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                self._writer_size = (frame.shape[1], frame.shape[0])
                self._writer = cv2.VideoWriter(
                    self._writer_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    25, self._writer_size)
                self.log_message.emit(f"🎥 Video kaydı başladı: {self._writer_path}")
            f_out = frame
            if (frame.shape[1], frame.shape[0]) != self._writer_size:
                f_out = cv2.resize(frame, self._writer_size)
            self._writer.write(f_out)

        # Emit
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.frame_ready.emit(q, metrics)
