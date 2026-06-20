"""
app_web.py — İstanbul Trafik Analiz ve Tahmin Platformu  v3.3
Streamlit Dashboard — İBB Açık Veri Seti (2020–2025)

v3.3 Yenilikler:
  • Gerçek ML modelleri (Random Forest, XGBoost, Gradient Boosting, Ridge)
  • models/ klasöründen eğitilmiş modeller yüklenir
  • Canlı Trafik iframe (Yandex Widget API)
  • Isı Haritası + Modern Noktalar (Dark)
  • Tek tarih seçici + opsiyonel hava durumu
  • Geçmiş tahminlerde Ground Truth karşılaştırması
  • Dinamik model değiştirme
"""

import streamlit as st
import streamlit.components.v1 as components
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import os
import requests
import pickle
import json
from datetime import datetime, date, timedelta
import random
import torch
import torch.nn as nn
from shared_models import TrafficCNNLSTMModel


# ─────────────────────────────────────────────────────────────────────────
#  ÇOKLU DİL DESTEĞİ / MULTILINGUAL SUPPORT
# ─────────────────────────────────────────────────────────────────────────
LANG_TR = {
    "hero_badge": "● YZ Destekli Analiz Platformu",
    "hero_title1": "🚦 İstanbul Trafik",
    "hero_title2": "Yoğunluk Tahmini",
    "hero_subtitle": "İBB Açık Veri Seti üzerinde derin öğrenme ile trafik analizi, tahmini ve rota optimizasyonu",
    "pill_records": "~100M Kayıt", "pill_period": "2020–2024",
    "pill_models": "5 YZ Modeli", "pill_weather": "Hava Durumu API",
    "sidebar_brand": "TrafikAI", "sidebar_subtitle": "İstanbul Trafik Platformu",
    "sidebar_filters": "⚙️ Filtreler", "sidebar_year": "📅 Yıl",
    "sidebar_month": "📆 Ay", "sidebar_day": "🗓️ Gün", "sidebar_hour": "⏰ Saat",
    "sidebar_data_info": "📊 Veri Seti Bilgisi", "all_month": "Tüm Ay",
    "total_records": "Toplam Kayıt", "vehicle_passage": "Araç Geçişi",
    "period": "Dönem", "monthly_file": "Aylık Dosya", "months_suffix": "ay",
    "tab1": "📍 Yoğunluk Haritası", "tab2": "📈 Zaman Analizi",
    "tab3": "📊 Yıl Karşılaştırma", "tab4": "🔮 YZ Tahmin + Hava Durumu",
    "tab5": "🤖 YZ Modelleri Karşılaştırma", "tab6": "📥 Veri Ekle / Karşılaştır",
    "total_vehicles": "Toplam Araç", "avg_speed": "Ortalama Hız",
    "active_locations": "Aktif Nokta", "record_count": "Kayıt Sayısı",
    "active_days": "Aktif Gün",
    "monthly_report": "Aylık Rapor Özeti", "peak_hours": "🕐 En Yoğun Saatler",
    "busiest_areas": "📍 En Yoğun Bölgeler", "co2_emission": "🌿 Tahmini CO₂ Emisyonu",
    "per_vehicle": "Araç başı", "speed_factor": "Hız faktörü",
    "map_title": "Trafik Yoğunluk Haritası", "map_type": "Harita Tipi",
    "color_by": "Renklendirme", "by_vehicle": "Araç Sayısı",
    "by_speed": "Ort. Hız (ters)", "hour_stats": "📊 Bu Saatin İstatistikleri:",
    "map_reset": "🔄 Haritayı Sıfırla (İstanbul Genel)",
    "region_analysis": "Bölge Analizi",
    "slowest_5": "🔴 En Yavaş 5 Nokta", "fastest_5": "🟢 En Hızlı 5 Nokta",
    "most_congested_5": "🟠 En Yoğun 5 Nokta", "least_congested_5": "🟣 En Az Yoğun 5 Nokta",
    "go_btn": "📍 Git", "no_data": "Veri bulunamadı.",
    "time_analysis": "📈 Zaman Serisi Analizi",
    "hourly_profile": "Saatlik Trafik Profili",
    "weekly_profile": "Haftalık Trafik Profili",
    "monthly_trend": "📆 Aylık Trafik Trendi (Tüm Yıllar)",
    "weekday_vs_weekend": "Hafta İçi vs Hafta Sonu Kıyaslama",
    "rush_analysis": "Zirve Saat (Rush Hour) Analizi",
    "weekday": "Hafta İçi", "weekend": "Hafta Sonu",
    "morning_rush": "Sabah (07-09)", "evening_rush": "Akşam (17-19)",
    "year_comparison": "📊 Trafik Zaman Karşılaştırması",
    "yoy": "📅 Yıllar Arası Karşılaştırma (YoY)",
    "mom": "📆 Yıl İçi Aylık Değişim (MoM)",
    "base_year": "1. Yıl (Temel/Referans)", "target_year": "2. Yıl (Hedef/Karşılaştırılacak)",
    "pandemic_effect": "🦠 Pandemi Etkisi: 2020 vs Diğer Yıllar",
    "pandemic": "Pandemi (2020)", "normal_period": "Normal Dönem",
    "improving_areas": "Gelişen vs Kötüleşen Bölgeler (Hız Değişimi)",
    "speeding_districts": "Hızlanan İlçeler", "slowing_districts": "Yavaşlayan İlçeler",
    "increased": "arttı", "decreased": "düştü",
    "avg_speed_short": "Ort. Hız (km/s)", "avg_vehicles": "Ort. Araç Sayısı",
    "comparison_type": "Karşılaştırma Türü", "year_to_inspect": "İncelemek İstediğiniz Yıl",
    "two_month_comp": "📆 İki Ay Detaylı Karşılaştırması",
    "month_1": "1. Ay", "month_2": "2. Ay",
    "ai_prediction": "🔮 YZ Destekli Trafik Yoğunluk Tahmini",
    "ai_prediction_subtitle": "Tek tarih seç, modeli seç, tahmin et. Geçmiş tarihler için **gerçek veri** karşılaştırması otomatik gösterilir.",
    "date_and_hour": "📅 Tarih ve Saat", "select_date": "📅 Tarih Seçin",
    "hour_slider": "🕐 Saat",
    "date_hint": "2020-01-01 → 2030-12-31 arası seçilebilir. 14 günden sonrası için hava durumu verisi çekilemez.",
    "past_date_caption": "📂 Geçmiş tarih — Gerçek veri karşılaştırması yapılacak",
    "future_date_caption": "🔮 Gelecek tarih — Forecast API kullanılacak",
    "model_selection": "🤖 YZ Model Seçimi", "prediction_model": "Tahmin Modeli",
    "accuracy_rate": "Doğruluk Oranı", "model_accuracy_fmt": "Doğruluk: %",
    "weather_api_cb": "🌦️ Hava Durumu API", "predict_btn": "🔮 Tahmin Et",
    "forecast_api": "🔮 Tahmin (Forecast API)", "archive_api": "📂 Geçmiş (Archive API)",
    "temperature": "Sıcaklık", "precipitation": "Yağış",
    "rain_impact_msg": "⚠️ Yağış tespit edildi! Yoğunluk tahminine +",
    "rain_impact_suffix": "% yük uygulandı.",
    "dry_weather_msg": "✅ Kuru hava: Tahmin düzeltmesi uygulanmadı.",
    "ai_pred_vs_actual": "Tahmini vs Gerçekleşen Veri",
    "ai_prediction_title": "🔮 YZ Tahmini", "actual_title": "✅ Gerçekleşen (O Gün)",
    "vehicle_label": "Araç", "speed_label": "Hız",
    "status_label": "Durum", "confidence_label": "Güven",
    "vehicle_error": "Araç Hata (%)", "speed_error": "Hız Hata (%)",
    "accuracy": "Doğruluk (%)", "pred_vehicles": "Tahmini Araç",
    "pred_speed": "Tahmini Hız", "co2_kg": "CO₂ (kg)",
    "all_models_comparison": "🔬 Bu Senaryo İçin Tüm Modellerin Karşılaştırması",
    "pred_details": "Tahmin Detayları", "model_label": "Model",
    "confidence_vehicles": "Güven Aralığı (Araç)", "confidence_speed": "Güven Aralığı (Hız)",
    "base_vehicles": "Temel Araç (Ham)", "base_speed_lbl": "Temel Hız",
    "sample_count": "Örnek Sayısı", "weather_effect": "Hava Durumu Etkisi",
    "co2_hourly": "🌿 Tahmini CO₂ Emisyonu (Bu Saat)",
    "signal_recommendation": "🚦 Dinamik Sinyalizasyon Önerisi",
    "signal_warning": "Ortalama hız", "signal_action": "Ana arter için yeşil ışık süresini +",
    "signal_action2": "saniye artırın.", "wait_reduction": "Tahmini bekleme süresi azalma: ~%",
    "route_optimization": "### 🗺️ Rota Optimizasyonu Önerisi",
    "start_district_lbl": "🟢 Başlangıç İlçesi", "end_district_lbl": "🔴 Varış İlçesi",
    "route_alternatives": "#### 📋 Alternatif Güzergahlar",
    "route_map_title": "#### 🗺️ Güzergah Haritası",
    "select_route_map": "🗺️ Haritada Gösterilecek Güzergahı Seçin",
    "recommended_badge": "✅ ÖNERİLEN",
    "predict_first": "👆 Parametreleri seçip **Tahmin Et** butonuna basın.",
    "osrm_fallback": "🌐 OpenStreetMap rota servisine (OSRM) ulaşılamadı — mesafeler simüle edildi.",
    "different_districts": "Başlangıç ve varış ilçesini farklı seçiniz.",
    "no_route_geometry": "Rota geometrisi alınamadı — güzergah kartlarındaki değerler geçerlidir.",
    "insuff_data": "Bu kombinasyon için yeterli veri bulunamadı.",
    "vehicle_count_label": "Araç Sayısı",
    "ai_engine_title": "🧠 Yapay Zeka Tahmin Motoru Hakkında",
    "ai_engine_desc": """Bu sekme, <i>"YZ Destekli Trafik Yoğunluğu Tahmin Raporu"</i>
               doğrultusunda geliştirilmiştir. Tahmin motoru şu bileşenleri kullanır:<br>
               • 📊 <b>100 Milyon Satırlık</b> İBB geçmiş trafik verisi<br>
               • 🌦️ <b>Open-Meteo API</b> — geçmiş (Archive) + gelecek (Forecast) hava durumu<br>
               • 🤖 <b>4 Farklı YZ Modeli:</b> CNN-LSTM, XGBoost, Random Forest, ARIMA<br>
               • ✅ <b>Gerçek Veri Karşılaştırması:</b> Geçmiş tarihler için dataset'ten Ground Truth<br>
               • 🌿 <b>CO₂ Emisyon Tahmini</b> + 🗺️ <b>Rota Optimizasyonu</b><br>
               • 🚦 <b>Dinamik Sinyalizasyon Önerisi</b>""",
    "ai_models": "### 🤖 YZ Modelleri Karşılaştırma",
    "ai_models_subtitle": "Farklı yapay zeka modellerinin performans metrikleri ve aynı veri üzerindeki tahmin sonuçları.",
    "model_metrics": "#### 📊 Model Performans Metrikleri",
    "model_charts": "#### 📈 Model Karşılaştırma Grafikleri",
    "model_accuracy_chart": "Model Doğruluk Oranları (%)",
    "model_error_chart": "Model Hata Metrikleri (MAE & RMSE)",
    "all_models_predict": "#### 🔮 Tüm Modellerle Aynı Anda Tahmin",
    "compare_btn": "🔬 Karşılaştır", "select_date2": "📅 Tarih",
    "radar_chart": "#### 🎯 Model Yetenek Radar Grafiği",
    "radar_cats": ["Doğruluk","Hız","Genelleme","Yorumlanabilirlik","Esneklik"],
    "actual_label": "Gerçek", "pred_vehicles_chart": "Tahmini Araç Sayısı",
    "pred_speed_chart": "Tahmini Hız (km/s)",
    "compare_first": "👆 Parametreleri seçip **Karşılaştır** butonuna basın.",
    "data_upload": "### 📥 Veri Ekle / Karşılaştır",
    "data_upload_desc": "Masaüstü uygulamasından aldığınız CSV dosyalarını yükleyin ve İBB veri setiyle karşılaştırın. Veriler **sadece bu oturum boyunca** saklanır.",
    "upload_csv": "📂 CSV Dosyası Yükle (traffic_density_*.csv)",
    "loaded_data": "**📋 Yüklü Veriler:**", "no_data_loaded": "Henüz veri yüklenmedi.",
    "delete_all": "🗑️ Tümünü Temizle",
    "comparison_results": "### 📊 Karşılaştırma Sonuçları",
    "region_label": "📍 Bölge", "pattern_match": "🔁 Desen Uyumu",
    "hourly_vehicles_label": "🚗 Saatlik Araç: Siz / İBB",
    "speed_diff": "⚡ Ort. Hız Farkı",
    "no_geohash_match": "Geohash eşleşmesi bulunamadı — İBB veri setinde bu konumlara ait kayıt yok.",
    "upload_required_cols": "Yüklenen CSV'de `geohash` ve `hour` sütunları gereklidir.",
    "upload_success": "yüklendi!", "records_suffix": "kayıt",
    "missing_cols": "Eksik sütunlar", "file_read_error": "Dosya okunamadı",
    "compatible_expected": "Dataset uyumlu CSV bekleniyor.",
    "ibb_label": "🏛️ İBB Tarihsel Desen", "your_measurement": "📹 Sizin Ölçümünüz",
    "density_index": "Yoğunluk Endeksi (ortalama = 100)", "hour_axis": "Saat",
    "regional_analysis_short": "Bölgesel Analizi — Kısa Kayıt Modu",
    "no_ibb_history": "Bu bölge için İBB geçmiş verisi bulunamadı.",
    "your_speed_lbl": "Sizin Hız", "ibb_speed_lbl": "İBB Hız",
    "ibb_avg_speed": "İBB Ort. (saat)", "your_vehicles": "Sizin Araç (saat)",
    "index_diff": "Endeks Farkı", "speed_diff_col": "Hız Farkı",
    "basis_col": "Temel", "district_col": "İlçe", "geohash_col": "Geohash",
    "hour_col": "Saat",
    "km_h": "km/s", "ton_co2": "ton CO₂", "vehicles_lbl": "araç",
    "level_very_congested": "🔴 ÇOK YOĞUN", "level_congested": "🟠 YOĞUN",
    "level_moderate": "🟡 ORTA", "level_flowing": "🟢 AKICI",
    "level_congested_route": "🔴 Yoğun", "level_moderate_route": "🟠 Orta",
    "level_flowing_route": "🟢 Akıcı",
    "day_names": ["Pazartesi","Salı","Çarşamba","Perşembe","Cuma","Cumartesi","Pazar"],
    "months": {1:"Ocak",2:"Şubat",3:"Mart",4:"Nisan",5:"Mayıs",6:"Haziran",
               7:"Temmuz",8:"Ağustos",9:"Eylül",10:"Ekim",11:"Kasım",12:"Aralık"},
    "wmo": {
        0:("☀️","Açık / Güneşli"),1:("🌤️","Parçalı Açık"),2:("⛅","Parçalı Bulutlu"),
        3:("☁️","Kapalı / Bulutlu"),45:("🌫️","Sisli"),48:("🌫️","Kırağılı Sis"),
        51:("🌦️","Hafif Çiseleme"),53:("🌦️","Orta Çiseleme"),55:("🌧️","Yoğun Çiseleme"),
        61:("🌧️","Hafif Yağmur"),63:("🌧️","Orta Yağmur"),65:("🌧️","Şiddetli Yağmur"),
        71:("🌨️","Hafif Kar"),73:("🌨️","Orta Kar"),75:("❄️","Yoğun Kar"),
        80:("🌧️","Sağanak Yağmur"),95:("⛈️","Gök Gürültülü Fırtına"),
    },
    "model_descriptions": {
        "CNN-LSTM": "CNN-LSTM — Zaman serisi tahminleri için evrişimsel ve uzun-kısa süreli bellek hibrit derin öğrenme modeli.",
        "Gradient Boosting": "Gradient Boosting — Ağaçları sıralı birleştiren güçlü model. En yüksek doğruluk.",
        "Random Forest": "Rassal Orman — Çoklu karar ağaçları ile genelleme. Dengeli performans.",
        "XGBoost": "eXtreme Gradient Boosting — Hızlı ve verimli boosting modeli.",
        "Ridge Regression": "Ridge (L2) Doğrusal Regresyon — Basit ve hızlı temel model.",
    },
    "footer": "© 2026 Tüm Hakları Saklıdır — Proje Abdurrahman Bayrakdar ve Şevket Can Yorulmaz tarafından geliştirilmiştir.",
    "unknown": "Bilinmiyor", "avg_speed_label": "Ort. Hız (km/s)",
    "avg_vehicles_label": "Ort. Araç Sayısı", "month_axis": "Ay",
    "change_pct": "Değişim (%)", "yoy_growth_rate": "Yıllık Büyüme Oranı",
    "mom_growth_rate": "Aylar Arası Büyüme Oranı (MoM)",
    "speed_trend": "Yılı Aylık Hız Trendi", "density_trend": "Yılı Aylık Araç Yoğunluğu Trendi",
    "pandemic_speed": "Ortalama Hız", "pandemic_vehicles": "Ortalama Araç Sayısı",
    "vehicle_count_change": "Araç Sayısı Değişimi", "speed_change_lbl": "Ortalama Hız Değişimi",
    "weather_fetching": "Open-Meteo API'den hava durumu çekiliyor...",
    "weather_no_data": "Seçilen tarih 14 günlük tahmin sınırından uzak olduğu için hava durumu otomatik olarak 'Açık/Normal' kabul edilmiştir.",
    "your_hour_rank": "'de Bu Saatin Sırası", "your_vs_history": "Hız: Siz vs",
    "your_vs_history2": "Geçmişi", "busiest_year": "'de En Yoğun Yıl",
    "rank_in_day": "Günün en yoğun", "rank_suffix": ". saati",
    "osrm_start": "🟢 Başlangıç:", "osrm_end": "🏁 Varış:",
    "actual_occurred": "O gün gerçekleşen",
    "consistent_msg": "ölçümünüz İBB'nin saatlik ortalamasıyla **uyumlu**",
    "higher_msg": "ölçümünüz, İBB'nin aynı koşullardaki saatlik ortalamasından",
    "more_dense": "daha yoğun", "calmer": "daha sakin",
    "no_data_warning": "Bu filtre için veri bulunamadı.",
    "comparison_basis": "Karşılaştırma temeli: İBB veri setinin",
    "basis_month_day_avg": "günleri ortalaması",
    "basis_month_avg": "ayı ortalaması (gün bazında yeterli kayıt yok)",
    "basis_general_avg": "tüm dönem ortalaması (ay bazında yeterli kayıt yok)",
    "hourly_vehicles_short": "Araç Sayısı", "hour_spd": "Hız (km/s)",
    "avg_speed_sec": "Ort. Hız", "vehicle_sec": "Ort. Araç",
    "rush_type_lbl": "Zirve Tipi",
    "short_rec_caption": "Kaydınız az sayıda saat içerdiği için bölgenin IBB geçmişi gösteriliyor.",
    "day_hour_profile_lbl": "Günleri Saatlik Profil",
    # ── Aliases / new keys added for full i18n ──
    "date_and_time": "Tarih ve Saat", "hour": "Saat",
    "date_help": "2020-01-01 → 2030-12-31 arası seçilebilir. 14 günden sonrası için hava durumu verisi çekilemez.",
    "ai_model_select": "YZ Model Seçimi", "weather_api": "Hava Durumu API",
    "weather_unavailable": "Seçilen tarih 14 günlük tahmin sınırından uzak olduğu için hava durumu otomatik olarak 'Açık/Normal' kabul edilmiştir.",
    "no_data_combination": "Bu kombinasyon için yeterli veri bulunamadı.",
    "weather_title": "Hava Durumu", "rain": "Yağış",
    "rain_detected": "Yağış tespit edildi! Yoğunluk tahminine +",
    "dry_weather": "Kuru hava: Tahmin düzeltmesi uygulanmadı",
    "very_heavy": "🔴 ÇOK YOĞUN", "heavy": "🟠 YOĞUN",
    "moderate": "🟡 ORTA", "flowing": "🟢 AKICI",
    "prediction_vs_actual": "Tahmini vs Gerçekleşen Veri",
    "ai_pred_label": "YZ Tahmini", "vehicle_lbl": "Araç",
    "speed_lbl": "Hız", "status_lbl": "Durum",
    "confidence": "Güven", "vehicles_unit": "araç",
    "vehicle_error_pct": "Araç Hata (%)", "speed_error_pct": "Hız Hata (%)",
    "prediction_result": "Tahmin Sonucu",
    "estimated_vehicle": "Tahmini Araç", "estimated_speed": "Tahmini Hız",
    "model_lbl": "Model", "date_lbl": "Tarih",
    "ci_vehicle": "Güven Aralığı (Araç)", "ci_speed": "Güven Aralığı (Hız)",
    "base_vehicle": "Temel Araç (Ham)", "base_speed": "Temel Hız",
    "hourly_records": "saatlik kayıt", "weather_impact": "Hava Durumu Etkisi",
    "all_models_compare": "Bu Senaryo İçin Tüm Modellerin Karşılaştırması",
    "per_vehicle": "Araç başı", "speed_factor": "Hız faktörü",
    "co2_title": "Tahmini CO₂ Emisyonu (Bu Saat)",
    "signal_title": "Dinamik Sinyalizasyon Önerisi",
    "green_light_advice": "Ana arter için yeşil ışık süresini +{extra_green} saniye artırın.",
    "wait_reduction": "Tahmini bekleme süresi azalma",
    "hourly_profile": "Günleri Saatlik Profil",
    "vehicle_count_lbl": "Araç Sayısı", "avg_speed_lbl_short": "Ort. Hız",
    "rainy": "Yağmurlu", "speed_kmh": "Hız (km/s)",
    "start_district": "Başlangıç İlçesi", "end_district": "Varış İlçesi",
    "osrm_unavailable": "OpenStreetMap rota servisine (OSRM) ulaşılamadı — mesafeler simüle edildi.",
    "alt_routes": "Alternatif Güzergahlar",
    "select_route_map": "Haritada Gösterilecek Güzergahı Seçin",
    "recommended": "ÖNERİLEN", "min": "dk", "route_map": "Güzergah Haritası",
    "start_label": "Başlangıç", "end_label": "Varış",
    "route_geometry_missing": "Rota geometrisi alınamadı — güzergah kartlarındaki değerler geçerlidir.",
    "select_diff_districts": "Başlangıç ve varış ilçesini farklı seçiniz.",
    "predict_hint": "Parametreleri seçip **Tahmin Et** butonuna basın.",
    "route_optimization": "Rota Optimizasyonu Önerisi",
    "ai_models_compare": "YZ Modelleri Karşılaştırma",
    "model_performance": "Model Performans Metrikleri",
    "model_compare_charts": "Model Karşılaştırma Grafikleri",
    "model_accuracy_title": "Model Doğruluk Oranları (%)",
    "model_error_title": "Model Hata Metrikleri (MAE & RMSE)",
    "all_models_predict": "Tüm Modellerle Aynı Anda Tahmin",
    "compare_hint": "Parametreleri seçip **Karşılaştır** butonuna basın.",
    "radar_chart_title": "Model Yetenek Radar Grafiği",
    "radar_categories": ["Doğruluk","Hız","Genelleme","Yorumlanabilirlik","Esneklik"],
    "actual_on_day": "O gün gerçekleşen", "actual_short": "Gerçek",
    "data_upload_title": "Veri Ekle / Karşılaştır",
    "data_upload_subtitle": "Masaüstü uygulamasından aldığınız CSV dosyalarını yükleyin ve İBB veri setiyle karşılaştırın. Veriler **sadece bu oturum boyunca** saklanır.",
    "upload_csv_label": "CSV Dosyası Yükle (traffic_density_*.csv)",
    "uploaded": "yüklendi", "records": "kayıt",
    "csv_format_required": "Dataset uyumlu CSV bekleniyor.",
    "uploaded_data": "Yüklü Veriler", "no_data_uploaded": "Henüz veri yüklenmedi.",
    "clear_all": "Tümünü Temizle", "compare_results": "Karşılaştırma Sonuçları",
    "region": "Bölge", "multiple_regions": "birden fazla bölge",
    "pattern_match_help": "Saatlik yoğunluk şeklinizin İBB profiliyle korelasyonu (≥3 farklı saat gerekir)",
    "hourly_vehicle_you_ibb": "Saatlik Araç: Siz / İBB",
    "hourly_vehicle_help": "Sayımınız saatlik orana çevrildi; İBB değeri bölgenin saatlik alan toplamı ortalamasıdır",
    "avg_speed_diff": "Ort. Hız Farkı",
    "basis_month_day": "{month} ayı, {day} günleri ortalaması",
    "basis_month": "{month} ayı ortalaması (gün bazında yeterli kayıt yok)",
    "basis_general": "tüm dönem ortalaması (ay bazında yeterli kayıt yok)",
    "ibb_hist_pattern": "İBB Tarihsel Desen",
    "regional_analysis": "Bölgesel Analizi", "short_record_mode": "Kısa Kayıt Modu",
    "hour_rank_in": "{district}'de Bu Saatin Sırası",
    "hour_rank_help": "{district} bölgesinin 24 saatlik İBB profilinde, ölçüm yaptığınız saatin yoğunluk sıralaması",
    "speed_you_vs": "Hız: Siz vs {district} Geçmişi",
    "busiest_year_in": "{district}'de En Yoğun Yıl",
    "busiest_year_help": "Bu bölgede saat {hours} için en yüksek ortalama yoğunluğun görüldüğü yıl",
    "avg_vehicle_lbl": "Ort. Araç", "ibb_speed": "İBB Hız",
    "your_speed": "Sizin Hız", "by_year": "Yıllara Göre", "by_day": "Günlere Göre",
    "your_day_marker": "ölçüm gününüz",
    "24h_profile": "24 Saat Profili", "your_hour_marker": "ölçüm saatiniz",
    "col_district": "İlçe", "col_hour": "Saat",
    "col_your_vehicle": "Sizin Araç (saat)", "col_ibb_vehicle": "İBB Ort. (saat)",
    "col_index_diff": "Endeks Farkı", "col_your_speed": "Sizin Hız",
    "col_ibb_speed": "İBB Hız", "col_speed_diff": "Hız Farkı", "col_basis": "Temel",
    "your_vehicle_help": "Sayımınızın saatlik orana çevrilmiş hali",
    "ibb_vehicle_help": "Bölgenin saatlik alan toplamı ortalaması",
    "index_diff_help": "Pozitif: o saat sizde İBB desenine göre daha yoğun",
    "compare_compatible": "✅ {district} ölçümünüz İBB'nin saatlik ortalamasıyla **uyumlu** ({pct}%).",
    "compare_denser": "⚠️ {district} ölçümünüz, İBB'nin aynı koşullardaki saatlik ortalamasından **%{pct} daha yoğun**.",
    "compare_lighter": "ℹ️ {district} ölçümünüz, İBB'nin aynı koşullardaki saatlik ortalamasından **%{pct} daha sakin**.",
    "csv_geohash_required": "Yüklenen CSV'de `geohash` ve `hour` sütunları gereklidir.",
    "short_rec_caption_long": "Kaydınız az sayıda saat içerdiği ({hours}) için **{district}** bölgesinin (geohash `{geohash}`) İBB geçmişi yıl / haftanın günü / günün saatleri kesitleriyle karşılaştırılıyor. Tüm grafikler bu bölgeye özeldir.",
}

LANG_EN = {
    "hero_badge": "● AI-Powered Analytics Platform",
    "hero_title1": "🚦 Istanbul Traffic",
    "hero_title2": "Density Prediction",
    "hero_subtitle": "Traffic analysis, prediction and route optimization with deep learning on IBB Open Dataset",
    "pill_records": "~100M Records", "pill_period": "2020–2024",
    "pill_models": "5 AI Models", "pill_weather": "Weather API",
    "sidebar_brand": "TrafficAI", "sidebar_subtitle": "Istanbul Traffic Platform",
    "sidebar_filters": "⚙️ Filters", "sidebar_year": "📅 Year",
    "sidebar_month": "📆 Month", "sidebar_day": "🗓️ Day", "sidebar_hour": "⏰ Hour",
    "sidebar_data_info": "📊 Dataset Info", "all_month": "All Month",
    "total_records": "Total Records", "vehicle_passage": "Vehicle Passages",
    "period": "Period", "monthly_file": "Monthly Files", "months_suffix": "months",
    "tab1": "📍 Density Map", "tab2": "📈 Time Analysis",
    "tab3": "📊 Year Comparison", "tab4": "🔮 AI Prediction + Weather",
    "tab5": "🤖 AI Models Comparison", "tab6": "📥 Add Data / Compare",
    "total_vehicles": "Total Vehicles", "avg_speed": "Average Speed",
    "active_locations": "Active Locations", "record_count": "Record Count",
    "active_days": "Active Days",
    "monthly_report": "Monthly Report Summary", "peak_hours": "🕐 Busiest Hours",
    "busiest_areas": "📍 Busiest Areas", "co2_emission": "🌿 Estimated CO₂ Emission",
    "per_vehicle": "Per vehicle", "speed_factor": "Speed factor",
    "map_title": "Traffic Density Map", "map_type": "Map Type",
    "color_by": "Color By", "by_vehicle": "Vehicle Count",
    "by_speed": "Avg Speed (inverse)", "hour_stats": "📊 This Hour's Statistics:",
    "map_reset": "🔄 Reset Map (Istanbul Overview)",
    "region_analysis": "Area Analysis",
    "slowest_5": "🔴 Slowest 5 Points", "fastest_5": "🟢 Fastest 5 Points",
    "most_congested_5": "🟠 Most Congested 5", "least_congested_5": "🟣 Least Congested 5",
    "go_btn": "📍 Go", "no_data": "No data found.",
    "time_analysis": "📈 Time Series Analysis",
    "hourly_profile": "Hourly Traffic Profile",
    "weekly_profile": "Weekly Traffic Profile",
    "monthly_trend": "📆 Monthly Traffic Trend (All Years)",
    "weekday_vs_weekend": "Weekday vs Weekend Comparison",
    "rush_analysis": "Rush Hour Analysis",
    "weekday": "Weekday", "weekend": "Weekend",
    "morning_rush": "Morning (07-09)", "evening_rush": "Evening (17-19)",
    "year_comparison": "📊 Traffic Time Comparison",
    "yoy": "📅 Year-over-Year Comparison (YoY)",
    "mom": "📆 Monthly Changes Within Year (MoM)",
    "base_year": "1st Year (Base/Reference)", "target_year": "2nd Year (Target/Comparison)",
    "pandemic_effect": "🦠 Pandemic Effect: 2020 vs Other Years",
    "pandemic": "Pandemic (2020)", "normal_period": "Normal Period",
    "improving_areas": "Improving vs Worsening Areas (Speed Change)",
    "speeding_districts": "Speeding Up Districts", "slowing_districts": "Slowing Down Districts",
    "increased": "increased", "decreased": "decreased",
    "avg_speed_short": "Avg Speed (km/h)", "avg_vehicles": "Avg Vehicle Count",
    "comparison_type": "Comparison Type", "year_to_inspect": "Year to Inspect",
    "two_month_comp": "📆 Two-Month Detailed Comparison",
    "month_1": "Month 1", "month_2": "Month 2",
    "ai_prediction": "🔮 AI-Powered Traffic Density Prediction",
    "ai_prediction_subtitle": "Select date, choose model, predict. For past dates, **actual data** comparison is shown automatically.",
    "date_and_hour": "📅 Date & Hour", "select_date": "📅 Select Date",
    "hour_slider": "🕐 Hour",
    "date_hint": "Selectable 2020-01-01 → 2030-12-31. Weather unavailable beyond 14 days.",
    "past_date_caption": "📂 Past date — actual data comparison will be shown",
    "future_date_caption": "🔮 Future date — Forecast API will be used",
    "model_selection": "🤖 AI Model Selection", "prediction_model": "Prediction Model",
    "accuracy_rate": "Accuracy Rate", "model_accuracy_fmt": "Accuracy: %",
    "weather_api_cb": "🌦️ Weather API", "predict_btn": "🔮 Predict",
    "forecast_api": "🔮 Forecast API", "archive_api": "📂 Archive API",
    "temperature": "Temperature", "precipitation": "Precipitation",
    "rain_impact_msg": "⚠️ Rain detected! Applied +",
    "rain_impact_suffix": "% load to density prediction.",
    "dry_weather_msg": "✅ Dry weather: No prediction adjustment applied.",
    "ai_pred_vs_actual": "Prediction vs Actual Data",
    "ai_prediction_title": "🔮 AI Prediction", "actual_title": "✅ Actual (That Day)",
    "vehicle_label": "Vehicles", "speed_label": "Speed",
    "status_label": "Status", "confidence_label": "Confidence",
    "vehicle_error": "Vehicle Error (%)", "speed_error": "Speed Error (%)",
    "accuracy": "Accuracy (%)", "pred_vehicles": "Predicted Vehicles",
    "pred_speed": "Predicted Speed", "co2_kg": "CO₂ (kg)",
    "all_models_comparison": "🔬 All Models Comparison for This Scenario",
    "pred_details": "Prediction Details", "model_label": "Model",
    "confidence_vehicles": "Confidence Interval (Vehicles)", "confidence_speed": "Confidence Interval (Speed)",
    "base_vehicles": "Base Vehicles (Raw)", "base_speed_lbl": "Base Speed",
    "sample_count": "Sample Count", "weather_effect": "Weather Effect",
    "co2_hourly": "🌿 Estimated CO₂ Emission (This Hour)",
    "signal_recommendation": "🚦 Dynamic Traffic Signal Recommendation",
    "signal_warning": "Average speed", "signal_action": "Increase green light duration by +",
    "signal_action2": "seconds for main arterials.",
    "wait_reduction": "Estimated wait time reduction: ~%",
    "route_optimization": "### 🗺️ Route Optimization Suggestion",
    "start_district_lbl": "🟢 Start District", "end_district_lbl": "🔴 Destination District",
    "route_alternatives": "#### 📋 Route Alternatives",
    "route_map_title": "#### 🗺️ Route Map",
    "select_route_map": "🗺️ Select Route to Display on Map",
    "recommended_badge": "✅ RECOMMENDED",
    "predict_first": "👆 Select parameters and click **Predict**.",
    "osrm_fallback": "🌐 Could not reach OpenStreetMap route service (OSRM) — distances simulated.",
    "different_districts": "Please select different start and destination districts.",
    "no_route_geometry": "Route geometry unavailable — route card values are valid.",
    "insuff_data": "Insufficient data for this combination.",
    "vehicle_count_label": "Vehicle Count",
    "ai_engine_title": "🧠 About the AI Prediction Engine",
    "ai_engine_desc": """This tab was developed as an <i>"AI-Powered Traffic Density Prediction Report"</i>.
               The prediction engine uses:<br>
               • 📊 <b>100 Million Rows</b> of IBB historical traffic data<br>
               • 🌦️ <b>Open-Meteo API</b> — historical (Archive) + future (Forecast) weather<br>
               • 🤖 <b>4 Different AI Models:</b> CNN-LSTM, XGBoost, Random Forest, ARIMA<br>
               • ✅ <b>Actual Data Comparison:</b> Ground Truth from dataset for past dates<br>
               • 🌿 <b>CO₂ Emission Estimation</b> + 🗺️ <b>Route Optimization</b><br>
               • 🚦 <b>Dynamic Traffic Signal Recommendation</b>""",
    "ai_models": "### 🤖 AI Models Comparison",
    "ai_models_subtitle": "Performance metrics of different AI models and prediction results on the same data.",
    "model_metrics": "#### 📊 Model Performance Metrics",
    "model_charts": "#### 📈 Model Comparison Charts",
    "model_accuracy_chart": "Model Accuracy Rates (%)",
    "model_error_chart": "Model Error Metrics (MAE & RMSE)",
    "all_models_predict": "#### 🔮 Predict with All Models Simultaneously",
    "compare_btn": "🔬 Compare", "select_date2": "📅 Date",
    "radar_chart": "#### 🎯 Model Capability Radar Chart",
    "radar_cats": ["Accuracy","Speed","Generalization","Interpretability","Flexibility"],
    "actual_label": "Actual", "pred_vehicles_chart": "Predicted Vehicle Count",
    "pred_speed_chart": "Predicted Speed (km/h)",
    "compare_first": "👆 Select parameters and click **Compare**.",
    "data_upload": "### 📥 Add Data / Compare",
    "data_upload_desc": "Upload CSV files from the desktop app and compare with the IBB dataset. Data is stored **only for this session**.",
    "upload_csv": "📂 Upload CSV File (traffic_density_*.csv)",
    "loaded_data": "**📋 Loaded Data:**", "no_data_loaded": "No data loaded yet.",
    "delete_all": "🗑️ Clear All",
    "comparison_results": "### 📊 Comparison Results",
    "region_label": "📍 Region", "pattern_match": "🔁 Pattern Match",
    "hourly_vehicles_label": "🚗 Hourly Vehicles: You / IBB",
    "speed_diff": "⚡ Avg Speed Difference",
    "no_geohash_match": "No geohash match found — no records for these locations in the IBB dataset.",
    "upload_required_cols": "Uploaded CSV requires `geohash` and `hour` columns.",
    "upload_success": "uploaded!", "records_suffix": "records",
    "missing_cols": "Missing columns", "file_read_error": "File could not be read",
    "compatible_expected": "Dataset-compatible CSV expected.",
    "ibb_label": "🏛️ IBB Historical Pattern", "your_measurement": "📹 Your Measurement",
    "density_index": "Density Index (average = 100)", "hour_axis": "Hour",
    "regional_analysis_short": "Regional Analysis — Short Recording Mode",
    "no_ibb_history": "No IBB historical data found for this area.",
    "your_speed_lbl": "Your Speed", "ibb_speed_lbl": "IBB Speed",
    "ibb_avg_speed": "IBB Avg (hourly)", "your_vehicles": "Your Vehicles (hourly)",
    "index_diff": "Index Diff", "speed_diff_col": "Speed Diff",
    "basis_col": "Basis", "district_col": "District", "geohash_col": "Geohash",
    "hour_col": "Hour",
    "km_h": "km/h", "ton_co2": "tons CO₂", "vehicles_lbl": "vehicles",
    "level_very_congested": "🔴 VERY CONGESTED", "level_congested": "🟠 CONGESTED",
    "level_moderate": "🟡 MODERATE", "level_flowing": "🟢 FLOWING",
    "level_congested_route": "🔴 Congested", "level_moderate_route": "🟠 Moderate",
    "level_flowing_route": "🟢 Flowing",
    "day_names": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
    "months": {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"},
    "wmo": {
        0:("☀️","Clear / Sunny"),1:("🌤️","Mainly Clear"),2:("⛅","Partly Cloudy"),
        3:("☁️","Overcast"),45:("🌫️","Foggy"),48:("🌫️","Rime Fog"),
        51:("🌦️","Light Drizzle"),53:("🌦️","Moderate Drizzle"),55:("🌧️","Dense Drizzle"),
        61:("🌧️","Slight Rain"),63:("🌧️","Moderate Rain"),65:("🌧️","Heavy Rain"),
        71:("🌨️","Slight Snow"),73:("🌨️","Moderate Snow"),75:("❄️","Heavy Snow"),
        80:("🌧️","Rain Showers"),95:("⛈️","Thunderstorm"),
    },
    "model_descriptions": {
        "CNN-LSTM": "CNN-LSTM — Hybrid deep learning combining convolutional and long-short term memory networks for time series.",
        "Gradient Boosting": "Gradient Boosting — Powerful sequential tree ensemble. Highest accuracy.",
        "Random Forest": "Random Forest — Generalization with multiple decision trees. Balanced performance.",
        "XGBoost": "eXtreme Gradient Boosting — Fast and efficient boosting model.",
        "Ridge Regression": "Ridge (L2) Linear Regression — Simple and fast baseline model.",
    },
    "footer": "© 2026 All Rights Reserved — Project developed by Abdurrahman Bayrakdar and Şevket Can Yorulmaz.",
    "unknown": "Unknown", "avg_speed_label": "Avg Speed (km/h)",
    "avg_vehicles_label": "Avg Vehicle Count", "month_axis": "Month",
    "change_pct": "Change (%)", "yoy_growth_rate": "Year-over-Year Growth Rate",
    "mom_growth_rate": "Month-over-Month Growth Rate (MoM)",
    "speed_trend": "Year Monthly Speed Trend", "density_trend": "Year Monthly Density Trend",
    "pandemic_speed": "Average Speed", "pandemic_vehicles": "Average Vehicle Count",
    "vehicle_count_change": "Vehicle Count Change", "speed_change_lbl": "Average Speed Change",
    "weather_fetching": "Fetching weather data from Open-Meteo API...",
    "weather_no_data": "Weather set to 'Clear/Normal' as the date is beyond the 14-day forecast limit.",
    "your_hour_rank": "This Hour's Rank in", "your_vs_history": "Speed: You vs",
    "your_vs_history2": "History", "busiest_year": "Busiest Year in",
    "rank_in_day": "Day's busiest hour #", "rank_suffix": "",
    "osrm_start": "🟢 Start:", "osrm_end": "🏁 Destination:",
    "actual_occurred": "Actual that day",
    "consistent_msg": "measurement is **consistent** with IBB's hourly average",
    "higher_msg": "measurement is",
    "more_dense": "denser than IBB's hourly average for these conditions",
    "calmer": "calmer than IBB's hourly average for these conditions",
    "no_data_warning": "No data found for this filter.",
    "comparison_basis": "Comparison basis: IBB dataset's",
    "basis_month_day_avg": "days average",
    "basis_month_avg": "month average (insufficient daily records)",
    "basis_general_avg": "overall average (insufficient monthly records)",
    "hourly_vehicles_short": "Vehicle Count", "hour_spd": "Speed (km/h)",
    "avg_speed_sec": "Avg Speed", "vehicle_sec": "Avg Vehicles",
    "rush_type_lbl": "Rush Type",
    "short_rec_caption": "Few hours in recording; showing the area's IBB historical data.",
    "day_hour_profile_lbl": "Days Hourly Profile",
    # ── Aliases / new keys added for full i18n ──
    "date_and_time": "Date & Time", "hour": "Hour",
    "date_help": "Selectable 2020-01-01 → 2030-12-31. Weather unavailable beyond 14 days.",
    "ai_model_select": "AI Model Selection", "weather_api": "Weather API",
    "weather_unavailable": "Weather set to 'Clear/Normal' as the date is beyond the 14-day forecast limit.",
    "no_data_combination": "Insufficient data for this combination.",
    "weather_title": "Weather", "rain": "Precipitation",
    "rain_detected": "Rain detected! Applied +",
    "dry_weather": "Dry weather: No prediction adjustment applied",
    "very_heavy": "🔴 VERY CONGESTED", "heavy": "🟠 CONGESTED",
    "moderate": "🟡 MODERATE", "flowing": "🟢 FLOWING",
    "prediction_vs_actual": "Prediction vs Actual Data",
    "ai_pred_label": "AI Prediction", "vehicle_lbl": "Vehicles",
    "speed_lbl": "Speed", "status_lbl": "Status",
    "confidence": "Confidence", "vehicles_unit": "vehicles",
    "vehicle_error_pct": "Vehicle Error (%)", "speed_error_pct": "Speed Error (%)",
    "prediction_result": "Prediction Result",
    "estimated_vehicle": "Predicted Vehicles", "estimated_speed": "Predicted Speed",
    "model_lbl": "Model", "date_lbl": "Date",
    "ci_vehicle": "Confidence Interval (Vehicles)", "ci_speed": "Confidence Interval (Speed)",
    "base_vehicle": "Base Vehicles (Raw)", "base_speed": "Base Speed",
    "hourly_records": "hourly records", "weather_impact": "Weather Effect",
    "all_models_compare": "All Models Comparison for This Scenario",
    "per_vehicle": "Per vehicle", "speed_factor": "Speed factor",
    "co2_title": "Estimated CO₂ Emission (This Hour)",
    "signal_title": "Dynamic Traffic Signal Recommendation",
    "green_light_advice": "Increase green light duration by +{extra_green} seconds for main arterials.",
    "wait_reduction": "Estimated wait time reduction",
    "hourly_profile": "Days Hourly Profile",
    "vehicle_count_lbl": "Vehicle Count", "avg_speed_lbl_short": "Avg Speed",
    "rainy": "Rainy", "speed_kmh": "Speed (km/h)",
    "start_district": "Start District", "end_district": "Destination District",
    "osrm_unavailable": "Could not reach OpenStreetMap route service (OSRM) — distances simulated.",
    "alt_routes": "Route Alternatives",
    "select_route_map": "Select Route to Display on Map",
    "recommended": "RECOMMENDED", "min": "min", "route_map": "Route Map",
    "start_label": "Start", "end_label": "Destination",
    "route_geometry_missing": "Route geometry unavailable — route card values are valid.",
    "select_diff_districts": "Please select different start and destination districts.",
    "predict_hint": "Select parameters and click **Predict**.",
    "route_optimization": "Route Optimization Suggestion",
    "ai_models_compare": "AI Models Comparison",
    "model_performance": "Model Performance Metrics",
    "model_compare_charts": "Model Comparison Charts",
    "model_accuracy_title": "Model Accuracy Rates (%)",
    "model_error_title": "Model Error Metrics (MAE & RMSE)",
    "all_models_predict": "Predict with All Models Simultaneously",
    "compare_hint": "Select parameters and click **Compare**.",
    "radar_chart_title": "Model Capability Radar Chart",
    "radar_categories": ["Accuracy","Speed","Generalization","Interpretability","Flexibility"],
    "actual_on_day": "Actual that day", "actual_short": "Actual",
    "data_upload_title": "Add Data / Compare",
    "data_upload_subtitle": "Upload CSV files from the desktop app and compare with the IBB dataset. Data is stored **only for this session**.",
    "upload_csv_label": "Upload CSV File (traffic_density_*.csv)",
    "uploaded": "uploaded", "records": "records",
    "csv_format_required": "Dataset-compatible CSV expected.",
    "uploaded_data": "Loaded Data", "no_data_uploaded": "No data loaded yet.",
    "clear_all": "Clear All", "compare_results": "Comparison Results",
    "region": "Region", "multiple_regions": "multiple regions",
    "pattern_match_help": "Correlation of your hourly density shape with the IBB profile (≥3 different hours required)",
    "hourly_vehicle_you_ibb": "Hourly Vehicles: You / IBB",
    "hourly_vehicle_help": "Your count converted to hourly rate; IBB value is the area's hourly sum average",
    "avg_speed_diff": "Avg Speed Difference",
    "basis_month_day": "{month}, {day} days average",
    "basis_month": "{month} average (insufficient daily records)",
    "basis_general": "overall average (insufficient monthly records)",
    "ibb_hist_pattern": "IBB Historical Pattern",
    "regional_analysis": "Regional Analysis", "short_record_mode": "Short Recording Mode",
    "hour_rank_in": "This Hour's Rank in {district}",
    "hour_rank_help": "Density rank of your measurement hour in the 24h IBB profile of {district}",
    "speed_you_vs": "Speed: You vs {district} History",
    "busiest_year_in": "Busiest Year in {district}",
    "busiest_year_help": "Year with highest average density at hours {hours} in this area",
    "avg_vehicle_lbl": "Avg Vehicles", "ibb_speed": "IBB Speed",
    "your_speed": "Your Speed", "by_year": "By Year", "by_day": "By Day",
    "your_day_marker": "your measurement day",
    "24h_profile": "24h Profile", "your_hour_marker": "your measurement hour",
    "col_district": "District", "col_hour": "Hour",
    "col_your_vehicle": "Your Vehicles (hourly)", "col_ibb_vehicle": "IBB Avg (hourly)",
    "col_index_diff": "Index Diff", "col_your_speed": "Your Speed",
    "col_ibb_speed": "IBB Speed", "col_speed_diff": "Speed Diff", "col_basis": "Basis",
    "your_vehicle_help": "Your count converted to hourly rate",
    "ibb_vehicle_help": "Area's hourly sum average from IBB dataset",
    "index_diff_help": "Positive: your count is denser than IBB pattern for this hour",
    "compare_compatible": "✅ {district} measurement is **consistent** with IBB's hourly average ({pct}%).",
    "compare_denser": "⚠️ {district} measurement is **%{pct} denser** than IBB's hourly average for these conditions.",
    "compare_lighter": "ℹ️ {district} measurement is **%{pct} calmer** than IBB's hourly average for these conditions.",
    "csv_geohash_required": "Uploaded CSV requires `geohash` and `hour` columns.",
    "short_rec_caption_long": "Your recording has few hours ({hours}) — showing IBB history for **{district}** (geohash `{geohash}`) across years / weekdays / hours. All charts are specific to this area.",
}

TRANSLATIONS = {"tr": LANG_TR, "en": LANG_EN}


def T(key, lang=None):
    """Çeviri yardımcısı / Translation helper."""
    if lang is None:
        lang = st.session_state.get("lang", "tr")
    tr = TRANSLATIONS.get(lang, LANG_TR)
    val = tr.get(key)
    if val is None:
        val = LANG_TR.get(key, key)
    return val

# ─────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_parquet")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(
    page_title="İstanbul Trafik Analiz Platformu",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state defaults
for key, default in {
    "fly_lat": 41.015, "fly_lon": 29.01, "fly_zoom": 10,
    "fly_target_lat": None, "fly_target_lon": None,
    "prev_year": None, "prev_month": None, "prev_hour": None, "prev_day": None,
    "prediction_base": None, "lang": "tr",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────
#  MODERN UI CSS YÜKLEME
# ─────────────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "modern_style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



def _style_fig(fig):
    """Modern UI açıkken tüm Plotly grafiklerine uygulanan ortak görsel stil."""
    try:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, 'Segoe UI', sans-serif", color="#cbd5e1", size=12),
            title_font=dict(family="'Space Grotesk', Inter, sans-serif", size=15, color="#e2e8f0"),
            hoverlabel=dict(
                bgcolor="rgba(13,16,32,0.95)", bordercolor="#667eea",
                font=dict(family="Inter, 'Segoe UI', sans-serif", size=12, color="#e2e8f0"),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(gridcolor="rgba(102,126,234,0.10)", zerolinecolor="rgba(102,126,234,0.20)")
        fig.update_yaxes(gridcolor="rgba(102,126,234,0.10)", zerolinecolor="rgba(102,126,234,0.20)")
    except Exception:
        pass
    return fig


# st.plotly_chart'ı sarmala: modern modda grafikler otomatik stillenir
_orig_plotly_chart = st.plotly_chart

def _plotly_chart_auto(fig, *args, **kwargs):
    if st.session_state.get("ui_modern", True):
        _style_fig(fig)
    return _orig_plotly_chart(fig, *args, **kwargs)

st.plotly_chart = _plotly_chart_auto


# ─────────────────────────────────────────────────────────────────────────
#  DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_monthly():
    p = os.path.join(DATA_DIR, "summary_monthly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"SELECT * FROM '{p}' ORDER BY year, month").df()
    db.close()
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_hourly():
    p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"SELECT * FROM '{p}'").df()
    db.close()
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_geo_hourly_filtered(year, month, hour):
    p = os.path.join(DATA_DIR, "summary_geo_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT lat, lon, geohash,
               SUM(total_vehicles) AS total_vehicles,
               AVG(avg_speed) AS avg_speed,
               SUM(record_count) AS record_count
        FROM '{p}'
        WHERE year = {year} AND month = {month} AND hour = {hour}
        GROUP BY lat, lon, geohash
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_geo_daily_filtered(year, month, day, hour):
    """Belirli bir gün için traffic_all.parquet'ı sorgular."""
    p = os.path.join(DATA_DIR, "traffic_all.parquet").replace("\\", "/")
    date_str = f"{year}-{month:02d}-{day:02d}"
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT lat, lon, geohash,
               SUM(vehicle_count) AS total_vehicles,
               AVG(avg_speed) AS avg_speed,
               COUNT(*) AS record_count
        FROM '{p}'
        WHERE CAST(date_time AS DATE) = '{date_str}' AND hour = {hour}
        GROUP BY lat, lon, geohash
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_comparison_baseline(gh5_tuple, month, dow):
    """Yüklenen saha verisiyle adil karşılaştırma için İBB tarihsel ortalamaları.
    Tek taramada 3 seviye hesaplanır, öncelik sırası: aynı ay + haftanın günü →
    aynı ay → genel ortalama. dow: Python weekday (Pzt=0). Geohash 5 haneye
    normalize edilir (masaüstü 6, dataset 5 hane uyumsuzluğunu giderir)."""
    if not gh5_tuple:
        return pd.DataFrame()
    p = os.path.join(DATA_DIR, "traffic_all.parquet").replace("\\", "/")
    gh_list = ",".join(f"'{g}'" for g in gh5_tuple)
    duck_dow = (int(dow) + 1) % 7  # DuckDB dayofweek: 0=Pazar
    db = duckdb.connect()
    # Önce her (bölge, gün, saat) için ALAN TOPLAMI alınır (alt-hücre ortalaması
    # değil) — kameranın gördüğü tüm yol trafiğiyle aynı ölçek. Sonra günler
    # arasında ortalanır → "o bölgede o saatte saatlik ortalama araç".
    df = db.execute(f"""
        SELECT gh5, hour,
               AVG(CASE WHEN m = {int(month)} AND dw = {duck_dow} THEN vc END) AS v_dow,
               AVG(CASE WHEN m = {int(month)} THEN vc END) AS v_month,
               AVG(vc) AS v_all,
               AVG(CASE WHEN m = {int(month)} AND dw = {duck_dow} THEN sp END) AS s_dow,
               AVG(CASE WHEN m = {int(month)} THEN sp END) AS s_month,
               AVG(sp) AS s_all
        FROM (
            SELECT gh5, d, hour, m, dw,
                   SUM(vehicle_count) AS vc,
                   AVG(avg_speed) AS sp
            FROM (
                SELECT substr(geohash, 1, 5) AS gh5, hour, vehicle_count, avg_speed,
                       CAST(date_time AS DATE) AS d,
                       month(CAST(date_time AS TIMESTAMP)) AS m,
                       dayofweek(CAST(date_time AS DATE)) AS dw
                FROM '{p}'
                WHERE substr(geohash, 1, 5) IN ({gh_list})
            )
            GROUP BY gh5, d, hour, m, dw
        )
        GROUP BY gh5, hour
    """).df()
    db.close()
    if df.empty:
        return df
    df["hist_vehicles"] = df["v_dow"].fillna(df["v_month"]).fillna(df["v_all"])
    df["hist_speed"] = df["s_dow"].fillna(df["s_month"]).fillna(df["s_all"])
    df["basis"] = np.where(df["v_dow"].notna(), "Ay+Gün",
                  np.where(df["v_month"].notna(), "Ay", "Genel"))
    return df[["gh5", "hour", "hist_vehicles", "hist_speed", "basis"]]


@st.cache_data(ttl=3600, show_spinner=False)
def load_location_history(gh5_tuple):
    """Belirli bölge(ler) için yıl/ay/haftanın günü/saat kırılımlı İBB geçmişi.
    Kısa kayıtların (tek saatlik ölçüm vb.) çok parametreli bölgesel analizinde kullanılır."""
    if not gh5_tuple:
        return pd.DataFrame()
    p = os.path.join(DATA_DIR, "traffic_all.parquet").replace("\\", "/")
    gh_list = ",".join(f"'{g}'" for g in gh5_tuple)
    db = duckdb.connect()
    # Saatlik ALAN TOPLAMI (gün bazında), sonra yıl/ay/gün/saat kırılımında ortalama
    df = db.execute(f"""
        SELECT y, m, dw, hour,
               AVG(vc) AS v,
               AVG(sp) AS s,
               COUNT(*) AS n
        FROM (
            SELECT year(CAST(date_time AS TIMESTAMP)) AS y,
                   month(CAST(date_time AS TIMESTAMP)) AS m,
                   dayofweek(CAST(date_time AS DATE)) AS dw,
                   CAST(date_time AS DATE) AS d,
                   hour,
                   SUM(vehicle_count) AS vc,
                   AVG(avg_speed) AS sp
            FROM '{p}'
            WHERE substr(geohash, 1, 5) IN ({gh_list})
            GROUP BY 1, 2, 3, 4, 5
        )
        GROUP BY y, m, dw, hour
    """).df()
    db.close()
    if not df.empty:
        df["dw_py"] = (df["dw"].astype(int) + 6) % 7  # DuckDB Pazar=0 → Python Pzt=0
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_hourly_profile(year=None):
    p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    where = f"WHERE year = {year}" if year else ""
    df = db.execute(f"""
        SELECT hour,
               AVG(total_vehicles) AS avg_vehicles,
               AVG(city_avg_speed) AS avg_speed,
               AVG(active_locations) AS avg_locations
        FROM '{p}' {where}
        GROUP BY hour ORDER BY hour
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_dow_profile(year=None):
    p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    where = f"WHERE year = {year}" if year else ""
    df = db.execute(f"""
        SELECT day_of_week,
               AVG(total_vehicles) AS avg_vehicles,
               AVG(city_avg_speed) AS avg_speed
        FROM '{p}' {where}
        GROUP BY day_of_week ORDER BY day_of_week
    """).df()
    db.close()
    _dn_list = T("day_names")
    day_names_map = {i: _dn_list[i] for i in range(7)}
    df["day_name"] = df["day_of_week"].map(day_names_map)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_top_slowest(year, month, n=5):
    p = os.path.join(DATA_DIR, "summary_geo_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT geohash, AVG(lat) AS lat, AVG(lon) AS lon,
               SUM(total_vehicles) AS total_vehicles,
               AVG(avg_speed) AS avg_speed
        FROM '{p}'
        WHERE year = {year} AND month = {month}
        GROUP BY geohash ORDER BY avg_speed ASC LIMIT {n}
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_top_fastest(year, month, n=5):
    p = os.path.join(DATA_DIR, "summary_geo_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT geohash, AVG(lat) AS lat, AVG(lon) AS lon,
               SUM(total_vehicles) AS total_vehicles,
               AVG(avg_speed) AS avg_speed
        FROM '{p}'
        WHERE year = {year} AND month = {month}
        GROUP BY geohash ORDER BY avg_speed DESC LIMIT {n}
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_top_congested(year, month, n=5):
    p = os.path.join(DATA_DIR, "summary_geo_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT geohash, AVG(lat) AS lat, AVG(lon) AS lon,
               SUM(total_vehicles) AS total_vehicles,
               AVG(avg_speed) AS avg_speed
        FROM '{p}'
        WHERE year = {year} AND month = {month}
        GROUP BY geohash ORDER BY total_vehicles DESC LIMIT {n}
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_least_congested(year, month, n=5):
    p = os.path.join(DATA_DIR, "summary_geo_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT geohash, AVG(lat) AS lat, AVG(lon) AS lon,
               SUM(total_vehicles) AS total_vehicles,
               AVG(avg_speed) AS avg_speed
        FROM '{p}'
        WHERE year = {year} AND month = {month}
        GROUP BY geohash ORDER BY total_vehicles ASC LIMIT {n}
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_year_comparison():
    p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT year, month, hour,
               AVG(total_vehicles) AS avg_vehicles,
               AVG(city_avg_speed) AS avg_speed
        FROM '{p}'
        GROUP BY year, month, hour ORDER BY year, month, hour
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_monthly_peak_hour(year, month):
    p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT hour,
               AVG(total_vehicles) AS avg_vehicles,
               AVG(city_avg_speed) AS avg_speed
        FROM '{p}'
        WHERE year = {year} AND month = {month}
        GROUP BY hour ORDER BY avg_vehicles DESC
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_monthly_top_areas(year, month, n=5):
    p = os.path.join(DATA_DIR, "summary_geo_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT geohash, AVG(lat) AS lat, AVG(lon) AS lon,
               SUM(total_vehicles) AS total_vehicles,
               AVG(avg_speed) AS avg_speed
        FROM '{p}'
        WHERE year = {year} AND month = {month}
        GROUP BY geohash ORDER BY total_vehicles DESC LIMIT {n}
    """).df()
    db.close()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_ground_truth(target_date_str, hour):
    """Belirli bir tarih ve saat için gerçek veriyi çeker."""
    p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"""
        SELECT total_vehicles, city_avg_speed, active_locations
        FROM '{p}'
        WHERE CAST(hour_ts AS DATE) = '{target_date_str}'
          AND hour = {hour}
    """).df()
    db.close()
    if not df.empty:
        return {
            "vehicles": df.iloc[0]["total_vehicles"],
            "speed": df.iloc[0]["city_avg_speed"],
            "locations": df.iloc[0]["active_locations"],
        }
    return None


# ─────────────────────────────────────────────────────────────────────────
#  WEATHER API (Open-Meteo - Historical + Forecast)
# ─────────────────────────────────────────────────────────────────────────
WMO_CODES = T("wmo")


def get_hourly_weather(lat, lon, target_date_str, hour):
    """Belirli bir saat için hava durumunu çeker (geçmiş veya gelecek)."""
    today = date.today()
    target = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    try:
        if target <= today:
            url = (f"https://archive-api.open-meteo.com/v1/archive"
                   f"?latitude={lat}&longitude={lon}"
                   f"&start_date={target_date_str}&end_date={target_date_str}"
                   f"&hourly=temperature_2m,rain,weathercode"
                   f"&timezone=Europe/Istanbul")
        else:
            url = (f"https://api.open-meteo.com/v1/forecast"
                   f"?latitude={lat}&longitude={lon}"
                   f"&start_date={target_date_str}&end_date={target_date_str}"
                   f"&hourly=temperature_2m,rain,weather_code"
                   f"&timezone=Europe/Istanbul")

        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "hourly" not in data:
            return None

        temps = data["hourly"]["temperature_2m"]
        rains = data["hourly"]["rain"]
        codes = data["hourly"].get("weathercode", data["hourly"].get("weather_code", [0]*24))

        if hour < len(temps):
            temp = temps[hour]
            rain = rains[hour] if rains[hour] is not None else 0
            code = codes[hour] if codes[hour] is not None else 0
            icon, desc = WMO_CODES.get(code, ("🌡️", f"Kod: {code}"))

            rain_impact = 0.0
            if rain > 0.5: rain_impact = 0.12
            if rain > 5.0: rain_impact = 0.20
            if code in (71, 73, 75): rain_impact = 0.25

            return {
                "temperature": temp, "rain": round(rain, 1),
                "icon": icon, "description": desc,
                "rain_impact": rain_impact,
                "is_forecast": target > today,
            }
        return None
    except:
        return None


# ─────────────────────────────────────────────────────────────────────────
#  GERÇEK ML MODELLERİ (Eğitilmiş)
# ─────────────────────────────────────────────────────────────────────────

# Eğitim sonuçlarını yükle
_training_results_path = os.path.join(MODEL_DIR, "training_results.json")
_trained_models_cache = {}

@st.cache_data(show_spinner=False)
def _load_training_results():
    if os.path.exists(_training_results_path):
        with open(_training_results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _get_trained_metrics():
    results = _load_training_results()
    metrics = {}
    for r in results:
        name = r["name"]
        v_metrics = r["metrics"].get("total_vehicles", {})
        s_metrics = r["metrics"].get("city_avg_speed", {})
        metrics[name] = {
            "accuracy": r["accuracy"],
            "mae_vehicles": v_metrics.get("mae", 0),
            "rmse_vehicles": v_metrics.get("rmse", 0),
            "r2_vehicles": v_metrics.get("r2", 0),
            "mae_speed": s_metrics.get("mae", 0),
            "rmse_speed": s_metrics.get("rmse", 0),
            "r2_speed": s_metrics.get("r2", 0),
        }
    return metrics

_real_metrics = _get_trained_metrics()

AI_MODELS = {
    "CNN-LSTM": {
        "accuracy": _real_metrics.get("CNN-LSTM", {}).get("accuracy", 74.5),
        "mae": _real_metrics.get("CNN-LSTM", {}).get("mae_speed", 1.20),
        "rmse": _real_metrics.get("CNN-LSTM", {}).get("rmse_speed", 1.78),
        "mae_vehicles": _real_metrics.get("CNN-LSTM", {}).get("mae_vehicles", 25766),
        "rmse_vehicles": _real_metrics.get("CNN-LSTM", {}).get("rmse_vehicles", 42147),
        "icon": "🧠", "color": "#a855f7",
        "pkl_name": "lstm_model.pth",
    },
    "Gradient Boosting": {
        "accuracy": _real_metrics.get("Gradient Boosting", {}).get("accuracy", 82.6),
        "mae": _real_metrics.get("Gradient Boosting", {}).get("mae_speed", 0.92),
        "rmse": _real_metrics.get("Gradient Boosting", {}).get("rmse_speed", 1.38),
        "mae_vehicles": _real_metrics.get("Gradient Boosting", {}).get("mae_vehicles", 20179),
        "rmse_vehicles": _real_metrics.get("Gradient Boosting", {}).get("rmse_vehicles", 34209),
        "icon": "🚀", "color": "#667eea",
        "pkl_name": "gradient_boosting.pkl",
    },
    "Random Forest": {
        "accuracy": _real_metrics.get("Random Forest", {}).get("accuracy", 80.4),
        "mae": _real_metrics.get("Random Forest", {}).get("mae_speed", 0.97),
        "rmse": _real_metrics.get("Random Forest", {}).get("rmse_speed", 1.46),
        "mae_vehicles": _real_metrics.get("Random Forest", {}).get("mae_vehicles", 21297),
        "rmse_vehicles": _real_metrics.get("Random Forest", {}).get("rmse_vehicles", 36151),
        "icon": "🌳", "color": "#22c55e",
        "pkl_name": "random_forest.pkl",
    },
    "XGBoost": {
        "accuracy": _real_metrics.get("XGBoost", {}).get("accuracy", 78.5),
        "mae": _real_metrics.get("XGBoost", {}).get("mae_speed", 1.01),
        "rmse": _real_metrics.get("XGBoost", {}).get("rmse_speed", 1.53),
        "mae_vehicles": _real_metrics.get("XGBoost", {}).get("mae_vehicles", 22269),
        "rmse_vehicles": _real_metrics.get("XGBoost", {}).get("rmse_vehicles", 37995),
        "icon": "🌲", "color": "#f97316",
        "pkl_name": "xgboost.pkl",
    },
    "Ridge Regression": {
        "accuracy": _real_metrics.get("Ridge Regression", {}).get("accuracy", 57.6),
        "mae": _real_metrics.get("Ridge Regression", {}).get("mae_speed", 1.40),
        "rmse": _real_metrics.get("Ridge Regression", {}).get("rmse_speed", 1.90),
        "mae_vehicles": _real_metrics.get("Ridge Regression", {}).get("mae_vehicles", 45374),
        "rmse_vehicles": _real_metrics.get("Ridge Regression", {}).get("rmse_vehicles", 61072),
        "icon": "📐", "color": "#ef4444",
        "pkl_name": "ridge_regression.pkl",
    },
}

class TrafficCNNLSTMModel(nn.Module):
    def __init__(self, input_dim=17, num_filters=32, kernel_size=3, hidden_dim=64, num_layers=2, output_dim=2):
        super(TrafficCNNLSTMModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(num_filters, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) -> transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        # transpose back to (batch_size, seq_len, num_filters)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

@st.cache_resource
def load_pytorch_model():
    """Eğitilmiş PyTorch CNN-LSTM modelini diskten yükler."""
    model_path = os.path.join(MODEL_DIR, "lstm_model.pth")
    if os.path.exists(model_path):
        model = TrafficCNNLSTMModel(input_dim=17, num_filters=32, kernel_size=3, hidden_dim=64, num_layers=2, output_dim=2)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None


@st.cache_resource
def load_ml_model(pkl_name):
    """Eğitilmiş ML modelini diskten yükler."""
    model_path = os.path.join(MODEL_DIR, pkl_name)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_scaler():
    """Eğitim sırasında kullanılan StandardScaler'ı yükler."""
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


def build_features(hour, day_of_week, month, year):
    """Tek bir zaman dilimi için feature vektörü oluşturur (train_models.py ile aynı sırada)."""
    return np.array([[
        float(hour),
        float(day_of_week),
        float(month),
        float(year),
        np.sin(2 * np.pi * hour / 24),      # hour_sin
        np.cos(2 * np.pi * hour / 24),      # hour_cos
        np.sin(2 * np.pi * day_of_week / 7), # dow_sin
        np.cos(2 * np.pi * day_of_week / 7), # dow_cos
        np.sin(2 * np.pi * month / 12),      # month_sin
        np.cos(2 * np.pi * month / 12),      # month_cos
        float(day_of_week >= 5),              # is_weekend
        float(7 <= hour <= 9),                # is_rush_morning
        float(17 <= hour <= 19),              # is_rush_evening
        float(hour >= 23 or hour <= 5),       # is_night
        float(month in (6, 7, 8)),            # is_summer
        float(month in (12, 1, 2)),           # is_winter
        (year - 2020) / 5.0,                  # year_norm
    ]])


@st.cache_data(show_spinner=False)
def real_model_predict(model_name, hour, day_of_week, month, year, rain_impact=0.0):
    """Gerçek eğitilmiş ML modeli ile tahmin yapar.
    Önbellekli: aynı parametrelerle tekrar çağrı anında döner (model değiştirme hızlı)."""
    model_info = AI_MODELS[model_name]
    scaler = load_scaler()

    if model_name == "CNN-LSTM":
        # PyTorch CNN-LSTM prediction logic
        model = load_pytorch_model()
        if model is None or scaler is None:
            return _fallback_predict(model_name, hour, day_of_week, month, year, rain_impact)
            
        # Find matching day in the month
        found_day = 1
        for d in range(1, 8):
            try:
                if datetime(year, month, d).weekday() == day_of_week:
                    found_day = d
                    break
            except ValueError:
                pass
        dt_target = datetime(year, month, found_day, hour)
        
        # Build 24 hour feature sequence (t-23, t-22, ..., t)
        seq_features = []
        for i in range(23, -1, -1):
            dt_step = dt_target - timedelta(hours=i)
            feat = build_features(dt_step.hour, dt_step.weekday(), dt_step.month, dt_step.year)
            feat_scaled = scaler.transform(feat)
            seq_features.append(feat_scaled[0])
            
        seq_tensor = torch.tensor([seq_features], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(seq_tensor).numpy()
            
        # Inverse scaling
        pred_vehicles = float(prediction[0][0] * 100000.0)
        pred_speed = float(prediction[0][1] * 50.0)
    else:
        # Traditional model prediction logic
        ml_model = load_ml_model(model_info["pkl_name"])
        if ml_model is None or scaler is None:
            return _fallback_predict(model_name, hour, day_of_week, month, year, rain_impact)

        X = build_features(hour, day_of_week, month, year)
        X_scaled = scaler.transform(X)
        prediction = ml_model.predict(X_scaled)

        pred_vehicles = float(prediction[0][0])
        pred_speed = float(prediction[0][1])

    # Hava durumu etkisi
    if rain_impact > 0:
        pred_vehicles *= (1 + rain_impact)
        pred_speed *= (1 - rain_impact * 0.8)

    # Güven aralığı (model doğruluğuna göre)
    accuracy = model_info["accuracy"]
    confidence_width = (100 - accuracy) / 100
    ci_vehicles = abs(pred_vehicles) * confidence_width
    ci_speed = abs(pred_speed) * confidence_width

    return {
        "vehicles": max(0.0, pred_vehicles),
        "speed": max(0.0, pred_speed),
        "ci_vehicles_low": max(0.0, pred_vehicles - ci_vehicles),
        "ci_vehicles_high": pred_vehicles + ci_vehicles,
        "ci_speed_low": max(0.0, pred_speed - ci_speed),
        "ci_speed_high": pred_speed + ci_speed,
    }


def _fallback_predict(model_name, hour, day_of_week, month, year, rain_impact=0.0):
    """ML modeli yüklenemezse basit istatistiksel fallback."""
    p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    hist = db.execute(f"""
        SELECT AVG(total_vehicles) AS v, AVG(city_avg_speed) AS s
        FROM '{p}'
        WHERE hour = {hour} AND day_of_week = {day_of_week} AND month = {month}
    """).df()
    db.close()
    if hist.empty or hist.iloc[0]["v"] is None:
        return {"vehicles": 0, "speed": 0, "ci_vehicles_low": 0, "ci_vehicles_high": 0, "ci_speed_low": 0, "ci_speed_high": 0}
    v, s = hist.iloc[0]["v"], hist.iloc[0]["s"]
    v *= (1 + rain_impact)
    s *= (1 - rain_impact * 0.8)
    return {"vehicles": v, "speed": s, "ci_vehicles_low": v*0.85, "ci_vehicles_high": v*1.15, "ci_speed_low": s*0.9, "ci_speed_high": s*1.1}


# ─────────────────────────────────────────────────────────────────────────
#  CO2 EMİSYON HESAPLAMA
# ─────────────────────────────────────────────────────────────────────────
def calculate_co2(vehicle_count, avg_speed):
    base_emission_gkm = 120
    if avg_speed < 10: speed_factor = 2.5
    elif avg_speed < 20: speed_factor = 1.8
    elif avg_speed < 30: speed_factor = 1.4
    elif avg_speed < 50: speed_factor = 1.1
    else: speed_factor = 1.0
    avg_distance_km = 5
    total_co2_g = vehicle_count * avg_distance_km * base_emission_gkm * speed_factor
    return {
        "total_kg": round(total_co2_g / 1000, 1),
        "total_ton": round(total_co2_g / 1_000_000, 2),
        "per_vehicle_g": round(avg_distance_km * base_emission_gkm * speed_factor, 0),
        "speed_factor": speed_factor,
    }


# ─────────────────────────────────────────────────────────────────────────
#  ROTA OPTİMİZASYONU (SİMÜLE)
# ─────────────────────────────────────────────────────────────────────────
ISTANBUL_DISTRICTS = {
    "Kadıköy":(40.9927,29.0290),"Beşiktaş":(41.0430,29.0050),"Fatih":(41.0186,28.9497),
    "Üsküdar":(41.0270,29.0150),"Bakırköy":(40.9819,28.8772),"Şişli":(41.0602,29.0000),
    "Maltepe":(40.9340,29.1300),"Kartal":(40.9070,29.1870),"Ataşehir":(40.9833,29.1167),
    "Beyoğlu":(41.0370,28.9770),"Sarıyer":(41.1670,29.0500),"Pendik":(40.8780,29.2330),
    "Sultanbeyli":(40.9680,29.2650),"Beykoz":(41.1310,29.0970),"Eyüpsultan":(41.0810,28.9340),
    "Bağcılar":(41.0344,28.8575),"Bahçelievler":(41.0010,28.8620),
    "Zeytinburnu":(41.0040,28.9000),"Güngören":(41.0090,28.8790),"Esenler":(41.0440,28.8720),
}


def simulate_route(start_district, end_district, avg_speed, hour):
    """İstanbul gerçeklerine kalibre edilmiş rota süresi simülasyonu.
    Tahmin edilen segment hızı, rota tipinin gerçekçi 'kapıdan kapıya' hız
    bandına ölçeklenir; ışık/kavşak gecikmesi ve zirve saat cezası eklenir."""
    start_pos = ISTANBUL_DISTRICTS[start_district]
    end_pos = ISTANBUL_DISTRICTS[end_district]
    dlat = abs(start_pos[0] - end_pos[0])
    dlon = abs(start_pos[1] - end_pos[1])
    crow_km = ((dlat * 111)**2 + (dlon * 85)**2)**0.5

    rush = 7 <= hour <= 9 or 17 <= hour <= 19

    def _effective_speed(lo, hi):
        # Tahmin hızını (≈0-70 km/s) rota tipinin hız bandına ölçekle
        n = max(0.0, min(1.0, avg_speed / 70.0))
        return lo + (hi - lo) * n

    def _build(name, dist_km, eff_speed, light_min_per_km, fixed_min, rush_factor):
        t = (dist_km / max(eff_speed, 5)) * 60 + dist_km * light_min_per_km + fixed_min
        if rush:
            t *= rush_factor
        door_speed = dist_km / max(t / 60, 0.01)  # kapıdan kapıya gerçek hız
        level = "🔴 Yoğun" if door_speed < 18 else "🟠 Orta" if door_speed < 32 else "🟢 Akıcı"
        return {"name": name, "distance_km": round(dist_km, 1),
                "duration_min": round(t), "avg_speed": round(door_speed),
                "traffic_level": level,
                "co2_g": round(dist_km * 120 * (1.4 if door_speed < 30 else 1.0))}

    routes = [
        # Ana arter: şehir içi, çok ışıklı/kavşaklı → km başına 0.7 dk gecikme
        _build("🛣️ Ana Arter (E-5/D-100)", crow_km * 1.3,
               _effective_speed(12, 38), 0.7, 0, 1.30),
        # Sahil: daha uzun, tek şerit darboğazları
        _build("🌊 Sahil / Alternatif Rota", crow_km * 1.5,
               _effective_speed(10, 32), 0.7, 0, 1.15),
        # Otoyol: hızlı ama bağlantı yolları + sabit giriş/çıkış payı (6 dk)
        _build("🚗 Otoyol (TEM/O-Yolları)", crow_km * 1.6,
               _effective_speed(22, 62), 0.15, 6, 1.15),
    ]

    best = min(range(len(routes)), key=lambda i: routes[i]["duration_min"])
    routes[best]["recommended"] = True
    return routes


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_osm_routes(lat1, lon1, lat2, lon2, via=None):
    """OpenStreetMap yol ağı (OSRM) üzerinden gerçek sürüş rotaları.
    via=(lat, lon) verilirse rota o noktadan geçirilir (alternatif üretmek için).
    Dönen liste: [{"distance_km", "summary", "geometry"}] — distance_km saf OSM verisidir."""
    try:
        coords = f"{lon1:.6f},{lat1:.6f};"
        if via is not None:
            coords += f"{via[1]:.6f},{via[0]:.6f};"
        coords += f"{lon2:.6f},{lat2:.6f}"
        url = f"https://router.project-osrm.org/route/v1/driving/{coords}"
        resp = requests.get(url, params={
            "alternatives": "false" if via else "true",
            "overview": "full", "geometries": "geojson", "steps": "true",
        }, headers={"User-Agent": "IstanbulTrafficPlatform/1.0"}, timeout=12)
        data = resp.json()
        if data.get("code") != "Ok":
            return []
        out = []
        for rt in data.get("routes", [])[:3]:
            legs = rt.get("legs", [])
            # Tüm bacakların yol adlarını birleştir (via'lı rotalarda 2 bacak olur)
            parts = []
            for lg in legs:
                s = (lg.get("summary") or "").strip()
                if s:
                    parts.extend(p.strip() for p in s.split(","))
            summary = ", ".join(dict.fromkeys(p for p in parts if p))
            out.append({
                "distance_km": rt.get("distance", 0) / 1000.0,
                "summary": summary,
                "geometry": rt.get("geometry", {}).get("coordinates", []),
            })
        return out
    except Exception:
        return []


# Yaka tespiti + alternatif üretmek için stratejik geçiş noktaları
ASIAN_DISTRICTS = {"Kadıköy", "Üsküdar", "Maltepe", "Kartal", "Ataşehir",
                   "Pendik", "Sultanbeyli", "Beykoz"}
VIAS_CROSSING = [("15 Temmuz Köprüsü", 41.0405, 29.0350),
                 ("FSM Köprüsü", 41.0907, 29.0712),
                 ("Avrasya Tüneli", 41.0009, 28.9912)]
VIAS_EUROPE = [("TEM (O-3)", 41.0750, 28.9480),
               ("Sahil (Kennedy Cd.)", 40.9870, 28.9180)]
VIAS_ASIA = [("D-100", 40.9760, 29.0850),
             ("TEM (O-4)", 41.0190, 29.1190)]


def gather_osm_alternatives(start_district, end_district, lat1, lon1, lat2, lon2):
    """Direkt OSRM rotası + stratejik via noktalarından geçen ek alternatifler.
    Benzer mesafeli (≈%2) kopyalar elenir, aşırı dolambaçlılar atılır; en az 2-3 rota hedeflenir."""
    candidates = []
    for r in fetch_osm_routes(lat1, lon1, lat2, lon2):
        r["via_label"] = None
        candidates.append(r)

    s_asian = start_district in ASIAN_DISTRICTS
    e_asian = end_district in ASIAN_DISTRICTS
    if s_asian != e_asian:
        vias = VIAS_CROSSING
    elif s_asian:
        vias = VIAS_ASIA
    else:
        vias = VIAS_EUROPE

    for label, vlat, vlon in vias:
        for r in fetch_osm_routes(lat1, lon1, lat2, lon2, via=(vlat, vlon)):
            r["via_label"] = label
            candidates.append(r)

    # Mesafeye göre sırala, neredeyse aynı olanları ele
    uniq = []
    for r in sorted(candidates, key=lambda x: x["distance_km"]):
        if all(abs(r["distance_km"] - u["distance_km"]) / max(u["distance_km"], 0.1) > 0.02
               for u in uniq):
            uniq.append(r)
    if uniq:
        d_min = uniq[0]["distance_km"]
        uniq = [r for r in uniq if r["distance_km"] <= d_min * 1.8][:4]
    return uniq


def blend_osm_route(idx, osm_route, pred_speed, pred_vehicles, veh_ref):
    """KM: doğrudan OSM verisi. DAKİKA: OSM km'si, sistemin o tarih/saat için
    tahmin ettiği araç yoğunluğu ve hızı ile hesaplanır. Hız ve CO₂ senkrondur."""
    dist_km = osm_route["distance_km"]

    # O tarihteki tahmini yoğunluğun, o gün/ay profil ortalamasına oranı
    ratio = pred_vehicles / max(veh_ref, 1e-6)
    cong = min(2.2, max(0.75, 0.55 + 0.55 * ratio))  # tıkanıklık çarpanı

    # Sensör hızı → kapıdan kapıya etkin hız (0.72 şehir düzeltmesi / yoğunluk)
    v_eff = (pred_speed * 0.72) / cong
    v_eff = min(max(v_eff, 8.0), 72.0)

    t = (dist_km / v_eff) * 60 + 4  # 4 dk başlangıç/varış payı
    door_speed = dist_km / (t / 60)
    level = "🔴 Yoğun" if door_speed < 18 else "🟠 Orta" if door_speed < 32 else "🟢 Akıcı"

    via_label = osm_route.get("via_label")
    summary = osm_route.get("summary", "")
    if via_label:
        name = f"🔀 {via_label} üzeri"
    elif summary:
        name = f"🛣️ {summary}"
    else:
        name = "🛣️ Ana Güzergah" if idx == 0 else f"🔀 Alternatif {idx + 1}"

    return {"name": name, "distance_km": round(dist_km, 1), "duration_min": round(t),
            "avg_speed": round(door_speed), "traffic_level": level,
            "co2_g": round(dist_km * 120 * (1.4 if door_speed < 30 else 1.0)),
            "geometry": osm_route["geometry"]}


# ─────────────────────────────────────────────────────────────────────────
#  İLÇE İSİMLERİ
# ─────────────────────────────────────────────────────────────────────────
def get_district_name(lat, lon):
    min_dist = float("inf")
    nearest = "Bilinmiyor"
    for name, (dlat, dlon) in ISTANBUL_DISTRICTS.items():
        dist = ((lat - dlat)**2 + (lon - dlon)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest = name
    return nearest


# ─────────────────────────────────────────────────────────────────────────
#  HEADER + DİL SEÇİCİ / LANGUAGE SWITCHER
# ─────────────────────────────────────────────────────────────────────────
hdr_main, hdr_toggle = st.columns([5, 1])

# Modern UI kalıcı standart olarak ayarlandı
st.session_state["ui_modern"] = True
ui_modern = True

with hdr_toggle:
    st.markdown("""
<div style="display:flex;align-items:center;justify-content:flex-end;height:100%;padding-top:18px;gap:6px;">
</div>""", unsafe_allow_html=True)
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        _tr_active = "background:linear-gradient(135deg,#e30a17,#e30a17);border:2px solid #fff;" if st.session_state["lang"]=="tr" else "background:rgba(17,20,42,0.7);border:1px solid rgba(148,163,184,0.25);"
        if st.button("🇹🇷", key="lang_tr", help="Türkçe", use_container_width=True):
            st.session_state["lang"] = "tr"
            st.rerun()
    with lang_col2:
        _en_active = "background:linear-gradient(135deg,#012169,#c8102e);border:2px solid #fff;" if st.session_state["lang"]=="en" else "background:rgba(17,20,42,0.7);border:1px solid rgba(148,163,184,0.25);"
        if st.button("🇬🇧", key="lang_en", help="English", use_container_width=True):
            st.session_state["lang"] = "en"
            st.rerun()

_pill = ("display:inline-flex;align-items:center;gap:6px;padding:7px 15px;border-radius:999px;"
         "background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.30);"
         "font-size:12.5px;font-weight:600;color:#c7d2fe;white-space:nowrap;")
with hdr_main:
    st.markdown(f"""
<div style="border-radius:18px;overflow:hidden;margin-bottom:8px;
            border:1px solid rgba(148,163,184,0.16);
            background:linear-gradient(160deg, rgba(23,26,48,0.78), rgba(9,11,24,0.88));
            backdrop-filter:blur(16px);box-shadow:0 14px 40px rgba(0,0,0,0.45);">
  <div class="mx-hero-bar"></div>
  <div style="display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;padding:22px 28px;">
    <div>
      <div style="font-size:11px;font-weight:700;letter-spacing:2.5px;color:#22d3ee;text-transform:uppercase;margin-bottom:7px;">
        {T("hero_badge")}
      </div>
      <div style="font-family:'Space Grotesk','Inter',sans-serif;font-size:2.0em;font-weight:700;line-height:1.15;color:#f1f5f9;">
        {T("hero_title1")}
        <span style="background:linear-gradient(90deg,#818cf8,#22d3ee);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{T("hero_title2")}</span>
      </div>
      <div style="color:#7c86a8;margin-top:7px;font-size:0.97em;">
        {T("hero_subtitle")}
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end;max-width:340px;">
      <span style="{_pill}">📊 {T("pill_records")}</span>
      <span style="{_pill}">📅 {T("pill_period")}</span>
      <span style="{_pill}">🤖 {T("pill_models")}</span>
      <span style="{_pill}">🌦️ {T("pill_weather")}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
month_names = T("months")

with st.sidebar:
    st.markdown(f"""
<div style='display:flex;align-items:center;gap:12px;padding:4px 2px 16px 2px;'>
  <div style='width:42px;height:42px;border-radius:13px;flex-shrink:0;
              background:linear-gradient(135deg,#6366f1,#8b5cf6);
              display:flex;align-items:center;justify-content:center;font-size:21px;
              box-shadow:0 6px 18px rgba(99,102,241,0.45);'>🚦</div>
  <div>
    <div style="font-family:'Space Grotesk','Inter',sans-serif;font-weight:700;font-size:16px;color:#f1f5f9;letter-spacing:.3px;">{T("sidebar_brand")}</div>
    <div style='font-size:11px;color:#64748b;'>{T("sidebar_subtitle")}</div>
  </div>
</div>""", unsafe_allow_html=True)
    st.markdown(f"### {T('sidebar_filters')}")
    monthly = load_monthly()
    years = sorted(monthly["year"].unique().tolist())
    selected_year = st.selectbox(T("sidebar_year"), years, index=len(years)-1)
    months_available = sorted(monthly[monthly["year"]==selected_year]["month"].unique().tolist())
    month_options = [month_names[m] for m in months_available]
    selected_month_name = st.selectbox(T("sidebar_month"), month_options, index=len(month_options)-1)
    selected_month = months_available[month_options.index(selected_month_name)]

    # Gün seçimi (opsiyonel)
    import calendar as _cal
    days_in_month = _cal.monthrange(selected_year, selected_month)[1]
    _all_month_lbl = T("all_month")
    day_options = [_all_month_lbl] + [str(d) for d in range(1, days_in_month + 1)]
    selected_day_str = st.selectbox(T("sidebar_day"), day_options, index=0)
    selected_day = int(selected_day_str) if selected_day_str != _all_month_lbl else None

    selected_hour = st.slider(T("sidebar_hour"), 0, 23, 8)

    st.markdown("---")
    st.markdown(f"### {T('sidebar_data_info')}")
    total_rows = monthly["record_count"].sum()
    total_vehicles_sum = monthly["total_vehicles"].sum()

    def _fmt_short(v):
        if v >= 1e9: return f"{v/1e9:.1f} Mr"
        if v >= 1e6: return f"{v/1e6:.0f} M"
        return f"{v:,.0f}"

    def _mini_card(icon, label, value):
        return (f"<div style='background:rgba(102,126,234,0.08);border:1px solid rgba(102,126,234,0.22);"
                f"border-radius:12px;padding:10px 12px;'>"
                f"<div style='font-size:11px;color:#8b93b8;'>{icon} {label}</div>"
                f"<div style='font-size:15px;font-weight:700;color:#e2e8f0;margin-top:2px;'>{value}</div></div>")

    st.markdown(
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>"
        + _mini_card("🗄️", T("total_records"), _fmt_short(total_rows))
        + _mini_card("🚗", T("vehicle_passage"), _fmt_short(total_vehicles_sum))
        + _mini_card("📅", T("period"), f"{years[0]}–{years[-1]}")
        + _mini_card("🗂️", T("monthly_file"), f"{len(monthly)} {T('months_suffix')}")
        + "</div>",
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
#  FİLTRE DEĞİŞİM KONTROLÜ → HARİTA ZOOM SIFIRLAMA
# ─────────────────────────────────────────────────────────────────────────
if (st.session_state.prev_year != selected_year or
    st.session_state.prev_month != selected_month or
    st.session_state.prev_hour != selected_hour or
    st.session_state.prev_day != selected_day):
    st.session_state.fly_lat = 41.015
    st.session_state.fly_lon = 29.01
    st.session_state.fly_zoom = 10
    st.session_state.fly_target_lat = None
    st.session_state.fly_target_lon = None
    st.session_state.prev_year = selected_year
    st.session_state.prev_month = selected_month
    st.session_state.prev_hour = selected_hour
    st.session_state.prev_day = selected_day


# ─────────────────────────────────────────────────────────────────────────
#  TOP METRICS
# ─────────────────────────────────────────────────────────────────────────
m_row = monthly[(monthly["year"]==selected_year) & (monthly["month"]==selected_month)]
if not m_row.empty:
    m = m_row.iloc[0]
    if ui_modern:
        def _kpi_card(icon, label, value, accent):
            return f"""<div style='flex:1;min-width:165px;position:relative;overflow:hidden;
background:linear-gradient(160deg, rgba(23,26,48,0.72), rgba(10,12,26,0.85));
border:1px solid rgba(148,163,184,0.14);border-radius:18px;padding:16px 18px;
box-shadow:inset 0 1px 0 rgba(255,255,255,0.05), 0 10px 30px rgba(0,0,0,0.35);'>
<div style='position:absolute;left:0;top:0;bottom:0;width:3px;background:{accent};'></div>
<div style='display:flex;align-items:center;gap:11px;'>
<div style='width:40px;height:40px;border-radius:12px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:19px;background:{accent}22;border:1px solid {accent}44;'>{icon}</div>
<div>
<div style='font-size:10.5px;font-weight:700;letter-spacing:1.2px;color:#7c86a8;text-transform:uppercase;'>{label}</div>
<div style="font-family:'Space Grotesk','Inter',sans-serif;font-size:21px;font-weight:700;color:#f1f5f9;margin-top:1px;">{value}</div>
</div></div></div>"""

        _kpis = [
            ("🚗", T("total_vehicles"), f"{m['total_vehicles']:,.0f}", "#6366f1"),
            ("⚡", T("avg_speed"), f"{m['avg_speed']:.0f} {T('km_h')}", "#22d3ee"),
            ("📍", T("active_locations"), f"{m['active_locations']:,.0f}", "#8b5cf6"),
            ("📊", T("record_count"), f"{m['record_count']:,.0f}", "#34d399"),
            ("📅", T("active_days"), f"{m['active_days']:.0f}", "#fbbf24"),
        ]
        st.markdown("<div style='display:flex;gap:13px;flex-wrap:wrap;'>"
                    + "".join(_kpi_card(*k) for k in _kpis) + "</div>",
                    unsafe_allow_html=True)
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric(f"🚗 {T('total_vehicles')}", f"{m['total_vehicles']:,.0f}")
        with c2: st.metric(f"⚡ {T('avg_speed')}", f"{m['avg_speed']:.0f} {T('km_h')}")
        with c3: st.metric(f"📍 {T('active_locations')}", f"{m['active_locations']:,.0f}")
        with c4: st.metric(f"📊 {T('record_count')}", f"{m['record_count']:,.0f}")
        with c5: st.metric(f"📅 {T('active_days')}", f"{m['active_days']:.0f}")

st.markdown("")


# ─────────────────────────────────────────────────────────────────────────
#  AYLIK RAPOR ÖZETİ
# ─────────────────────────────────────────────────────────────────────────
st.markdown(f"### 📋 {month_names[selected_month]} {selected_year} — {T('monthly_report')}")

rpt_col1, rpt_col2, rpt_col3 = st.columns(3)

def _ranked_bar_row(label_html, value_html, pct, bar_grad):
    """Modern UI: rapor kartlarında progress bar'lı sıralama satırı."""
    return (f"<div style='margin:9px 0;'>"
            f"<div style='display:flex;justify-content:space-between;gap:8px;font-size:13px;'>"
            f"<span>{label_html}</span><span style='color:#a5b4fc;white-space:nowrap;'>{value_html}</span></div>"
            f"<div style='height:6px;border-radius:3px;background:rgba(102,126,234,0.15);margin-top:5px;'>"
            f"<div style='width:{pct:.0f}%;height:100%;border-radius:3px;background:{bar_grad};'></div></div></div>")

peak_hours = load_monthly_peak_hour(selected_year, selected_month)
if not peak_hours.empty:
    with rpt_col1:
        top3 = peak_hours.head(3)
        h_html = ""
        if ui_modern:
            max_v = max(1, top3["avg_vehicles"].max())
            for i, row in top3.iterrows():
                h = int(row["hour"]); v = row["avg_vehicles"]; s = row["avg_speed"]
                badge = "🥇" if i==0 else "🥈" if i==1 else "🥉"
                h_html += _ranked_bar_row(
                    f"{badge} <b>Saat {h:02d}:00</b>",
                    f"{v:,.0f} araç • {s:.0f} km/s",
                    v / max_v * 100,
                    "linear-gradient(90deg,#667eea,#a78bfa)")
        else:
            for i, row in top3.iterrows():
                h = int(row["hour"]); v = row["avg_vehicles"]; s = row["avg_speed"]
                badge = "🥇" if i==0 else "🥈" if i==1 else "🥉"
                h_html += f"{badge} <b>Saat {h:02d}:00</b> — Ort. {v:,.0f} araç, {s:.0f} km/s<br>"
        st.markdown(f'<div class="report-box"><h4 style="margin:0 0 12px 0;">{T("peak_hours")}</h4>{h_html}</div>', unsafe_allow_html=True)

top_areas = load_monthly_top_areas(selected_year, selected_month)
if not top_areas.empty:
    with rpt_col2:
        a_html = ""
        if ui_modern:
            max_v = max(1, top_areas["total_vehicles"].max())
            for i, row in top_areas.iterrows():
                district = get_district_name(row["lat"], row["lon"])
                v = row["total_vehicles"]; s = row["avg_speed"]
                badge = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}"
                spd_icon = "🔴" if s<25 else "🟠" if s<35 else "🟡" if s<50 else "🟢"
                a_html += _ranked_bar_row(
                    f"{badge} <b>{district}</b>",
                    f"{v:,.0f} araç {spd_icon} {s:.0f} km/s",
                    v / max_v * 100,
                    "linear-gradient(90deg,#f97316,#fbbf24)")
        else:
            for i, row in top_areas.iterrows():
                district = get_district_name(row["lat"], row["lon"])
                v = row["total_vehicles"]; s = row["avg_speed"]
                badge = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}"
                spd_icon = "🔴" if s<25 else "🟠" if s<35 else "🟡" if s<50 else "🟢"
                a_html += f"{badge} <b>{district}</b> — {v:,.0f} araç {spd_icon} {s:.0f} km/s<br>"
        st.markdown(f'<div class="report-box"><h4 style="margin:0 0 12px 0;">{T("busiest_areas")}</h4>{a_html}</div>', unsafe_allow_html=True)

if not m_row.empty:
    with rpt_col3:
        m_co2 = calculate_co2(m["total_vehicles"], m["avg_speed"])
        st.markdown(f"""
        <div class="co2-box">
            <h4 style="margin:0 0 12px 0;">{T("co2_emission")}</h4>
            <span style="font-size:2em; font-weight:800;">{m_co2['total_ton']:,.1f}</span>
            <span style="font-size:1.1em; color:#6dbe82;"> {T("ton_co2")}</span><br><br>
            📊 {T("per_vehicle")}: <b>{m_co2['per_vehicle_g']:,.0f} g</b><br>
            ⚡ {T("speed_factor")}: <b>×{m_co2['speed_factor']}</b>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")


# ─────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5"), T("tab6"),
])


# ═════════════════════════════════════════════════════════════════════════
#  TAB 1 — HARİTA (v3.2)
# ═════════════════════════════════════════════════════════════════════════
with tab1:
    if selected_day:
        _day_label = f"{selected_day} {month_names[selected_month]} {selected_year}"
    else:
        _day_label = f"{month_names[selected_month]} {selected_year}"
    st.markdown(f"### 📍 {T('map_title')} — {_day_label}, {T('sidebar_hour')} {selected_hour}:00")

    if selected_day:
        geo_data = load_geo_daily_filtered(selected_year, selected_month, selected_day, selected_hour)
    else:
        geo_data = load_geo_hourly_filtered(selected_year, selected_month, selected_hour)

    map_col1, map_col2 = st.columns([3, 1])

    with map_col2:
        map_mode = st.radio(
            T("map_type"),
            [
                "🌌 Noktalar",
                "🔵 Noktalar (2D)",
                "📊 3D Sütunlar",
                "🔥 Isı Haritası",
                "🗺️ İstanbul Haritası",
            ],
            index=0,
        )

        color_by = st.radio(T("color_by"), [T("by_vehicle"), T("by_speed")], index=0)

        if not geo_data.empty:
            st.markdown("---")
            st.markdown(T("hour_stats"))
            st.metric(T("total_vehicles"), f"{geo_data['total_vehicles'].sum():,.0f}")
            st.metric(T("avg_speed"), f"{geo_data['avg_speed'].mean():.0f} {T('km_h')}")
            st.metric(T("active_locations"), f"{len(geo_data):,}")

        if st.session_state.get("uploaded_data"):
            st.markdown("""
<div style='margin-top:10px;padding:8px 10px;background:rgba(15,23,42,0.6);
            border:1px solid rgba(99,102,241,0.3);border-radius:10px;font-size:11px;color:#94a3b8;'>
  ⚪ <b style='color:#cbd5e1;'>Beyaz halkalı noktalar:</b> yüklediğiniz ölçümler<br>
  <span style='color:#ef4444;'>●</span> İBB ort. üstü &nbsp;
  <span style='color:#3b82f6;'>●</span> altı &nbsp;
  <span style='color:#e2e8f0;'>●</span> uyumlu &nbsp;
  <span style='color:#a855f7;'>●</span> kıyas yok
</div>""", unsafe_allow_html=True)

    with map_col1:
        if not geo_data.empty:
            # İlçe isimleri (vektörel hesap — satır satır apply'dan çok daha hızlı)
            _dn = list(ISTANBUL_DISTRICTS.keys())
            _dc = np.array([ISTANBUL_DISTRICTS[n] for n in _dn])
            _d2 = ((geo_data["lat"].to_numpy()[:, None] - _dc[None, :, 0]) ** 2
                   + (geo_data["lon"].to_numpy()[:, None] - _dc[None, :, 1]) ** 2)
            geo_data["district"] = [_dn[i] for i in _d2.argmin(axis=1)]

            # Normalize
            if "Isı" not in map_mode:
                if color_by == T("by_vehicle"):
                    geo_data["weight"] = geo_data["total_vehicles"]
                    vmin, vmax = geo_data["total_vehicles"].min(), geo_data["total_vehicles"].max()
                    geo_data["normalized"] = (geo_data["total_vehicles"] - vmin) / max(1, vmax - vmin)
                else:
                    max_speed = geo_data["avg_speed"].max()
                    geo_data["weight"] = max_speed - geo_data["avg_speed"]
                    geo_data["normalized"] = geo_data["weight"] / max(1, geo_data["weight"].max())

                # Eşikler: sabit 1/3 yerine p33/p66 persentil — dağılım ne kadar çarpık
                # olursa olsun her renk bandına yaklaşık eşit nokta düşer.
                _norm_arr = geo_data["normalized"].to_numpy()
                _p33 = float(np.percentile(_norm_arr, 33))
                _p66 = float(np.percentile(_norm_arr, 66))
                _bins = np.digitize(_norm_arr, [_p33, _p66])
                _palette = np.array([[34, 197, 94, 180], [234, 179, 8, 200], [239, 68, 68, 220]])
                geo_data[["color_r", "color_g", "color_b", "color_a"]] = _palette[_bins]

            geo_data["avg_speed_int"] = geo_data["avg_speed"].round(0).astype(int)
            geo_data["total_vehicles_int"] = geo_data["total_vehicles"].round(0).astype(int)

            view = pdk.ViewState(
                latitude=st.session_state.fly_lat,
                longitude=st.session_state.fly_lon,
                zoom=st.session_state.fly_zoom,
                pitch=45 if ("3D" in map_mode or "İstanbul Haritası" in map_mode) else 0,
            )
            if map_mode == "🌌 Noktalar":
                map_style = "mapbox://styles/mapbox/dark-v11"
            elif "İstanbul Haritası" in map_mode:
                # CARTO Dark Matter: ilçe/semt isimleri altlığın kendisinde yazılı
                map_style = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
            else:
                map_style = "mapbox://styles/mapbox/streets-v12"

            layers = []

            if "Isı" in map_mode:
                # ── ISI HARİTASI (HeatmapLayer) ─────────────────
                # Weight: araç sayısı
                geo_data["heat_weight"] = geo_data["total_vehicles"].astype(float)

                heat_layer = pdk.Layer(
                    "HeatmapLayer",
                    data=geo_data,
                    get_position=["lon", "lat"],
                    get_weight="heat_weight",
                    aggregation="SUM",
                    radius_pixels=60,
                    intensity=1,
                    threshold=0.05,
                    color_range=[
                        [0, 128, 0, 160],      # yeşil
                        [85, 170, 0, 180],      # açık yeşil
                        [255, 255, 0, 200],     # sarı
                        [255, 165, 0, 220],     # turuncu
                        [255, 69, 0, 230],      # koyu turuncu
                        [255, 0, 0, 240],       # kırmızı
                    ],
                )
                layers.append(heat_layer)

            elif "3D" in map_mode:
                max_v = geo_data["total_vehicles"].max()
                geo_data["elevation"] = (geo_data["total_vehicles"] / max(1, max_v)) * 3000
                layer = pdk.Layer(
                    "ColumnLayer", data=geo_data,
                    get_position=["lon","lat"], get_elevation="elevation",
                    elevation_scale=1, radius=300,
                    get_fill_color=["color_r","color_g","color_b","color_a"],
                    pickable=True, auto_highlight=True,
                )
                layers.append(layer)
            elif "İstanbul Haritası" in map_mode:
                # ── İSTANBUL HARİTASI (CARTO Dark Matter + yoğunluk noktaları) ──
                # İlçe/semt isimleri CARTO altlığının kendisinde yazılıdır;
                # trafik yoğunluğu küçük yeşil/sarı/kırmızı noktalarla gösterilir.
                layer = pdk.Layer(
                    "ScatterplotLayer", data=geo_data,
                    get_position=["lon", "lat"],
                    get_radius=110,
                    get_fill_color=["color_r", "color_g", "color_b", 230],
                    pickable=True, auto_highlight=True,
                    highlight_color=[255, 255, 255, 120],
                    radius_min_pixels=2, radius_max_pixels=5,
                )
                layers.append(layer)
            else:
                # 2D Noktalar
                if map_mode == "🌌 Noktalar":
                    geo_data["radius"] = 150  # fixed small radius for aesthetic look
                else:
                    geo_data["radius"] = (geo_data["normalized"] * 600 + 200).astype(int)

                layer = pdk.Layer(
                    "ScatterplotLayer", data=geo_data,
                    get_position=["lon","lat"], get_radius="radius",
                    get_fill_color=["color_r","color_g","color_b","color_a"],
                    pickable=True, stroked=True,
                    get_line_color=[40,40,40,100], line_width_min_pixels=1,
                )
                layers.append(layer)

            # ── Yüklenen saha ölçümleri (Veri Karşılaştır sekmesi) — haritada işaretle ──
            if st.session_state.get("uploaded_data"):
                try:
                    _up_all = pd.concat([_i["data"] for _i in st.session_state.uploaded_data],
                                        ignore_index=True)
                except Exception:
                    _up_all = pd.DataFrame()
                if not _up_all.empty and {"lat", "lon", "vehicle_count"}.issubset(_up_all.columns):
                    _up_all = _up_all.dropna(subset=["lat", "lon"]).copy()
                    # İBB ortalamasıyla fark rengi (geohash+saat varsa)
                    if {"geohash", "hour", "date_time"}.issubset(_up_all.columns):
                        _up_all["gh5"] = _up_all["geohash"].astype(str).str[:5]
                        _udt1 = pd.to_datetime(_up_all["date_time"], errors="coerce")
                        _um1 = int(_udt1.dt.month.mode().iloc[0]) if _udt1.notna().any() else selected_month
                        _ud1 = int(_udt1.dt.weekday.mode().iloc[0]) if _udt1.notna().any() else 0
                        _ub1 = load_comparison_baseline(
                            tuple(sorted(_up_all["gh5"].dropna().unique())), _um1, _ud1)
                        if not _ub1.empty:
                            _up_all = _up_all.merge(_ub1, on=["gh5", "hour"], how="left")
                    # Sayımı saatlik orana çevir (İBB değeri saatliktir)
                    if "period_minutes" in _up_all.columns:
                        _pm1 = pd.to_numeric(_up_all["period_minutes"], errors="coerce").clip(0.5, 120).fillna(60.0)
                        _up_all["vehicle_rate"] = _up_all["vehicle_count"] * (60.0 / _pm1)
                    else:
                        _up_all["vehicle_rate"] = _up_all["vehicle_count"]
                    _aggs = {"vehicle_rate": ("vehicle_rate", "mean")}
                    if "hist_vehicles" in _up_all.columns:
                        _aggs["hist_vehicles"] = ("hist_vehicles", "mean")
                    _up_pts = _up_all.groupby(["lat", "lon"], as_index=False).agg(**_aggs)

                    def _up_color(row):
                        hv = row.get("hist_vehicles")
                        if hv is None or pd.isna(hv) or hv <= 0:
                            return [168, 85, 247, 235]       # mor: kıyas verisi yok
                        d = (row["vehicle_rate"] - hv) / hv * 100
                        if d > 10: return [239, 68, 68, 235]    # İBB ortalamasının üstü
                        if d < -10: return [59, 130, 246, 235]  # altı
                        return [226, 232, 240, 235]             # uyumlu

                    _ucols = _up_pts.apply(_up_color, axis=1)
                    _up_pts["ur"] = _ucols.apply(lambda c: c[0])
                    _up_pts["ug"] = _ucols.apply(lambda c: c[1])
                    _up_pts["ubl"] = _ucols.apply(lambda c: c[2])
                    _up_pts["ua"] = _ucols.apply(lambda c: c[3])
                    layers.append(pdk.Layer(
                        "ScatterplotLayer", data=_up_pts,
                        get_position=["lon", "lat"],
                        get_fill_color=["ur", "ug", "ubl", "ua"],
                        get_radius=520,
                        radius_min_pixels=7, radius_max_pixels=16,
                        stroked=True, get_line_color=[255, 255, 255, 255],
                        line_width_min_pixels=2,
                        pickable=False,
                    ))

            # ── "Noktaya Git" hedef işaretçisi — TÜM harita modlarında görünür ──
            fly_t_lat = st.session_state.fly_target_lat
            fly_t_lon = st.session_state.fly_target_lon
            if fly_t_lat is not None and fly_t_lon is not None:
                _target_point = pd.DataFrame([{"lat": fly_t_lat, "lon": fly_t_lon}])
                # Çekirdek: küçük turkuaz nokta, beyaz kontur
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=_target_point,
                    get_position=["lon", "lat"],
                    get_radius=300,
                    get_fill_color=[34, 211, 238, 255],
                    stroked=True, get_line_color=[255, 255, 255, 230],
                    line_width_min_pixels=2,
                    radius_min_pixels=5, radius_max_pixels=12,
                    pickable=False,
                ))
                # İç halka: ince turkuaz odak çemberi
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=_target_point,
                    get_position=["lon", "lat"],
                    get_radius=650,
                    filled=False, stroked=True,
                    get_line_color=[34, 211, 238, 220],
                    line_width_min_pixels=2,
                    radius_min_pixels=10, radius_max_pixels=26,
                    pickable=False,
                ))
                # Dış halka: soluk geniş pulse efekti
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=_target_point,
                    get_position=["lon", "lat"],
                    get_radius=1100,
                    get_fill_color=[34, 211, 238, 35],
                    radius_min_pixels=16, radius_max_pixels=40,
                    pickable=False,
                ))

            # İlçe merkezleri TextLayer (İstanbul Haritası modunda isimler CARTO altlığından gelir)
            if "İstanbul Haritası" not in map_mode:
                district_centers = pd.DataFrame([
                    {"name": name, "lat": coords[0], "lon": coords[1]}
                    for name, coords in ISTANBUL_DISTRICTS.items()
                ])
                text_layer = pdk.Layer(
                    "TextLayer", data=district_centers,
                    get_position=["lon","lat"], get_text="name",
                    get_size=14, get_color=[30,30,80,230],
                    get_angle=0, get_text_anchor="'middle'",
                    get_alignment_baseline="'center'",
                    font_family="'Inter', 'Arial', sans-serif",
                    font_weight=700, billboard=False,
                    size_min_pixels=10, size_max_pixels=18,
                )
                layers.append(text_layer)

            tooltip_config = None
            if "Isı" not in map_mode:
                tooltip_config = {
                    "html": f"<b>📍 {T('district_col')}:</b> {{district}}<br/><b>🚗 {T('total_vehicles')}:</b> {{total_vehicles_int}}<br/><b>⚡ {T('avg_speed')}:</b> {{avg_speed_int}} {T('km_h')}<br/><b>📌 Geohash:</b> {{geohash}}",
                    "style": {
                        "backgroundColor":"#1a1a2e","color":"#ffffff",
                        "border":"1px solid #667eea","borderRadius":"8px",
                        "padding":"10px 14px","fontSize":"13px","fontFamily":"Inter, sans-serif",
                    },
                }

            deck = pdk.Deck(
                layers=layers, initial_view_state=view,
                map_style=map_style, tooltip=tooltip_config,
            )
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.warning(T("no_data_warning"))

    # ── 4 Sekmeli Nokta Analizi ──────────────────────────────────
    st.markdown(f"### 📊 {T('region_analysis')}")
    ntab1, ntab2, ntab3, ntab4 = st.tabs([
        T("slowest_5"), T("fastest_5"), T("most_congested_5"), T("least_congested_5"),
    ])

    def _render_points(df, tab_key):
        if df.empty:
            st.info(T("no_data"))
            return
        for idx, row in df.iterrows():
            cols = st.columns([1, 2, 2, 2, 2, 2])
            rank = idx + 1
            district = get_district_name(row["lat"], row["lon"])
            cols[0].markdown(f"**#{rank}**")
            cols[1].markdown(f"📍 **{district}** (`{row['geohash']}`)")
            cols[2].markdown(f"🗺️ {row['lat']:.4f}, {row['lon']:.4f}")
            cols[3].markdown(f"🚗 {row['total_vehicles']:,.0f}")
            spd = row['avg_speed']
            spd_color = "🔴" if spd < 25 else "🟠" if spd < 35 else "🟡" if spd < 50 else "🟢"
            cols[4].markdown(f"{spd_color} {spd:.0f} {T('km_h')}")
            if cols[5].button(T("go_btn"), key=f"fly_{tab_key}_{idx}"):
                st.session_state.fly_lat = float(row["lat"])
                st.session_state.fly_lon = float(row["lon"])
                st.session_state.fly_zoom = 14
                st.session_state.fly_target_lat = float(row["lat"])
                st.session_state.fly_target_lon = float(row["lon"])
                st.session_state.prev_year = selected_year
                st.session_state.prev_month = selected_month
                st.session_state.prev_hour = selected_hour
                st.rerun()

    with ntab1:
        _render_points(load_top_slowest(selected_year, selected_month), "slow")
    with ntab2:
        _render_points(load_top_fastest(selected_year, selected_month), "fast")
    with ntab3:
        _render_points(load_top_congested(selected_year, selected_month), "cong")
    with ntab4:
        _render_points(load_least_congested(selected_year, selected_month), "least")

    if st.button(T("map_reset")):
        st.session_state.fly_lat = 41.015
        st.session_state.fly_lon = 29.01
        st.session_state.fly_zoom = 10
        st.session_state.fly_target_lat = None
        st.session_state.fly_target_lon = None
        st.session_state.prev_year = selected_year
        st.session_state.prev_month = selected_month
        st.session_state.prev_hour = selected_hour
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════
#  TAB 2 — ZAMAN ANALİZİ
# ═════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### {T('time_analysis')}")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"#### 🕐 {T('hourly_profile')} ({selected_year})")
        hourly_profile = load_hourly_profile(selected_year)
        fig_h = make_subplots(specs=[[{"secondary_y":True}]])
        fig_h.add_trace(go.Bar(x=hourly_profile["hour"],y=hourly_profile["avg_vehicles"],
                               name=T("vehicle_sec"),marker_color="#667eea",opacity=0.7),secondary_y=False)
        fig_h.add_trace(go.Scatter(x=hourly_profile["hour"],y=hourly_profile["avg_speed"],
                                   name=T("avg_speed_short"),line=dict(color="#f97316",width=3),
                                   mode="lines+markers"),secondary_y=True)
        fig_h.update_layout(template="plotly_dark",height=400,margin=dict(l=20,r=20,t=30,b=20),
                            legend=dict(orientation="h",y=1.12),xaxis_title=T("hour_axis"),hovermode="x unified")
        fig_h.update_yaxes(title_text=T("vehicle_count_label"),secondary_y=False)
        fig_h.update_yaxes(title_text=T("avg_speed_short"),secondary_y=True)
        st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        st.markdown(f"#### 📅 {T('weekly_profile')} ({selected_year})")
        dow_profile = load_dow_profile(selected_year)
        fig_d = make_subplots(specs=[[{"secondary_y":True}]])
        fig_d.add_trace(go.Bar(x=dow_profile["day_name"],y=dow_profile["avg_vehicles"],
                               name=T("vehicle_sec"),marker_color="#764ba2",opacity=0.7),secondary_y=False)
        fig_d.add_trace(go.Scatter(x=dow_profile["day_name"],y=dow_profile["avg_speed"],
                                   name=T("avg_speed_short"),line=dict(color="#22c55e",width=3),
                                   mode="lines+markers"),secondary_y=True)
        fig_d.update_layout(template="plotly_dark",height=400,margin=dict(l=20,r=20,t=30,b=20),
                            legend=dict(orientation="h",y=1.12),hovermode="x unified")
        fig_d.update_yaxes(title_text=T("vehicle_count_label"),secondary_y=False)
        fig_d.update_yaxes(title_text=T("avg_speed_short"),secondary_y=True)
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown(f"#### {T('monthly_trend')}")
    monthly_data = load_monthly()
    fig_m = make_subplots(specs=[[{"secondary_y":True}]])
    fig_m.add_trace(go.Scatter(x=monthly_data["date"],y=monthly_data["total_vehicles"],
                               name=T("total_vehicles"),fill="tozeroy",
                               line=dict(color="#667eea",width=2),
                               fillcolor="rgba(102,126,234,0.15)"),secondary_y=False)
    fig_m.add_trace(go.Scatter(x=monthly_data["date"],y=monthly_data["avg_speed"],
                               name=T("avg_speed_short"),line=dict(color="#f97316",width=2)),secondary_y=True)
    fig_m.update_layout(template="plotly_dark",height=400,margin=dict(l=20,r=20,t=30,b=20),
                        legend=dict(orientation="h",y=1.08),hovermode="x unified")
    fig_m.update_yaxes(title_text=T("vehicle_count_label"),secondary_y=False)
    fig_m.update_yaxes(title_text=T("avg_speed_short"),secondary_y=True)
    st.plotly_chart(fig_m, use_container_width=True)

    # ── Hafta İçi vs Hafta Sonu Kıyaslama ────────────────────────
    st.markdown(f"#### 🔄 {T('weekday_vs_weekend')} ({selected_year})")
    hourly_data = load_hourly()
    yr_data = hourly_data[hourly_data["hour_ts"].dt.year == selected_year].copy()
    if not yr_data.empty:
        yr_data["is_weekend"] = yr_data["day_of_week"].isin([5, 6])
        yr_data["day_type"] = yr_data["is_weekend"].map({True: T("weekend"), False: T("weekday")})
        wk_profile = yr_data.groupby(["day_type", "hour"]).agg(
            avg_vehicles=("total_vehicles", "mean"),
            avg_speed=("city_avg_speed", "mean"),
        ).reset_index()

        wk_col1, wk_col2 = st.columns(2)
        with wk_col1:
            fig_wk_v = go.Figure()
            for dt, color in [(T("weekday"), "#667eea"), (T("weekend"), "#f97316")]:
                sub = wk_profile[wk_profile["day_type"] == dt]
                fig_wk_v.add_trace(go.Scatter(x=sub["hour"], y=sub["avg_vehicles"], name=dt,
                                              line=dict(color=color, width=3), mode="lines+markers"))
            fig_wk_v.update_layout(template="plotly_dark", height=350, title=T("vehicle_count_label"),
                                   xaxis_title=T("hour_axis"), yaxis_title=T("vehicle_sec"), margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_wk_v, use_container_width=True)

        with wk_col2:
            fig_wk_s = go.Figure()
            for dt, color in [(T("weekday"), "#22c55e"), (T("weekend"), "#ef4444")]:
                sub = wk_profile[wk_profile["day_type"] == dt]
                fig_wk_s.add_trace(go.Scatter(x=sub["hour"], y=sub["avg_speed"], name=dt,
                                              line=dict(color=color, width=3), mode="lines+markers"))
            fig_wk_s.update_layout(template="plotly_dark", height=350, title=T("avg_speed"),
                                   xaxis_title=T("hour_axis"), yaxis_title=T("avg_speed_short"), margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_wk_s, use_container_width=True)

    # ── Rush Hour (Zirve Saat) Analizi ───────────────────────────
    st.markdown(f"#### ⏰ {T('rush_analysis')} ({selected_year})")
    if not yr_data.empty:
        yr_data["rush_type"] = yr_data["hour"].apply(
            lambda h: T("morning_rush") if 7 <= h <= 9 else (T("evening_rush") if 17 <= h <= 19 else None))
        rush = yr_data[yr_data["rush_type"].notna()].copy()
        if not rush.empty:
            rush_monthly = rush.groupby([rush["hour_ts"].dt.month, "rush_type"]).agg(
                avg_vehicles=("total_vehicles", "mean")).reset_index()
            rush_monthly.columns = ["month", "rush_type", "avg_vehicles"]
            rush_monthly["month_name"] = rush_monthly["month"].map(month_names)
            fig_rush = px.bar(rush_monthly, x="month_name", y="avg_vehicles", color="rush_type",
                              barmode="group", template="plotly_dark",
                              color_discrete_map={T("morning_rush"): "#f97316", T("evening_rush"): "#667eea"},
                              labels={"avg_vehicles": T("vehicle_sec"), "month_name": T("month_axis"), "rush_type": T("rush_type_lbl")})
            fig_rush.update_layout(height=380, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig_rush, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════
#  TAB 3 — YIL KARŞILAŞTIRMA
# ═════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"### {T('year_comparison')}")
    comp_data = load_year_comparison()

    monthly_comp_all = comp_data.groupby(["year","month"]).agg(
        avg_speed=("avg_speed","mean"),avg_vehicles=("avg_vehicles","mean")).reset_index()

    all_years = sorted(monthly_comp_all["year"].unique())

    comp_mode = st.radio(
        T("comparison_type"),
        [T("yoy"), T("mom")],
        horizontal=True,
    )

    if comp_mode == T("yoy"):
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            y1 = st.selectbox(T("base_year"), all_years, index=max(0, len(all_years)-2) if len(all_years)>=2 else 0)
        with col_sel2:
            y2 = st.selectbox(T("target_year"), all_years, index=max(0, len(all_years)-1))

        monthly_comp = monthly_comp_all[monthly_comp_all["year"].isin([y1, y2])].copy()
        monthly_comp["year"] = monthly_comp["year"].astype(str)

        col_y1, col_y2 = st.columns(2)
        with col_y1:
            fig_yc = px.line(monthly_comp, x="month", y="avg_speed", color="year", template="plotly_dark",
                             labels={"avg_speed": T("avg_speed_short"),"month": T("month_axis"),"year": ""},
                             title=f"{y1} & {y2} {T('avg_speed_monthly')}", markers=True,
                             color_discrete_map={str(y1): "#667eea", str(y2): "#f97316"})
            fig_yc.update_layout(height=400, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_yc, use_container_width=True)

        with col_y2:
            fig_yv = px.line(monthly_comp, x="month", y="avg_vehicles", color="year", template="plotly_dark",
                             labels={"avg_vehicles": T("avg_vehicles_label"),"month": T("month_axis"),"year": ""},
                             title=f"{y1} & {y2} {T('avg_vehicles_monthly')}", markers=True,
                             color_discrete_map={str(y1): "#667eea", str(y2): "#f97316"})
            fig_yv.update_layout(height=400, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_yv, use_container_width=True)

        # Pandemi Etkisi
        st.markdown(f"#### {T('pandemic_effect')}")
        pandemic_comp = monthly_comp_all.copy()
        pandemic_comp["period"] = pandemic_comp["year"].apply(lambda y: T("pandemic") if y==2020 else T("normal_period"))
        agg_pan = pandemic_comp.groupby(["period","month"]).agg(avg_speed=("avg_speed","mean"),avg_vehicles=("avg_vehicles","mean")).reset_index()
        fig_pan = make_subplots(rows=1, cols=2, subplot_titles=[T("pandemic_speed"), T("pandemic_vehicles")])
        for period, color in [(T("pandemic"),"#ef4444"),(T("normal_period"),"#22c55e")]:
            sub = agg_pan[agg_pan["period"]==period]
            fig_pan.add_trace(go.Scatter(x=sub["month"], y=sub["avg_speed"], name=f"{period} ({T('avg_speed_short')})",
                                         line=dict(color=color, width=3), mode="lines+markers"), row=1, col=1)
            fig_pan.add_trace(go.Scatter(x=sub["month"], y=sub["avg_vehicles"], name=f"{period} ({T('vehicle_sec')})",
                                         line=dict(color=color, width=3, dash="dash"), mode="lines+markers"), row=1, col=2)
        fig_pan.update_layout(template="plotly_dark", height=400, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_pan, use_container_width=True)

        # YoY Büyüme Oranı
        st.markdown(f"#### 📈 {T('yoy_growth_rate')} ({y1} → {y2})")
        st.caption(f"{y1} → {y2}")
        prev_data = monthly_comp_all[monthly_comp_all["year"] == y1].set_index("month")
        curr_data = monthly_comp_all[monthly_comp_all["year"] == y2].set_index("month")
        common_months = sorted(set(prev_data.index) & set(curr_data.index))
        if common_months:
            yoy_pct = []
            for m in common_months:
                pv = prev_data.loc[m, "avg_vehicles"]
                cv = curr_data.loc[m, "avg_vehicles"]
                pct = ((cv - pv) / max(pv, 1)) * 100
                yoy_pct.append({"month": month_names[m], "pct": round(pct, 1)})
            yoy_df = pd.DataFrame(yoy_pct)
            colors_bar = ["#22c55e" if p >= 0 else "#ef4444" for p in yoy_df["pct"]]
            fig_yoy = go.Figure(go.Bar(
                x=yoy_df["month"], y=yoy_df["pct"],
                marker_color=colors_bar,
                text=[f"{p:+.1f}%" for p in yoy_df["pct"]], textposition="outside",
            ))
            fig_yoy.update_layout(template="plotly_dark", height=380,
                                  yaxis_title=T("change_pct"), xaxis_title=T("month_axis"),
                                  margin=dict(l=20,r=20,t=20,b=20))
            fig_yoy.add_hline(y=0, line_dash="dash", line_color="#555")
            st.plotly_chart(fig_yoy, use_container_width=True)

        # YoY Gelişen vs Kötüleşen Bölgeler
        st.markdown(f"#### 🏘️ {T('improving_areas')}")
        p_path = os.path.join(DATA_DIR, "summary_geo_hourly.parquet").replace("\\", "/")
        db = duckdb.connect()
        district_comp = db.execute(f"""
            SELECT geohash, AVG(lat) AS lat, AVG(lon) AS lon,
                   AVG(CASE WHEN year = {y1} THEN avg_speed END) AS prev_speed,
                   AVG(CASE WHEN year = {y2} THEN avg_speed END) AS curr_speed
            FROM '{p_path}'
            WHERE year IN ({y1}, {y2})
            GROUP BY geohash
            HAVING prev_speed IS NOT NULL AND curr_speed IS NOT NULL
        """).df()
        db.close()
        if not district_comp.empty:
            district_comp["speed_change"] = district_comp["curr_speed"] - district_comp["prev_speed"]
            district_comp["district"] = district_comp.apply(lambda r: get_district_name(r["lat"], r["lon"]), axis=1)
            by_dist = district_comp.groupby("district").agg(avg_change=("speed_change", "mean")).reset_index()
            by_dist = by_dist.sort_values("avg_change", ascending=False)
            
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**🟢 {T('speeding_districts')} ({y1}→{y2})**")
                top3 = by_dist.head(3)
                for _idx, r in top3.iterrows():
                    st.markdown(f"✅ **{r['district']}** → {T('avg_speed_short')} **+{r['avg_change']:.1f}** {T('km_h')} {T('increased')}")
            with dc2:
                st.markdown(f"**🔴 {T('slowing_districts')} ({y1}→{y2})**")
                bot3 = by_dist.tail(3)
                for _idx, r in bot3.iterrows():
                    st.markdown(f"⚠️ **{r['district']}** → {T('avg_speed_short')} **{r['avg_change']:.1f}** {T('km_h')} {T('decreased')}")

    else:
        mom_year = st.selectbox(T("year_to_inspect"), all_years, index=max(0, len(all_years)-1))
        
        yearly_data = monthly_comp_all[monthly_comp_all["year"] == mom_year].sort_values("month").copy()
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_speed_mom = px.line(yearly_data, x="month", y="avg_speed", template="plotly_dark",
                                    labels={"avg_speed": T("avg_speed_short"), "month": T("month_axis")},
                                    title=f"{mom_year} {T('speed_trend')}", markers=True,
                                    color_discrete_sequence=["#22c55e"])
            fig_speed_mom.update_layout(height=380, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_speed_mom, use_container_width=True)

        with col_m2:
            fig_veh_mom = px.line(yearly_data, x="month", y="avg_vehicles", template="plotly_dark",
                                   labels={"avg_vehicles": T("vehicle_sec"), "month": T("month_axis")},
                                   title=f"{mom_year} {T('density_trend')}", markers=True,
                                   color_discrete_sequence=["#eab308"])
            fig_veh_mom.update_layout(height=380, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_veh_mom, use_container_width=True)

        # MoM Büyüme Oranı
        st.markdown(f"#### 📈 {T('mom_growth_rate')} — {mom_year}")
        if len(yearly_data) >= 2:
            mom_pct = []
            for i in range(1, len(yearly_data)):
                prev_row = yearly_data.iloc[i-1]
                curr_row = yearly_data.iloc[i]
                pv = prev_row["avg_vehicles"]
                cv = curr_row["avg_vehicles"]
                pct = ((cv - pv) / max(pv, 1)) * 100
                mom_pct.append({"month": month_names[int(curr_row["month"])], "pct": round(pct, 1)})
            mom_df = pd.DataFrame(mom_pct)
            colors_bar_mom = ["#22c55e" if p >= 0 else "#ef4444" for p in mom_df["pct"]]
            fig_mom = go.Figure(go.Bar(
                x=mom_df["month"], y=mom_df["pct"],
                marker_color=colors_bar_mom,
                text=[f"{p:+.1f}%" for p in mom_df["pct"]], textposition="outside",
            ))
            fig_mom.update_layout(template="plotly_dark", height=380,
                                  yaxis_title=T("change_pct"), xaxis_title=T("month_axis"),
                                  margin=dict(l=20,r=20,t=20,b=20))
            fig_mom.add_hline(y=0, line_dash="dash", line_color="#555")
            st.plotly_chart(fig_mom, use_container_width=True)

        # Detaylı İki Ay Karşılaştırma
        st.markdown(f"#### {T('two_month_comp')}")
        avail_months = sorted(yearly_data["month"].unique())
        col_m_sel1, col_m_sel2 = st.columns(2)
        with col_m_sel1:
            m1 = st.selectbox(T("month_1"), avail_months, index=0, format_func=lambda x: month_names[x])
        with col_m_sel2:
            m2 = st.selectbox(T("month_2"), avail_months, index=min(1, len(avail_months)-1), format_func=lambda x: month_names[x])

        m1_data = yearly_data[yearly_data["month"] == m1]
        m2_data = yearly_data[yearly_data["month"] == m2]

        if not m1_data.empty and not m2_data.empty:
            m1_row = m1_data.iloc[0]
            m2_row = m2_data.iloc[0]

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"**📊 {T('vehicle_count_change')} ({month_names[m1]} → {month_names[m2]})**")
                v_diff = m2_row["avg_vehicles"] - m1_row["avg_vehicles"]
                v_diff_pct = (v_diff / max(m1_row["avg_vehicles"], 1)) * 100
                st.metric(
                    f"{month_names[m2]} {T('vehicle_sec')}",
                    f"{m2_row['avg_vehicles']:,.0f}",
                    delta=f"{v_diff:+.0f} ({v_diff_pct:+.1f}%)"
                )
            with m_col2:
                st.markdown(f"**⚡ {T('speed_change_lbl')} ({month_names[m1]} → {month_names[m2]})**")
                s_diff = m2_row["avg_speed"] - m1_row["avg_speed"]
                st.metric(
                    f"{month_names[m2]} {T('avg_speed_short')}",
                    f"{m2_row['avg_speed']:.1f} km/s",
                    delta=f"{s_diff:+.1f} km/s"
                )

            # MoM Gelişen vs Kötüleşen Bölgeler
            st.markdown(f"#### 🏘️ {T('improving_areas')} ({month_names[m1]} → {month_names[m2]})")
            p_path = os.path.join(DATA_DIR, "summary_geo_hourly.parquet").replace("\\", "/")
            db = duckdb.connect()
            geo_month_comp = db.execute(f"""
                SELECT geohash, AVG(lat) AS lat, AVG(lon) AS lon,
                       AVG(CASE WHEN month = {m1} THEN avg_speed END) AS m1_speed,
                       AVG(CASE WHEN month = {m2} THEN avg_speed END) AS m2_speed
                FROM '{p_path}'
                WHERE year = {mom_year} AND month IN ({m1}, {m2})
                GROUP BY geohash
                HAVING m1_speed IS NOT NULL AND m2_speed IS NOT NULL
            """).df()
            db.close()
            if not geo_month_comp.empty:
                geo_month_comp["speed_change"] = geo_month_comp["m2_speed"] - geo_month_comp["m1_speed"]
                geo_month_comp["district"] = geo_month_comp.apply(lambda r: get_district_name(r["lat"], r["lon"]), axis=1)
                by_dist_mom = geo_month_comp.groupby("district").agg(avg_change=("speed_change", "mean")).reset_index()
                by_dist_mom = by_dist_mom.sort_values("avg_change", ascending=False)

                dm1, dm2 = st.columns(2)
                with dm1:
                    st.markdown(f"**🟢 {T('speeding_districts')} ({month_names[m1]}→{month_names[m2]})**")
                    top3_mom = by_dist_mom.head(3)
                    for _idx, r in top3_mom.iterrows():
                        st.markdown(f"✅ **{r['district']}** → {T('avg_speed_short')} **+{r['avg_change']:.1f}** km/s {T('increased')}")
                with dm2:
                    st.markdown(f"**🔴 {T('slowing_districts')} ({month_names[m1]}→{month_names[m2]})**")
                    bot3_mom = by_dist_mom.tail(3)
                    for _idx, r in bot3_mom.iterrows():
                        st.markdown(f"⚠️ **{r['district']}** → {T('avg_speed_short')} **{r['avg_change']:.1f}** km/s {T('decreased')}")

# ═════════════════════════════════════════════════════════════════════════
#  TAB 5 — YZ TAHMİN + HAVA DURUMU (v3.2 — DİNAMİK)
# ═════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"### {T('ai_prediction')}")
    st.markdown(T("ai_prediction_subtitle"))

    pred_col1, pred_col2 = st.columns([1, 2])

    with pred_col1:
        st.markdown(f"#### 📅 {T('date_and_time')}")

        # Tek tarih seçici
        min_date = date(2020, 1, 1)
        max_date = date(2030, 12, 31)
        pred_date = st.date_input(
            f"📅 {T('select_date')}",
            value=date(2024, 10, 15),
            min_value=min_date,
            max_value=max_date,
            help=T("date_help"),
        )
        pred_hour = st.slider(f"🕐 {T('hour')}", 0, 23, 8, key="pred_hour")

        # Tarihten otomatik çıkarımlar
        pred_dow_val = pred_date.weekday()
        pred_month_val = pred_date.month
        pred_year_val = pred_date.year
        _day_names_list = T("day_names")
        pred_dow_name = _day_names_list[pred_dow_val]
        is_past = pred_date <= date.today()

        st.info(f"📌 **{pred_dow_name}**, {pred_date.day} {month_names[pred_month_val]} {pred_year_val}, {T('hour')} {pred_hour}:00")
        if is_past:
            st.caption(f"📂 {T('past_date_caption')}")
        else:
            st.caption(f"🔮 {T('future_date_caption')}")

        st.markdown("---")
        st.markdown(f"#### 🤖 {T('ai_model_select')}")
        selected_model = st.selectbox(
            T("prediction_model"), list(AI_MODELS.keys()), index=0,
            format_func=lambda m: f"{AI_MODELS[m]['icon']} {m} ({T('accuracy')}: %{AI_MODELS[m]['accuracy']})",
        )
        model_info = AI_MODELS[selected_model]
        _model_desc = T("model_descriptions").get(selected_model, "")
        st.markdown(f"""
        <div class="model-card-selected">
            <span style="font-size:2em;">{model_info['icon']}</span><br>
            <b style="font-size:1.2em;">{selected_model}</b><br>
            <small>{_model_desc}</small><br><br>
            <span style="color:{model_info['color']}; font-size:1.5em; font-weight:800;">%{model_info['accuracy']}</span><br>
            <small>{T('accuracy_rate')}</small><br>
            <small>MAE: {model_info['mae']} | RMSE: {model_info['rmse']}</small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        use_weather = st.checkbox(f"🌦️ {T('weather_api')}", value=True)

        st.markdown("---")
        predict_btn = st.button(f"🔮 {T('predict_btn')}", type="primary", use_container_width=True)

    with pred_col2:
        # ── TAHMİN BUTONUNA BASILDIĞINDA: baz verileri session_state'e kaydet ──
        if predict_btn:
            p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
            db = duckdb.connect()
            hist = db.execute(f"""
                SELECT
                    AVG(total_vehicles) AS pred_vehicles,
                    AVG(city_avg_speed) AS pred_speed,
                    STDDEV(total_vehicles) AS std_vehicles,
                    STDDEV(city_avg_speed) AS std_speed,
                    COUNT(*) AS sample_count
                FROM '{p}'
                WHERE hour = {pred_hour}
                  AND day_of_week = {pred_dow_val}
                  AND month = {pred_month_val}
            """).df()

            day_profile = db.execute(f"""
                SELECT hour,
                       AVG(total_vehicles) AS avg_vehicles,
                       AVG(city_avg_speed) AS avg_speed
                FROM '{p}'
                WHERE day_of_week = {pred_dow_val}
                  AND month = {pred_month_val}
                GROUP BY hour ORDER BY hour
            """).df()
            db.close()

            if not hist.empty and hist.iloc[0]["sample_count"] > 0:
                r = hist.iloc[0]
                rain_impact = 0.0
                weather_info = None

                if use_weather:
                    date_str = pred_date.strftime("%Y-%m-%d")
                    with st.spinner(f"🌐 Open-Meteo API {date_str} {T('weather_fetching')}..."):
                        weather_info = get_hourly_weather(41.015, 29.01, date_str, pred_hour)
                    if weather_info:
                        rain_impact = weather_info["rain_impact"]
                    else:
                        st.warning(T("weather_unavailable"))

                # Ground truth (geçmiş tarih ise)
                ground_truth = None
                if is_past:
                    ground_truth = load_ground_truth(pred_date.strftime("%Y-%m-%d"), pred_hour)

                st.session_state.prediction_base = {
                    "base_vehicles": r["pred_vehicles"],
                    "base_speed": r["pred_speed"],
                    "std_vehicles": r["std_vehicles"],
                    "std_speed": r["std_speed"],
                    "sample_count": r["sample_count"],
                    "rain_impact": rain_impact,
                    "weather_info": weather_info,
                    "ground_truth": ground_truth,
                    "pred_date": pred_date,
                    "pred_hour": pred_hour,
                    "pred_dow_val": pred_dow_val,
                    "pred_month_val": pred_month_val,
                    "pred_year_val": pred_year_val,
                    "pred_dow_name": pred_dow_name,
                    "is_past": is_past,
                    "day_profile": day_profile,
                    "use_weather": use_weather,
                }
            else:
                st.session_state.prediction_base = None
                st.warning(T("no_data_combination"))

        # ── SONUÇLARI DİNAMİK OLARAK GÖSTER ─────────────────────
        base = st.session_state.prediction_base
        if base is not None:
            # Dinamik model değiştirme: gerçek ML modeli ile tahmin
            prediction = real_model_predict(
                selected_model, base["pred_hour"], base["pred_dow_val"],
                base["pred_month_val"], base["pred_year_val"], base["rain_impact"]
            )
            all_model_results = {}
            for mn in AI_MODELS:
                all_model_results[mn] = real_model_predict(
                    mn, base["pred_hour"], base["pred_dow_val"],
                    base["pred_month_val"], base["pred_year_val"], base["rain_impact"]
                )

            m_info = AI_MODELS[selected_model]
            adj_vehicles = prediction["vehicles"]
            adj_speed = prediction["speed"]

            # ── Hava durumu bilgisi ──────────────────────────────
            wi = base["weather_info"]
            if wi:
                api_label = f"🔮 {T('forecast_api')}" if wi.get("is_forecast", False) else f"📂 {T('archive_api')}"
                ri = base["rain_impact"]
                st.markdown(f"""
                <div class="weather-box">
                    <h4 style="margin:0 0 8px 0;">🌦️ {T('weather_title')} — {base['pred_date'].strftime('%d/%m/%Y')} {T('hour')} {base['pred_hour']}:00 — <small>{api_label}</small></h4>
                    <span style="font-size:2em;">{wi['icon']}</span>
                    <b style="font-size:1.3em;"> {wi['description']}</b><br>
                    🌡️ <b>{T('temperature')}:</b> {wi['temperature']}°C &nbsp;|&nbsp;
                    🌧️ <b>{T('rain')}:</b> {wi['rain']} mm<br>
                    {"⚠️ <b style='color:#f97316;'>" + T('rain_detected') + " +" + str(int(ri*100)) + "%.</b>" if ri > 0 else "✅ <b style='color:#22c55e;'>" + T('dry_weather') + ".</b>"}
                </div>
                """, unsafe_allow_html=True)

            # ── ANA TAHMİN vs GERÇEK VERİ ────────────────────────
            gt = base["ground_truth"]

            def _traffic_level(spd):
                if spd < 25: return T("very_heavy"), "#ef4444"
                elif spd < 35: return T("heavy"), "#f97316"
                elif spd < 50: return T("moderate"), "#eab308"
                else: return T("flowing"), "#22c55e"

            if gt:
                # Geçmiş tarih: Tahmin ve Gerçek Veri yan yana
                st.markdown(f"#### 📊 {m_info['icon']} {selected_model} {T('prediction_vs_actual')}")
                gt_col1, gt_col2 = st.columns(2)
                with gt_col1:
                    level, _ = _traffic_level(adj_speed)
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h4 style="margin:0 0 10px 0;">🔮 {T('ai_pred_label')} ({selected_model})</h4>
                        🚗 <b>{T('vehicle_lbl')}:</b> <span style="font-size:1.4em; font-weight:800;">{adj_vehicles:,.0f}</span><br>
                        ⚡ <b>{T('speed_lbl')}:</b> <span style="font-size:1.4em; font-weight:800;">{adj_speed:.0f} km/s</span><br>
                        📈 <b>{T('status_lbl')}:</b> {level}<br>
                        📏 {T('confidence')}: {prediction['ci_vehicles_low']:,.0f} — {prediction['ci_vehicles_high']:,.0f} {T('vehicles_unit')}<br>
                        📐 {T('confidence')}: {prediction['ci_speed_low']:.0f} — {prediction['ci_speed_high']:.0f} km/s
                    </div>
                    """, unsafe_allow_html=True)
                with gt_col2:
                    gt_level, _ = _traffic_level(gt["speed"])
                    v_err = abs(adj_vehicles - gt["vehicles"]) / max(gt["vehicles"], 1) * 100
                    s_err = abs(adj_speed - gt["speed"]) / max(gt["speed"], 1) * 100
                    st.markdown(f"""
                    <div class="ground-truth-box">
                        <h4 style="margin:0 0 10px 0;">✅ {T('actual_label')}</h4>
                        🚗 <b>{T('vehicle_lbl')}:</b> <span style="font-size:1.4em; font-weight:800;">{gt['vehicles']:,.0f}</span><br>
                        ⚡ <b>{T('speed_lbl')}:</b> <span style="font-size:1.4em; font-weight:800;">{gt['speed']:.0f} km/s</span><br>
                        📈 <b>{T('status_lbl')}:</b> {gt_level}<br><br>
                        📊 <b style="color:#eab308;">{T('vehicle_error')}: %{v_err:.1f}</b> | <b style="color:#eab308;">{T('speed_error')}: %{s_err:.1f}</b>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Gelecek tarih veya veri yok: sadece tahmin
                st.markdown(f"#### 📊 {T('prediction_result')} — {m_info['icon']} {selected_model}")
                pc1, pc2, pc3 = st.columns(3)
                with pc1: st.metric(f"🚗 {T('estimated_vehicle')}", f"{adj_vehicles:,.0f}")
                with pc2: st.metric(f"⚡ {T('estimated_speed')}", f"{adj_speed:.0f} km/s")
                with pc3:
                    level, lc = _traffic_level(adj_speed)
                    st.markdown(f"### <span style='color:{lc}'>{level}</span>", unsafe_allow_html=True)

            # ── Detay kutusu ─────────────────────────────────────
            _model_desc2 = T("model_descriptions").get(selected_model, "")
            st.markdown(f"""
            <div class="info-box">
                <b>📋 {selected_model} {T('pred_details')}:</b><br>
                • <b>{T('model_lbl')}:</b> {m_info['icon']} {selected_model} ({T('accuracy')}: %{m_info['accuracy']})<br>
                • <b>MAE:</b> {m_info['mae']} | <b>RMSE:</b> {m_info['rmse']}<br>
                • <b>{T('date_lbl')}:</b> {base['pred_date'].strftime('%d/%m/%Y')} {base['pred_dow_name']} | <b>{T('hour')}:</b> {base['pred_hour']}:00<br>
                • <b>{T('ci_vehicle')}:</b> {prediction['ci_vehicles_low']:,.0f} — {prediction['ci_vehicles_high']:,.0f}<br>
                • <b>{T('ci_speed')}:</b> {prediction['ci_speed_low']:.0f} — {prediction['ci_speed_high']:.0f} km/s<br>
                • <b>{T('base_vehicle')}:</b> {base['base_vehicles']:,.0f} | <b>{T('base_speed')}:</b> {base['base_speed']:.0f} km/s<br>
                • <b>{T('sample_count')}:</b> {base['sample_count']:.0f} {T('hourly_records')}
                {"<br>• <b>" + T('weather_impact') + ":</b> +" + str(int(base['rain_impact']*100)) + "%" if base['rain_impact'] > 0 else ""}
            </div>
            """, unsafe_allow_html=True)

            # ── TÜM MODELLER KARŞILAŞTIRMASI ─────────────────────
            st.markdown(f"#### 🔬 {T('all_models_compare')}")
            compare_rows = []
            for mn, pred_data in all_model_results.items():
                co2_d = calculate_co2(pred_data["vehicles"], pred_data["speed"])
                is_sel = "✅" if mn == selected_model else ""

                row_data = {
                    "": is_sel,
                    "Model": f"{AI_MODELS[mn]['icon']} {mn}",
                    f"{T('accuracy')} (%)": AI_MODELS[mn]["accuracy"],
                    T("estimated_vehicle"): f"{pred_data['vehicles']:,.0f}",
                    T("estimated_speed"): f"{pred_data['speed']:.0f} km/s",
                    "CO₂ (kg)": f"{co2_d['total_kg']:,.1f}",
                }

                if gt:
                    v_err = abs(pred_data["vehicles"] - gt["vehicles"]) / max(gt["vehicles"], 1) * 100
                    s_err = abs(pred_data["speed"] - gt["speed"]) / max(gt["speed"], 1) * 100
                    row_data[T("vehicle_error_pct")] = f"{v_err:.1f}"
                    row_data[T("speed_error_pct")] = f"{s_err:.1f}"

                compare_rows.append(row_data)

            compare_df = pd.DataFrame(compare_rows)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            # CO2
            co2 = calculate_co2(adj_vehicles, adj_speed)
            st.markdown(f"""
            <div class="co2-box">
                <h4 style="margin:0 0 8px 0;">🌿 {T('co2_title')}</h4>
                <span style="font-size:1.8em; font-weight:800;">{co2['total_kg']:,.1f} kg</span>
                <span style="color:#6dbe82;"> ({co2['total_ton']} ton)</span><br>
                📊 {T('per_vehicle')}: <b>{co2['per_vehicle_g']:,.0f} g CO₂</b> | {T('speed_factor')}: <b>×{co2['speed_factor']}</b>
            </div>
            """, unsafe_allow_html=True)

            # Sinyalizasyon
            if adj_speed < 30:
                extra_green = min(int((30 - adj_speed) * 1.5), 30)
                st.markdown(f"""
                <div class="signal-box">
                    <h4 style="margin:0 0 8px 0;">🚦 {T('signal_title')}</h4>
                    ⚠️ {T('avg_speed_label')} <b>{adj_speed:.0f} km/s</b>.<br>
                    ✅ <b style="color:#22c55e;">{T('green_light_advice').format(extra_green=extra_green)}</b><br>
                    📉 {T('wait_reduction')}: <b>~%{min(extra_green, 25)}</b>
                </div>
                """, unsafe_allow_html=True)

            # Saatlik profil
            st.markdown(f"#### 📈 {base['pred_dow_name']} {T('hourly_profile')} ({month_names[base['pred_month_val']]})")
            day_profile = base["day_profile"]
            if not day_profile.empty:
                fig_pred = make_subplots(specs=[[{"secondary_y":True}]])
                fig_pred.add_trace(go.Bar(x=day_profile["hour"],y=day_profile["avg_vehicles"],
                                         name=T("vehicle_count_lbl"),marker_color="#667eea",opacity=0.6),secondary_y=False)
                fig_pred.add_trace(go.Scatter(x=day_profile["hour"],y=day_profile["avg_speed"],
                                             name=T("avg_speed_lbl_short"),line=dict(color="#f97316",width=3),
                                             mode="lines+markers"),secondary_y=True)
                fig_pred.add_vline(x=base["pred_hour"],line_dash="dash",line_color="#ef4444",line_width=2)
                fig_pred.add_annotation(x=base["pred_hour"],y=1.05,yref="paper",
                                        text=f"📍 {base['pred_hour']}:00",showarrow=False,
                                        font=dict(color="#ef4444",size=14,family="Inter"))

                if base["rain_impact"] > 0:
                    adjusted_v = day_profile["avg_vehicles"] * (1 + base["rain_impact"])
                    fig_pred.add_trace(go.Scatter(
                        x=day_profile["hour"],y=adjusted_v,
                        name=f"🌧️ {T('rainy')} (+{int(base['rain_impact']*100)}%)",
                        line=dict(color="#ef4444",width=2,dash="dot"),mode="lines"),secondary_y=False)

                fig_pred.update_layout(template="plotly_dark",height=400,margin=dict(l=20,r=20,t=30,b=20),
                                       legend=dict(orientation="h",y=1.12),hovermode="x unified")
                fig_pred.update_yaxes(title_text=T("vehicle_count_lbl"),secondary_y=False)
                fig_pred.update_yaxes(title_text=T("speed_kmh"),secondary_y=True)
                st.plotly_chart(fig_pred, use_container_width=True)

            # ── ROTA OPTİMİZASYONU ───────────────────────────────
            st.markdown("---")
            st.markdown(f"### 🗺️ {T('route_optimization')}")
            rt_col1, rt_col2 = st.columns(2)
            with rt_col1:
                start_district = st.selectbox(f"🟢 {T('start_district')}",list(ISTANBUL_DISTRICTS.keys()),index=0,key="route_start")
            with rt_col2:
                end_district = st.selectbox(f"🔴 {T('end_district')}",list(ISTANBUL_DISTRICTS.keys()),index=1,key="route_end")

            if start_district != end_district:
                
                # Coordinates
                start_pos = ISTANBUL_DISTRICTS[start_district]
                end_pos = ISTANBUL_DISTRICTS[end_district]
                lat1, lon1 = start_pos[0], start_pos[1]
                lat2, lon2 = end_pos[0], end_pos[1]
                # OpenStreetMap yol ağından gerçek rotalar (sessiz/arka planda, önbellekli)
                _osm = gather_osm_alternatives(start_district, end_district,
                                               lat1, lon1, lat2, lon2)

                if _osm:
                    # Dakika hesabı: o tarihin tahmini araç yoğunluğu / profil ortalaması
                    _dp = base.get("day_profile")
                    if _dp is not None and not _dp.empty:
                        _veh_ref = float(_dp["avg_vehicles"].mean())
                    else:
                        _veh_ref = max(float(adj_vehicles), 1.0)
                    routes = [blend_osm_route(i, r, adj_speed, float(adj_vehicles), _veh_ref)
                              for i, r in enumerate(_osm)]
                    # Aynı isimli alternatifleri ayrıştır
                    _seen = {}
                    for _r in routes:
                        if _r["name"] in _seen:
                            _seen[_r["name"]] += 1
                            _r["name"] += f" #{_seen[_r['name']]}"
                        else:
                            _seen[_r["name"]] = 1
                    _best = min(range(len(routes)), key=lambda i: routes[i]["duration_min"])
                    routes[_best]["recommended"] = True
                else:
                    st.warning(f"🌐 {T('osrm_unavailable')}")
                    routes = simulate_route(start_district, end_district, adj_speed, base["pred_hour"])
                
                route_col1, route_col2 = st.columns([1, 1])
                
                with route_col1:
                    st.markdown(f"#### 📋 {T('alt_routes')}")

                    selected_route = st.selectbox(
                        f"🗺️ {T('select_route_map')}",
                        options=[r["name"] for r in routes],
                        index=0
                    )

                    for route in routes:
                        is_best = route.get("recommended", False)
                        is_selected = (route["name"] == selected_route)

                        bg_style = "background-color: rgba(249, 115, 22, 0.08);" if is_selected else ""
                        border = "border: 2px solid #22c55e;" if is_best else ("border: 2px solid #f97316;" if is_selected else "border: 1px solid #2a4a7a;")
                        badge = f' <span style="background:#22c55e;color:#000;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:700;">✅ {T("recommended")}</span>' if is_best else ""

                        st.markdown(f"""
                        <div class="route-box" style="{border} {bg_style}">
                            <h4 style="margin:0 0 8px 0;">{route['name']}{badge}</h4>
                            ⏱️ <b>{route['duration_min']} {T('min')}</b> &nbsp;|&nbsp;
                            📏 <b>{route['distance_km']} km</b> &nbsp;|&nbsp;
                            ⚡ <b>{route['avg_speed']} km/s</b> &nbsp;|&nbsp;
                            {route['traffic_level']}<br>
                            🌿 <b>{route['co2_g']} g CO₂</b>
                        </div>
                        """, unsafe_allow_html=True)

                with route_col2:
                    st.markdown(f"#### 🗺️ {T('route_map')}")

                    _route_layers = []
                    for r in routes:
                        coords = r.get("geometry") or []
                        if not coords:
                            continue
                        # Çizim için noktaları seyrekleştir (~600 nokta yeter)
                        step = max(1, len(coords) // 600)
                        pts = [[c[1], c[0]] for c in coords[::step]]
                        if pts[-1] != [coords[-1][1], coords[-1][0]]:
                            pts.append([coords[-1][1], coords[-1][0]])
                        lvl = r["traffic_level"]
                        color = ("#22c55e" if "Akıcı" in lvl
                                 else "#eab308" if "Orta" in lvl else "#ef4444")
                        _route_layers.append({
                            "name": r["name"], "coords": pts, "color": color,
                            "selected": r["name"] == selected_route,
                            "info": f"⏱️ {r['duration_min']} dk • 📏 {r['distance_km']} km",
                        })

                    if _route_layers:
                        _map_payload = json.dumps({
                            "routes": _route_layers,
                            "start": {"lat": lat1, "lon": lon1, "label": f"🟢 {T('start_label')}: {start_district}"},
                            "end": {"lat": lat2, "lon": lon2, "label": f"🏁 {T('end_label')}: {end_district}"},
                        }, ensure_ascii=False)
                        _leaflet_html = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="rmap" style="height:470px;border-radius:16px;overflow:hidden;
     border:1px solid rgba(99,102,241,0.35);box-shadow:0 10px 30px rgba(0,0,0,0.4);"></div>
<script>
  var data = __DATA__;
  var map = L.map('rmap');
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors', maxZoom: 19
  }).addTo(map);
  // Seçili olmayan alternatifler: gri kesikli
  data.routes.forEach(function(r) {
      if (r.selected) return;
      L.polyline(r.coords, {color:'#64748b', weight:4, opacity:0.6, dashArray:'6 8'})
        .bindTooltip('<b>' + r.name + '</b><br>' + r.info).addTo(map);
  });
  // Seçili rota: beyaz dış hat + trafik rengi çekirdek
  var selLayer = null;
  data.routes.forEach(function(r) {
      if (!r.selected) return;
      L.polyline(r.coords, {color:'#ffffff', weight:9, opacity:0.9}).addTo(map);
      selLayer = L.polyline(r.coords, {color:r.color, weight:5, opacity:1})
        .bindTooltip('<b>' + r.name + '</b><br>' + r.info).addTo(map);
  });
  L.circleMarker([data.start.lat, data.start.lon],
      {radius:8, color:'#ffffff', weight:2, fillColor:'#22c55e', fillOpacity:1})
      .bindTooltip(data.start.label).addTo(map);
  L.circleMarker([data.end.lat, data.end.lon],
      {radius:8, color:'#ffffff', weight:2, fillColor:'#ef4444', fillOpacity:1})
      .bindTooltip(data.end.label).addTo(map);
  if (selLayer) { map.fitBounds(selLayer.getBounds(), {padding:[30,30]}); }
  else { map.setView([41.015, 29.01], 10); }
</script>"""
                        components.html(_leaflet_html.replace("__DATA__", _map_payload), height=486)
                    else:
                        st.info(T("route_geometry_missing"))
            else:
                st.info(T("select_diff_districts"))

        elif not predict_btn:
            st.info(f"👆 {T('predict_hint')}")
            st.markdown(f"""
            <div class="info-box">
               <b>🧠 {T('ai_engine_title')}:</b><br><br>
               {T('ai_engine_desc')}
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  TAB 6 — YZ MODELLERİ KARŞILAŞTIRMA
# ═════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"### 🤖 {T('ai_models_compare')}")
    st.markdown(T("ai_models_subtitle"))

    st.markdown(f"#### 📊 {T('model_performance')}")
    model_cols = st.columns(len(AI_MODELS))
    for i, (mn, md) in enumerate(AI_MODELS.items()):
        with model_cols[i]:
            acc_color = "#22c55e" if md["accuracy"]>=90 else "#eab308" if md["accuracy"]>=80 else "#f97316"
            _desc_card = T("model_descriptions").get(mn, "")
            st.markdown(f"""
            <div class="model-card">
                <span style="font-size:2.5em;">{md['icon']}</span><br>
                <b style="font-size:1.3em;">{mn}</b><br><br>
                <span style="color:{acc_color}; font-size:2.2em; font-weight:800;">%{md['accuracy']}</span><br>
                <small style="color:#aaa;">{T('accuracy_rate')}</small><br><br>
                📏 MAE: <b>{md['mae']}</b><br>📐 RMSE: <b>{md['rmse']}</b><br><br>
                <small style="color:#888;">{_desc_card}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"#### 📈 {T('model_compare_charts')}")
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        model_names = list(AI_MODELS.keys())
        accuracies = [AI_MODELS[m]["accuracy"] for m in model_names]
        colors = [AI_MODELS[m]["color"] for m in model_names]
        fig_acc = go.Figure(data=[go.Bar(x=model_names,y=accuracies,marker_color=colors,
                                         text=[f"%{a}" for a in accuracies],textposition="outside",
                                         textfont=dict(size=16,color="white"))])
        fig_acc.update_layout(template="plotly_dark",title=T("model_accuracy_title"),
                              yaxis_range=[0,105],height=400,margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_acc, use_container_width=True)

    with comp_col2:
        maes = [AI_MODELS[m]["mae"] for m in model_names]
        rmses = [AI_MODELS[m]["rmse"] for m in model_names]
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(name="MAE",x=model_names,y=maes,marker_color="#667eea",
                                 text=[str(m) for m in maes],textposition="outside"))
        fig_err.add_trace(go.Bar(name="RMSE",x=model_names,y=rmses,marker_color="#f97316",
                                 text=[str(r) for r in rmses],textposition="outside"))
        fig_err.update_layout(template="plotly_dark",title=T("model_error_title"),
                              barmode="group",height=400,margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_err, use_container_width=True)

    # Tüm modeller ile tahmin
    st.markdown("---")
    st.markdown(f"#### 🔮 {T('all_models_predict')}")
    cmp_col1, cmp_col2 = st.columns([1, 3])

    with cmp_col1:
        cmp_date = st.date_input(f"📅 {T('date_lbl')}", value=date(2024, 6, 15), key="cmp_date",
                                  min_value=date(2020,1,1), max_value=date.today()+timedelta(days=14))
        cmp_hour = st.slider(f"🕐 {T('hour')}", 0, 23, 17, key="cmp_hour")
        compare_btn = st.button(f"🔬 {T('compare_btn')}", type="primary", use_container_width=True)

    with cmp_col2:
        if compare_btn:
            cmp_dow = cmp_date.weekday()
            cmp_month = cmp_date.month

            p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
            db = duckdb.connect()
            hist = db.execute(f"""
                SELECT AVG(total_vehicles) AS pred_vehicles, AVG(city_avg_speed) AS pred_speed
                FROM '{p}'
                WHERE hour = {cmp_hour} AND day_of_week = {cmp_dow} AND month = {cmp_month}
            """).df()
            db.close()

            if not hist.empty and hist.iloc[0]["pred_vehicles"] is not None:
                base_v = hist.iloc[0]["pred_vehicles"]
                base_s = hist.iloc[0]["pred_speed"]

                # Ground truth for comparison
                cmp_gt = load_ground_truth(cmp_date.strftime("%Y-%m-%d"), cmp_hour) if cmp_date <= date.today() else None

                results = []
                for mn_name in AI_MODELS:
                    pred = real_model_predict(mn_name, cmp_hour, cmp_dow, cmp_month, cmp_date.year)
                    co2_data = calculate_co2(pred["vehicles"], pred["speed"])
                    row = {
                        "Model": f"{AI_MODELS[mn_name]['icon']} {mn_name}",
                        f"{T('accuracy')} (%)": AI_MODELS[mn_name]["accuracy"],
                        T("estimated_vehicle"): f"{pred['vehicles']:,.0f}",
                        f"{T('estimated_speed')} (km/s)": f"{pred['speed']:.0f}",
                        "CO₂ (kg)": f"{co2_data['total_kg']:,.1f}",
                    }
                    if cmp_gt:
                        v_err = abs(pred["vehicles"] - cmp_gt["vehicles"]) / max(cmp_gt["vehicles"], 1) * 100
                        s_err = abs(pred["speed"] - cmp_gt["speed"]) / max(cmp_gt["speed"], 1) * 100
                        row[T("vehicle_error_pct")] = f"{v_err:.1f}"
                        row[T("speed_error_pct")] = f"{s_err:.1f}"
                    results.append(row)

                if cmp_gt:
                    st.success(f"✅ {T('actual_on_day')}: **{cmp_gt['vehicles']:,.0f}** {T('vehicles_unit')}, **{cmp_gt['speed']:.0f}** km/s")

                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

                # Grafik
                vehicle_vals = [real_model_predict(m, cmp_hour, cmp_dow, cmp_month, cmp_date.year)["vehicles"] for m in AI_MODELS]
                speed_vals = [real_model_predict(m, cmp_hour, cmp_dow, cmp_month, cmp_date.year)["speed"] for m in AI_MODELS]
                model_labels = [f"{AI_MODELS[m]['icon']} {m}" for m in AI_MODELS]

                fig_cmp = make_subplots(rows=1,cols=2,subplot_titles=[T("estimated_vehicle"), f"{T('estimated_speed')} (km/s)"])
                fig_cmp.add_trace(go.Bar(x=model_labels,y=vehicle_vals,
                                         marker_color=[AI_MODELS[m]["color"] for m in AI_MODELS],
                                         text=[f"{v:,.0f}" for v in vehicle_vals],textposition="outside"),row=1,col=1)
                fig_cmp.add_trace(go.Bar(x=model_labels,y=speed_vals,
                                         marker_color=[AI_MODELS[m]["color"] for m in AI_MODELS],
                                         text=[f"{s:.0f}" for s in speed_vals],textposition="outside"),row=1,col=2)

                if cmp_gt:
                    fig_cmp.add_hline(y=cmp_gt["vehicles"],line_dash="dash",line_color="#22c55e",
                                      annotation_text=T("actual_short"),row=1,col=1)
                    fig_cmp.add_hline(y=cmp_gt["speed"],line_dash="dash",line_color="#22c55e",
                                      annotation_text=T("actual_short"),row=1,col=2)

                fig_cmp.update_layout(template="plotly_dark",height=400,showlegend=False,
                                      margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig_cmp, use_container_width=True)
            else:
                st.warning(T("no_data_combination"))
        else:
            st.info(f"👆 {T('compare_hint')}")

            # Radar chart
            st.markdown(f"#### 🎯 {T('radar_chart_title')}")
            categories = T("radar_categories")
            fig_radar = go.Figure()
            model_radar = {
                "Gradient Boosting":[83,75,85,60,80],"Random Forest":[80,90,82,70,75],
                "XGBoost":[79,95,80,65,85],"Ridge Regression":[58,99,55,95,40],
            }
            for mn, vals in model_radar.items():
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]], theta=categories+[categories[0]],
                    name=f"{AI_MODELS[mn]['icon']} {mn}",
                    line=dict(color=AI_MODELS[mn]["color"],width=2),
                    fill="toself", opacity=0.7,
                ))
            fig_radar.update_layout(template="plotly_dark",
                                     polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                                     height=500,margin=dict(l=40,r=40,t=20,b=20))
            st.plotly_chart(fig_radar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
#  TAB 6 — VERİ EKLE / KARŞILAŞTIR (Session Bazlı)
# ═════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown(f"### 📥 {T('data_upload_title')}")
    st.markdown(T("data_upload_subtitle"))

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = []

    if "upload_success_msg" not in st.session_state:
        st.session_state.upload_success_msg = ""

    upload_col, info_col = st.columns([2, 1])

    with upload_col:
        if st.session_state.upload_success_msg:
            st.success(st.session_state.upload_success_msg)
            st.session_state.upload_success_msg = ""

        uploaded_file = st.file_uploader(
            f"📂 {T('upload_csv_label')}",
            type=["csv"], accept_multiple_files=False,
            key=f"csv_uploader_{st.session_state.uploader_key}"
        )
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                
                # Column mapping for compatibility (dataset vs desktop csv)
                COLUMN_MAP = {
                    "DATE_TIME": "date_time",
                    "LATITUDE": "lat",
                    "LONGITUDE": "lon",
                    "GEOHASH": "geohash",
                    "AVERAGE_SPEED": "avg_speed",
                    "NUMBER_OF_VEHICLES": "vehicle_count",
                }
                df_upload = df_upload.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df_upload.columns})
                
                # Auto-calculate hour if missing
                if "hour" not in df_upload.columns and "date_time" in df_upload.columns:
                    df_upload["hour"] = pd.to_datetime(df_upload["date_time"]).dt.hour
                
                # Auto-calculate geohash if missing
                if "geohash" not in df_upload.columns and "lat" in df_upload.columns and "lon" in df_upload.columns:
                    try:
                        import pygeohash as pgh
                        df_upload["geohash"] = df_upload.apply(lambda r: pgh.encode(r["lat"], r["lon"], precision=5), axis=1)
                    except Exception as e_gh:
                        st.warning(f"Geohash hesaplanamadı: {e_gh}")
                
                required_cols = {"date_time", "lat", "lon", "vehicle_count", "avg_speed"}
                if required_cols.issubset(set(df_upload.columns)):
                    st.session_state.uploaded_data.append({
                        "name": uploaded_file.name,
                        "data": df_upload,
                    })
                    st.session_state.upload_success_msg = f"✅ **{uploaded_file.name}** {T('uploaded')}! ({len(df_upload)} {T('records')})"
                    st.session_state.uploader_key += 1
                    st.rerun()
                else:
                    missing = required_cols - set(df_upload.columns)
                    st.error(f"❌ {T('missing_cols')}: {', '.join(missing)}. {T('csv_format_required')}")
            except Exception as e:
                st.error(f"❌ {T('file_read_error')}: {e}")

    with info_col:
        st.markdown(f"**📋 {T('uploaded_data')}:**")
        if not st.session_state.uploaded_data:
            st.info(T("no_data_uploaded"))
        else:
            for i, item in enumerate(list(st.session_state.uploaded_data)):
                cols_r = st.columns([3, 1])
                cols_r[0].markdown(f"📄 **{item['name']}** ({len(item['data'])} {T('records')})")
                if cols_r[1].button("🗑️", key=f"del_upload_{i}"):
                    st.session_state.uploaded_data.pop(i)
                    st.rerun()

            if st.button(f"🗑️ {T('clear_all')}"):
                st.session_state.uploaded_data = []
                st.rerun()

    # ── Karşılaştırma ────────────────────────────────────────────
    if st.session_state.uploaded_data:
        st.markdown("---")
        st.markdown(f"### 📊 {T('compare_results')}")

        # Tüm yüklenen veriyi birleştir
        all_uploaded = pd.concat([item["data"] for item in st.session_state.uploaded_data], ignore_index=True)

        if "geohash" in all_uploaded.columns and "hour" in all_uploaded.columns:
            up = all_uploaded.dropna(subset=["geohash"]).copy()
            up["gh5"] = up["geohash"].astype(str).str[:5]
            up["hour"] = up["hour"].astype(int)

            # Tarih bağlamı: aynı ay + haftanın günü ile adil karşılaştırma
            _udt = pd.to_datetime(up["date_time"], errors="coerce")
            _u_month = int(_udt.dt.month.mode().iloc[0]) if _udt.notna().any() else selected_month
            _u_dow = int(_udt.dt.weekday.mode().iloc[0]) if _udt.notna().any() else 0
            _u_year = int(_udt.dt.year.mode().iloc[0]) if _udt.notna().any() else selected_year
            _day_names = T("day_names")

            # Sayımı SAATLİK ORANA çevir — İBB satırları 60 dakikalıktır.
            # Örn. 10 dk'da 252 araç ≈ saatte 1512 araç; aksi halde %fark absürt çıkar.
            if "period_minutes" in up.columns:
                _pm = pd.to_numeric(up["period_minutes"], errors="coerce").clip(0.5, 120).fillna(60.0)
            else:
                # Eski CSV'ler: periyodu ardışık satır farkından çıkar (tek satırda 60 dk varsay)
                _diffs = _udt.sort_values().diff().dt.total_seconds().div(60)
                _med = float(_diffs.median()) if _diffs.notna().sum() else 60.0
                _pm = pd.Series(min(max(_med, 0.5), 120.0), index=up.index)
            up["vehicle_rate"] = up["vehicle_count"] * (60.0 / _pm)

            # Tek sorguda tüm geohash+saat çiftlerinin tarihsel ortalamaları (önbellekli)
            hist = load_comparison_baseline(
                tuple(sorted(up["gh5"].dropna().unique())), _u_month, _u_dow)
            matched = (up.merge(hist, on=["gh5", "hour"], how="left")
                       if not hist.empty else pd.DataFrame())
            if not matched.empty:
                matched = matched.dropna(subset=["hist_vehicles"])

            if matched is None or matched.empty:
                st.info(T("no_geohash_match"))
            else:
                # İlçe tespiti (başlıklar ve tablo için — vektörel)
                _dn2 = list(ISTANBUL_DISTRICTS.keys())
                _dc2 = np.array([ISTANBUL_DISTRICTS[n] for n in _dn2])
                _dd2 = ((matched["lat"].to_numpy()[:, None] - _dc2[None, :, 0]) ** 2
                        + (matched["lon"].to_numpy()[:, None] - _dc2[None, :, 1]) ** 2)
                matched["ilce"] = [_dn2[i] for i in _dd2.argmin(axis=1)]
                _focus_ilce = matched["ilce"].mode().iloc[0]
                _focus_gh = matched["gh5"].mode().iloc[0]
                _single_region = matched["gh5"].nunique() == 1

                # ── Karşılaştırma (saatlik oran bazında — aynı ölçek) ──
                _u_mean = max(float(matched["vehicle_rate"].mean()), 1e-9)
                _h_mean = max(float(matched["hist_vehicles"].mean()), 1e-9)
                matched["endeks_farki"] = (matched["vehicle_rate"] / _u_mean
                                           - matched["hist_vehicles"] / _h_mean) * 100
                matched["hiz_farki"] = matched["avg_speed"] - matched["hist_speed"]

                u_prof = matched.groupby("hour")["vehicle_rate"].mean()
                h_prof = matched.groupby("hour")["hist_vehicles"].mean()
                u_idx = u_prof / max(float(u_prof.mean()), 1e-9) * 100
                h_idx = h_prof / max(float(h_prof.mean()), 1e-9) * 100
                _corr = (float(np.corrcoef(u_idx.values, h_idx.values)[0, 1]) * 100
                         if len(u_idx) >= 3 else None)
                _pct_diff = (_u_mean - _h_mean) / _h_mean * 100
                _spd_diff = float(matched["hiz_farki"].mean())

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric(f"📍 {T('region')}", f"{_focus_ilce}" + ("" if _single_region else " +"),
                           help=f"Geohash: {_focus_gh}" + ("" if _single_region else f" ({T('multiple_regions')})"))
                mc2.metric(f"🔁 {T('pattern_match')}", f"%{_corr:.0f}" if _corr is not None else "—",
                           help=T("pattern_match_help"))
                mc3.metric(f"🚗 {T('hourly_vehicle_you_ibb')}",
                           f"{_u_mean:,.0f} / {_h_mean:,.0f}",
                           delta=f"{_pct_diff:+.0f}%",
                           help=T("hourly_vehicle_help"))
                mc4.metric(f"⚡ {T('avg_speed_diff')}", f"{_spd_diff:+.1f} km/s")

                _basis = matched["basis"].mode().iloc[0]
                _basis_txt = {
                    "Ay+Gün": T("basis_month_day").format(month=month_names[_u_month], day=_day_names[_u_dow]),
                    "Ay": T("basis_month").format(month=month_names[_u_month]),
                    "Genel": T("basis_general"),
                }.get(_basis, _basis)
                st.caption(f"⚖️ {T('comparison_basis')}: {_basis_txt}.")

                if len(u_idx) >= 3:
                    # ── Saatlik profil grafiği (endeks) ─────────────
                    fig_up = go.Figure()
                    fig_up.add_trace(go.Bar(
                        x=h_idx.index, y=h_idx.values,
                        name=f"🏛️ {T('ibb_hist_pattern')}", marker_color="#667eea", opacity=0.7))
                    fig_up.add_trace(go.Scatter(
                        x=u_idx.index, y=u_idx.values,
                        name=f"📹 {T('your_measurement')}", mode="lines+markers",
                        line=dict(color="#22d3ee", width=3)))
                    fig_up.update_layout(
                        template="plotly_dark", height=320,
                        xaxis_title=T("hour"), yaxis_title=T("density_index"),
                        margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_up, use_container_width=True)
                else:
                    st.markdown(f"#### 📍 {_focus_ilce} {T('regional_analysis')} — {T('short_record_mode')}")
                    _hs = sorted(matched["hour"].unique())
                    _hs_lbl = ", ".join(f"{h:02d}:00" for h in _hs)
                    st.caption(T("short_rec_caption_long").format(
                        hours=_hs_lbl, district=_focus_ilce, geohash=_focus_gh))

                    loc = load_location_history(tuple(sorted(up["gh5"].dropna().unique())))
                    if loc.empty:
                        st.info(T("no_ibb_history"))
                    else:
                        _user_speed = float(matched["avg_speed"].mean())
                        _hist_speed = float(matched["hist_speed"].mean())

                        # 24 saat profili (aynı ay + gün; yoksa aynı ay)
                        _hp = loc[(loc["m"] == _u_month) & (loc["dw_py"] == _u_dow)]
                        if _hp.empty:
                            _hp = loc[loc["m"] == _u_month]
                        _hp_g = _hp.groupby("hour").agg(v=("v", "mean"), s=("s", "mean")).reset_index()

                        # Ölçülen saatin tarihsel sıralaması (yoğunluk)
                        _rank_txt = "—"
                        if not _hp_g.empty and _hs:
                            _ranked = _hp_g.sort_values("v", ascending=False).reset_index(drop=True)
                            _pos = _ranked.index[_ranked["hour"].isin(_hs)].tolist()
                            if _pos:
                                _rank_txt = f"{T('rank_in_day')} {_pos[0] + 1}{T('rank_suffix')}"

                        # Yıllara göre (aynı ay + saat; gün filtresi varsa uygula)
                        _yr = loc[(loc["m"] == _u_month) & (loc["hour"].isin(_hs))]
                        _yr_d = _yr[_yr["dw_py"] == _u_dow]
                        _yr_use = _yr_d if not _yr_d.empty else _yr
                        _yr_g = _yr_use.groupby("y").agg(v=("v", "mean"), s=("s", "mean")).reset_index()
                        _busy_year = int(_yr_g.loc[_yr_g["v"].idxmax(), "y"]) if not _yr_g.empty else None

                        rm1, rm2, rm3 = st.columns(3)
                        rm1.metric(f"🕐 {T('hour_rank_in').format(district=_focus_ilce)}", _rank_txt,
                                   help=T("hour_rank_help").format(district=_focus_ilce))
                        rm2.metric(f"⚡ {T('speed_you_vs').format(district=_focus_ilce)}",
                                   f"{_user_speed:.0f} / {_hist_speed:.0f} km/s",
                                   delta=f"{_user_speed - _hist_speed:+.1f} km/s")
                        rm3.metric(f"📅 {T('busiest_year_in').format(district=_focus_ilce)}",
                                   str(_busy_year) if _busy_year else "—",
                                   help=T("busiest_year_help").format(hours=_hs_lbl))

                        rc1, rc2 = st.columns(2)
                        with rc1:
                            fig_yr = make_subplots(specs=[[{"secondary_y": True}]])
                            fig_yr.add_trace(go.Bar(
                                x=_yr_g["y"], y=_yr_g["v"], name=f"🚗 {T('avg_vehicle_lbl')}",
                                marker_color="#667eea", opacity=0.8), secondary_y=False)
                            fig_yr.add_trace(go.Scatter(
                                x=_yr_g["y"], y=_yr_g["s"], name=f"⚡ {T('ibb_speed')}",
                                mode="lines+markers", line=dict(color="#f97316", width=3)), secondary_y=True)
                            fig_yr.add_hline(y=_user_speed, line_dash="dash", line_color="#22d3ee",
                                             annotation_text=f"📹 {T('your_speed')}: {_user_speed:.0f}",
                                             annotation_font_color="#22d3ee", secondary_y=True)
                            fig_yr.update_layout(
                                template="plotly_dark", height=300,
                                title=f"📅 {_focus_ilce} — {T('by_year')} ({T('hour')} {_hs_lbl})",
                                margin=dict(l=10, r=10, t=45, b=10))
                            st.plotly_chart(fig_yr, use_container_width=True)
                        with rc2:
                            _dwg = (loc[(loc["m"] == _u_month) & (loc["hour"].isin(_hs))]
                                    .groupby("dw_py").agg(v=("v", "mean")).reset_index())
                            _dwg["gun"] = _dwg["dw_py"].map(lambda d: _day_names[int(d)])
                            _dwg["renk"] = np.where(_dwg["dw_py"] == _u_dow, "#22d3ee", "#667eea")
                            fig_dw = go.Figure(go.Bar(
                                x=_dwg["gun"], y=_dwg["v"], marker_color=_dwg["renk"]))
                            fig_dw.update_layout(
                                template="plotly_dark", height=300,
                                title=f"📆 {_focus_ilce} — {T('by_day')} ({T('hour')} {_hs_lbl}, 🔵 {T('your_day_marker')})",
                                margin=dict(l=10, r=10, t=45, b=10))
                            st.plotly_chart(fig_dw, use_container_width=True)

                        _hp_g["renk"] = np.where(_hp_g["hour"].isin(_hs), "#22d3ee", "#667eea")
                        fig_24 = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_24.add_trace(go.Bar(
                            x=_hp_g["hour"], y=_hp_g["v"], name=f"🚗 {T('avg_vehicle_lbl')}",
                            marker_color=_hp_g["renk"], opacity=0.85), secondary_y=False)
                        fig_24.add_trace(go.Scatter(
                            x=_hp_g["hour"], y=_hp_g["s"], name=f"⚡ {T('ibb_speed')}",
                            mode="lines", line=dict(color="#f97316", width=2)), secondary_y=True)
                        fig_24.add_trace(go.Scatter(
                            x=_hs, y=[_user_speed] * len(_hs), name=f"📹 {T('your_speed')}",
                            mode="markers", marker=dict(color="#22d3ee", size=14, symbol="diamond",
                                                        line=dict(color="#ffffff", width=2))), secondary_y=True)
                        fig_24.update_layout(
                            template="plotly_dark", height=320,
                            title=f"🕐 {_focus_ilce} {T('24h_profile')} — {month_names[_u_month]}, {_day_names[_u_dow]} (🔵 {T('your_hour_marker')})",
                            xaxis_title=T("hour"), margin=dict(l=10, r=10, t=45, b=10))
                        st.plotly_chart(fig_24, use_container_width=True)

                # ── Detay tablosu (sayısal, sıralanabilir) ──────────
                _col_district = T("col_district")
                _col_geohash = "Geohash"
                _col_hour = T("col_hour")
                _col_your_veh = T("col_your_vehicle")
                _col_ibb_veh = T("col_ibb_vehicle")
                _col_idx_diff = T("col_index_diff")
                _col_your_spd = T("col_your_speed")
                _col_ibb_spd = T("col_ibb_speed")
                _col_spd_diff = T("col_speed_diff")
                _col_basis = T("col_basis")
                tbl = matched[["ilce", "gh5", "hour", "vehicle_rate", "hist_vehicles",
                               "endeks_farki", "avg_speed", "hist_speed", "hiz_farki", "basis"]]
                tbl = tbl.rename(columns={
                    "ilce": _col_district, "gh5": _col_geohash, "hour": _col_hour,
                    "vehicle_rate": _col_your_veh, "hist_vehicles": _col_ibb_veh,
                    "endeks_farki": _col_idx_diff, "avg_speed": _col_your_spd,
                    "hist_speed": _col_ibb_spd, "hiz_farki": _col_spd_diff, "basis": _col_basis,
                }).head(200)
                st.dataframe(
                    tbl, use_container_width=True, hide_index=True,
                    column_config={
                        _col_hour: st.column_config.NumberColumn(format="%d:00"),
                        _col_your_veh: st.column_config.NumberColumn(
                            format="%.0f", help=T("your_vehicle_help")),
                        _col_ibb_veh: st.column_config.NumberColumn(
                            format="%.0f", help=T("ibb_vehicle_help")),
                        _col_idx_diff: st.column_config.NumberColumn(
                            format="%+.1f puan",
                            help=T("index_diff_help")),
                        _col_your_spd: st.column_config.NumberColumn(format="%.0f km/s"),
                        _col_ibb_spd: st.column_config.NumberColumn(format="%.0f km/s"),
                        _col_spd_diff: st.column_config.NumberColumn(format="%+.1f km/s"),
                    })

                if abs(_pct_diff) < 15:
                    st.success(T("compare_compatible").format(district=_focus_ilce, pct=f"{_pct_diff:+.0f}"))
                elif _pct_diff > 0:
                    st.warning(T("compare_denser").format(district=_focus_ilce, pct=f"{abs(_pct_diff):.0f}"))
                else:
                    st.info(T("compare_lighter").format(district=_focus_ilce, pct=f"{abs(_pct_diff):.0f}"))
        else:
            st.warning(T("csv_geohash_required"))

st.markdown("---")
st.markdown(
    f'<p style="text-align: center; color: #8888aa; font-size: 0.9em; margin-top: 30px;">'
    f'{T("footer")}'
    f'</p>',
    unsafe_allow_html=True
)
