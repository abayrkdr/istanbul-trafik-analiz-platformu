"""
generate_synthetic_data.py — 2025 Şubat–Aralık Yapay Veri Üretimi
=================================================================
Geçmiş yılın (2024) aynı gün/saat kalıplarını baz alarak
2025 yılının eksik ayları (Şubat–Aralık) için gerçekçi sentetik veri üretir.
Daha sonra tüm özet tabloları DuckDB ile hızlıca yeniden oluşturur.
"""

import duckdb
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_parquet")

def generate_traffic_all():
    """traffic_all.parquet için 2025 Şubat-Aralık sentetik verisini üretir ve ekler."""
    print("\n📊 traffic_all.parquet işleniyor...")
    p = os.path.join(DATA_DIR, "traffic_all.parquet").replace("\\", "/")
    p_temp = os.path.join(DATA_DIR, "traffic_all_temp.parquet").replace("\\", "/")
    
    t0 = time.time()
    db = duckdb.connect()
    
    print("  Sentetik 2025 verileri 2024 verilerinden üretiliyor ve birleştiriliyor...")
    
    # 2024'ün 2-12 ayları arasındaki verileri çekip tarihleri 364 gün ileri alıp,
    # araç sayılarını %2 artırıp, %5 gürültü ekliyoruz.
    # Tarih bazlı month ve day_of_week değerlerini yeni tarihe göre güncelliyoruz.
    db.execute(f"""
        COPY (
            -- 1. Build 2025 data skeleton by selecting 2024 base records
            WITH skeleton AS (
                SELECT 
                    (date_time + INTERVAL 364 DAY) AS date_time,
                    lat,
                    lon,
                    geohash,
                    min_speed AS speed_2024,
                    max_speed AS max_speed_2024,
                    avg_speed AS avg_speed_2024,
                    vehicle_count AS vehicles_2024,
                    2025 AS year,
                    EXTRACT(MONTH FROM (date_time + INTERVAL 364 DAY)) AS month,
                    hour,
                    EXTRACT(DOW FROM (date_time + INTERVAL 364 DAY)) AS day_of_week
                FROM '{p}'
                WHERE year = 2024 AND month BETWEEN 2 AND 12
            )
            -- 2. Keep existing data, excluding 2025 Feb-Dec (which we will regenerate)
            SELECT * FROM '{p}' WHERE NOT (year = 2025 AND month BETWEEN 2 AND 12)
            UNION ALL
            -- 3. Append the new 2025 records with macro (sin/cos) and micro (random) noise
            SELECT 
                s.date_time,
                s.lat,
                s.lon,
                s.geohash,
                CAST(s.speed_2024 * (1.0 + cos(s.month * 0.5) * 0.02 + (random() * 0.05 - 0.025)) AS INT) AS min_speed,
                CAST(s.max_speed_2024 * (1.0 + cos(s.month * 0.5) * 0.02 + (random() * 0.05 - 0.025)) AS INT) AS max_speed,
                CAST(s.avg_speed_2024 * (1.0 + cos(s.month * 0.5) * 0.02 + (random() * 0.05 - 0.025)) AS INT) AS avg_speed,
                CAST(s.vehicles_2024 * 1.02 * (1.0 + sin(s.month * 0.5) * 0.04 + (random() * 0.1 - 0.05)) AS INT) AS vehicle_count,
                s.year,
                s.month,
                s.hour,
                s.day_of_week
            FROM skeleton s
        ) TO '{p_temp}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    
    db.close()
    
    # Windows dosya kilitlerini serbest bırakması için kısa bir süre bekle
    import gc
    gc.collect()
    time.sleep(1.5)
    
    # Eski dosyayı sil ve yenisini adlandır
    if os.path.exists(p):
        os.remove(p)
    os.rename(p_temp, p)
    elapsed = time.time() - t0
    print(f"  ✅ Saved: traffic_all.parquet (2025 Şubat-Aralık eklendi) ({elapsed:.1f} saniye)")


def generate_summaries_from_all():
    """Güncellenmiş traffic_all.parquet'ten diğer tüm özet dosyaları oluşturur."""
    print("\n⚡ traffic_all.parquet'ten özet tablolar üretiliyor...")
    pq = os.path.join(DATA_DIR, "traffic_all.parquet").replace("\\", "/")
    
    db = duckdb.connect()
    
    # 1. Saatlik şehir geneli özet
    print("  ⏳ Saatlik şehir özeti oluşturuluyor...")
    t = time.time()
    hourly_path = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
    db.execute(f"""
        COPY (
            SELECT
                date_trunc('hour', date_time) AS hour_ts,
                year, month, hour, day_of_week,
                COUNT(*)                      AS record_count,
                SUM(vehicle_count)            AS total_vehicles,
                AVG(avg_speed)                AS city_avg_speed,
                AVG(min_speed)                AS city_avg_min_speed,
                AVG(max_speed)                AS city_avg_max_speed,
                COUNT(DISTINCT geohash)       AS active_locations
            FROM '{pq}'
            GROUP BY date_trunc('hour', date_time), year, month, hour, day_of_week
            ORDER BY hour_ts
        ) TO '{hourly_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"  ✅ summary_hourly.parquet ({time.time()-t:.1f}s)")
    
    # 2. Geohash-bazlı günlük özet
    print("  ⏳ Geohash günlük özeti oluşturuluyor...")
    t = time.time()
    geo_daily_path = os.path.join(DATA_DIR, "summary_geo_daily.parquet").replace("\\", "/")
    db.execute(f"""
        COPY (
            SELECT
                CAST(date_time AS DATE) AS day_date,
                geohash,
                AVG(lat)              AS lat,
                AVG(lon)              AS lon,
                year, month, day_of_week,
                SUM(vehicle_count)    AS total_vehicles,
                AVG(avg_speed)        AS avg_speed,
                AVG(min_speed)        AS avg_min_speed,
                COUNT(*)              AS record_count
            FROM '{pq}'
            GROUP BY CAST(date_time AS DATE), geohash, year, month, day_of_week
            ORDER BY day_date
        ) TO '{geo_daily_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"  ✅ summary_geo_daily.parquet ({time.time()-t:.1f}s)")
    
    # 3. Geohash-saat özeti
    print("  ⏳ Geohash-saat özeti oluşturuluyor...")
    t = time.time()
    geo_hour_path = os.path.join(DATA_DIR, "summary_geo_hourly.parquet").replace("\\", "/")
    db.execute(f"""
        COPY (
            SELECT
                geohash,
                AVG(lat)              AS lat,
                AVG(lon)              AS lon,
                hour,
                day_of_week,
                year,
                month,
                SUM(vehicle_count)    AS total_vehicles,
                AVG(avg_speed)        AS avg_speed,
                COUNT(*)              AS record_count
            FROM '{pq}'
            GROUP BY geohash, hour, day_of_week, year, month
            ORDER BY geohash, year, month, hour
        ) TO '{geo_hour_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"  ✅ summary_geo_hourly.parquet ({time.time()-t:.1f}s)")
    
    # 4. Aylık genel istatistikler
    print("  ⏳ Aylık özet oluşturuluyor...")
    t = time.time()
    monthly_path = os.path.join(DATA_DIR, "summary_monthly.parquet").replace("\\", "/")
    db.execute(f"""
        COPY (
            SELECT
                year, month,
                SUM(vehicle_count)    AS total_vehicles,
                AVG(avg_speed)        AS avg_speed,
                AVG(min_speed)        AS avg_min_speed,
                AVG(max_speed)        AS avg_max_speed,
                COUNT(*)              AS record_count,
                COUNT(DISTINCT geohash)  AS active_locations,
                COUNT(DISTINCT CAST(date_time AS DATE)) AS active_days
            FROM '{pq}'
            GROUP BY year, month
            ORDER BY year, month
        ) TO '{monthly_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"  ✅ summary_monthly.parquet ({time.time()-t:.1f}s)")
    
    db.close()


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("🔧 2025 Yapay Veri Üretici (DuckDB Optimized)")
    print("=" * 60)
    
    # 1. Ana dosyaya 2025 Şubat-Aralık verilerini ekle
    generate_traffic_all()
    
    # 2. Özet dosyaları baştan oluştur
    generate_summaries_from_all()
    
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"✅ Tüm yapay veriler ve özetler başarıyla üretildi! ({elapsed_total:.1f} saniye)")
    print("=" * 60)
