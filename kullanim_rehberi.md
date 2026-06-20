# İstanbul Trafik Analiz ve Tahmin Platformu - Kullanım Rehberi

Bu rehber, projenin kurulumunu, sistem gereksinimlerini ve farklı modüllerinin (Veri üretimi, model eğitimi, arayüz) nasıl kullanılacağını adım adım açıklamaktadır.

---

## 1. Sistem Gereksinimleri ve Kurulum

Sistemin tam performansla çalışabilmesi için güncel bir Python (3.10+) sürümü kurulu olmalıdır. Ayrıca YZ modellerinin (özellikle CNN-LSTM) hızlı eğitilebilmesi için NVIDIA tabanlı bir CUDA destekli ekran kartı tavsiye edilir.

### 1.1. Kütüphanelerin Kurulması
Projeyi çalıştırmadan önce terminal veya komut satırını açarak proje dizinine gidin ve aşağıdaki komutu çalıştırarak gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

---

## 2. Platformu Çalıştırma

Platform, web tabanlı bir Streamlit arayüzüne ve arka plan veri/model scriptlerine sahiptir. Günlük kullanım için sadece web arayüzünü başlatmanız yeterlidir.

### 2.1. Web Arayüzünü (Dashboard) Başlatma
Terminalde proje ana dizininde iken aşağıdaki komutu çalıştırın:
```bash
streamlit run app_web.py
```
Bu komut, yerel sunucunuzu başlatacak ve varsayılan web tarayıcınızda `http://localhost:8501` adresinde projeyi açacaktır.

---

## 3. Arayüz Bölümleri ve Kullanımı

Platform 6 ana sekmeden oluşmaktadır:

1. **📍 Yoğunluk Haritası:** 
   - Sol menüden seçtiğiniz yıl, ay ve saat bilgilerine göre İstanbul'un trafik yoğunluğunu 3D Isı Haritası (Heatmap) olarak görüntüler.
   - Harita stilini "Noktalar", "3D Sütunlar" veya "Isı Haritası" olarak değiştirebilirsiniz.

2. **📈 Zaman Analizi:**
   - Seçili ayın en yoğun günlerini, en yüksek hızlı/yavaş bölgelerini detaylı veri grafikleri (Plotly) eşliğinde analiz edebilirsiniz.

3. **📊 Yıl Karşılaştırma:**
   - Pandemi dönemi (2020) ile diğer yıllar (örneğin 2024 veya 2025) arasındaki araç sayıları ve ortalama hızları yan yana inceleyebilirsiniz.

4. **🔮 YZ Tahmin + Hava Durumu (Ana Modül):**
   - Geleceğe yönelik trafik tahmini yapmak için bir tarih seçin.
   - Sistem otomatik olarak o tarihin hava durumunu (Açık, Yağmurlu vb.) Open-Meteo API üzerinden çeker.
   - **"Tahmin Et"** butonuna bastığınızda seçtiğiniz YZ modeli (örneğin CNN-LSTM) devreye girer. Ayrıca seçtiğiniz rota üzerindeki trafik yükü ve CO2 emisyon salınımı otomatik olarak hesaplanır.

5. **🤖 YZ Modelleri Karşılaştırma:**
   - Random Forest, XGBoost, CNN-LSTM gibi modellerin MAE, RMSE ve Doğruluk (R²) metriklerini görsel tablolar üzerinden karşılaştırmanızı sağlar.

6. **📥 Veri Ekle / Karşılaştır:**
   - Sisteme dışarıdan `.csv` veya `.parquet` tabanlı yeni trafik analiz dosyaları yüklemenizi sağlar.

---

## 4. Gelişmiş İşlemler (Geliştiriciler İçin)

Proje, gelecekteki verileri (örneğin eksik 2025 aylarını) otomatik genişletebilecek ve yeni modeller eğitebilecek altyapıya sahiptir.

### 4.1. Sentetik Veri Genişletme
Mevcut 100 milyon satırlık veriye yeni sentetik veriler (geohash, mevsimsel ve trigonometrik gürültü eklenerek) üretmek isterseniz:
```bash
python generate_synthetic_data.py
```
*Not: Bu işlem DuckDB kullandığı için saniyeler içinde tamamlanır ve `data_parquet` klasöründeki dosyaları günceller.*

### 4.2. YZ Modellerini Yeniden Eğitme
Eğer verilere yeni kayıtlar eklenmişse modelleri tekrar eğitmeniz gerekir. Aşağıdaki komut, tüm modelleri yeniden çalıştırarak güncel veriler üzerinden eğitir ve `models/` klasörüne kaydeder:
```bash
python train_models.py
```
*Not: Bu işlem esnasında PyTorch tabanlı CNN-LSTM modeli CUDA (GPU) kullanarak eğitimi hızlandıracaktır.*
