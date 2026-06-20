"""
train_models.py — Gerçek Makine Öğrenmesi Model Eğitimi
=======================================================
İBB trafik veri seti (2020-2025) üzerinde 4 farklı ML modeli eğitir:
  1. Random Forest
  2. XGBoost
  3. Gradient Boosting
  4. Ridge Regression

Modeller 'models/' klasörüne .pkl olarak kaydedilir.
app_web.py bu dosyaları yükleyerek gerçek tahminler yapar.
"""

import duckdb
import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# ML Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("!! xgboost bulunamadi. XGBoost modeli atlanacak.")
    print("   Kurmak icin: pip install xgboost")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_parquet")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

from shared_models import TrafficCNNLSTMModel


class TrafficSequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len + 1
        
    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]


def load_training_data():
    """summary_hourly.parquet'ten egitim verisini yukler ve feature engineering yapar."""
    print("\n[1/5] Veri yukleniyor...")
    p = os.path.join(DATA_DIR, "summary_hourly.parquet").replace("\\", "/")
    db = duckdb.connect()
    df = db.execute(f"SELECT * FROM '{p}' ORDER BY hour_ts").df()
    db.close()
    print(f"  Toplam satir: {len(df)}")

    # Feature Engineering
    print("[2/5] Feature engineering yapiliyor...")
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])

    # Temel ozellikler
    features = pd.DataFrame()
    features["hour"] = df["hour"].astype(float)
    features["day_of_week"] = df["day_of_week"].astype(float)
    features["month"] = df["month"].astype(float)
    features["year"] = df["year"].astype(float)

    # Siklik ozellikler (saat ve gun dongusal yapilar)
    features["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    features["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    features["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    features["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Hafta sonu / hafta ici
    features["is_weekend"] = (df["day_of_week"] >= 5).astype(float)

    # Mesai saati
    features["is_rush_morning"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(float)
    features["is_rush_evening"] = ((df["hour"] >= 17) & (df["hour"] <= 19)).astype(float)
    features["is_night"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(float)

    # Mevsim
    features["is_summer"] = df["month"].isin([6, 7, 8]).astype(float)
    features["is_winter"] = df["month"].isin([12, 1, 2]).astype(float)

    # Yil trendi (normalize)
    features["year_norm"] = (df["year"] - 2020) / 5.0

    # Hedef degiskenler
    targets = pd.DataFrame()
    targets["total_vehicles"] = df["total_vehicles"]
    targets["city_avg_speed"] = df["city_avg_speed"]

    # NaN temizle
    mask = features.notna().all(axis=1) & targets.notna().all(axis=1)
    features = features[mask].reset_index(drop=True)
    targets = targets[mask].reset_index(drop=True)

    # --- Outlier Kırpma (Percentile Clipping / Winsorization) ---
    print("[2.5/5] Aykırı Değerler (Outliers) %1 ve %99 dilimlere kırpılıyor...")
    for col in ["total_vehicles", "city_avg_speed"]:
        lower_bound = targets[col].quantile(0.01)
        upper_bound = targets[col].quantile(0.99)
        outliers_count = ((targets[col] < lower_bound) | (targets[col] > upper_bound)).sum()
        print(f"  {col}: {outliers_count} adet aykırı değer kırpılıyor ({lower_bound:.1f} - {upper_bound:.1f} aralığına)")
        targets[col] = targets[col].clip(lower=lower_bound, upper=upper_bound)

    print(f"  Feature sayisi: {features.shape[1]}")
    print(f"  Temiz satir sayisi: {len(features)}")
    print(f"  Feature listesi: {list(features.columns)}")

    return features, targets


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, scaler):
    """Tek bir modeli egitir, degerlendirir ve kaydeder."""
    print(f"\n  --- {name} ---")
    print(f"  Egitim basliyor... ({len(X_train)} satir)")

    start_time = datetime.now()
    model.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"  Egitim suresi: {train_time:.1f} saniye")

    # Tahmin
    y_pred = model.predict(X_test)
    if len(y_pred.shape) == 1:
        # Tek ciktili model icin
        y_pred = y_pred.reshape(-1, 1)

    results = {}
    target_names = ["total_vehicles", "city_avg_speed"]
    for i, target in enumerate(target_names):
        y_true_col = y_test.iloc[:, i].values
        y_pred_col = y_pred[:, i] if y_pred.shape[1] > i else y_pred[:, 0]

        mae = mean_absolute_error(y_true_col, y_pred_col)
        rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
        r2 = r2_score(y_true_col, y_pred_col)
        mape = np.mean(np.abs((y_true_col - y_pred_col) / np.maximum(y_true_col, 1))) * 100

        results[target] = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4), "mape": round(mape, 2)}
        print(f"  [{target}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")

    # Genel dogruluk (R2 ortalamasi)
    avg_r2 = np.mean([results[t]["r2"] for t in target_names])
    accuracy_pct = max(0, min(100, round(avg_r2 * 100, 1)))
    print(f"  => Genel Dogruluk: %{accuracy_pct}")

    # Modeli kaydet
    model_path = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Kaydedildi: {model_path}")

    return {
        "name": name,
        "accuracy": accuracy_pct,
        "train_time_sec": round(train_time, 1),
        "metrics": results,
        "model_path": model_path,
    }


def main():
    print("=" * 60)
    print("Gercek Makine Ogrenmesi Model Egitimi")
    print("Istanbul Trafik Veri Seti (2020-2025)")
    print("=" * 60)

    # Veri yukle
    X, y = load_training_data()

    # Train/Test bolunmesi
    print("\n[3/5] Train/Test bolunuyor (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Olceklendirme
    print("[4/5] Veriler olceklendiriliyor (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scaler'i kaydet
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler kaydedildi: {scaler_path}")

    # Feature listesini kaydet
    feature_path = os.path.join(MODEL_DIR, "features.json")
    with open(feature_path, "w") as f:
        json.dump(list(X.columns), f)
    print(f"  Feature listesi kaydedildi: {feature_path}")

    # Modelleri tanimla
    print("\n[5/5] Modeller egitiliyor...")
    models = {
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=500,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                min_samples_split=5,
                random_state=42,
            )
        ),
        "Ridge Regression": MultiOutputRegressor(
            Ridge(alpha=1.0)
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=500,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )
        )

    all_results = []
    for name, model in models.items():
        result = train_and_evaluate(name, model, X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        all_results.append(result)

    # ── 6) PyTorch CNN-LSTM Modeli Eğitimi ─────────────────────────────────
    print("\n[5b/5] PyTorch CNN-LSTM Modeli eğitiliyor...")
    t_start = datetime.now()
    
    # Hedefleri sinir ağı eğitimi için ölçeklendir
    y_scaled = y.copy()
    y_scaled["total_vehicles"] = y_scaled["total_vehicles"] / 100000.0
    y_scaled["city_avg_speed"] = y_scaled["city_avg_speed"] / 50.0
    
    # Tüm ölçeklendirilmiş veri üzerinden sekansları oluştur
    X_all_scaled = scaler.transform(X)
    lstm_dataset = TrafficSequenceDataset(X_all_scaled, y_scaled, seq_len=24)
    
    # Train/Test bölünmesi (sequence index bazında)
    indices = list(range(len(lstm_dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_subset = Subset(lstm_dataset, train_idx)
    test_subset = Subset(lstm_dataset, test_idx)
    
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=512, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  CNN-LSTM Eğitim Cihazı: {device}")
    
    model_cnn_lstm = TrafficCNNLSTMModel(input_dim=17, num_filters=32, kernel_size=3, hidden_dim=64, num_layers=2, output_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_cnn_lstm.parameters(), lr=0.001)
    
    epochs = 50
    model_cnn_lstm.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model_cnn_lstm(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)
        epoch_loss /= len(train_subset)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")
        
    train_time = (datetime.now() - t_start).total_seconds()
    print(f"  CNN-LSTM Eğitim süresi: {train_time:.1f} saniye")
    
    # Test verisi üzerinde değerlendirme
    model_cnn_lstm.eval()
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            pred = model_cnn_lstm(x_batch)
            y_preds.append(pred.cpu().numpy())
            y_trues.append(y_batch.numpy())
            
    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    
    # Hedefleri gerçek değerlerine geri ölçeklendir
    y_preds[:, 0] = y_preds[:, 0] * 100000.0
    y_preds[:, 1] = y_preds[:, 1] * 50.0
    y_trues[:, 0] = y_trues[:, 0] * 100000.0
    y_trues[:, 1] = y_trues[:, 1] * 50.0
    
    results_lstm = {}
    target_names = ["total_vehicles", "city_avg_speed"]
    for i, target in enumerate(target_names):
        y_true_col = y_trues[:, i]
        y_pred_col = y_preds[:, i]
        
        mae = float(mean_absolute_error(y_true_col, y_pred_col))
        rmse = float(np.sqrt(mean_squared_error(y_true_col, y_pred_col)))
        r2 = float(r2_score(y_true_col, y_pred_col))
        mape = float(np.mean(np.abs((y_true_col - y_pred_col) / np.maximum(y_true_col, 1))) * 100)
        
        results_lstm[target] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mape": round(mape, 2)
        }
        print(f"  [{target}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")
        
    avg_r2 = np.mean([results_lstm[t]["r2"] for t in target_names])
    accuracy_pct = max(0.0, min(100.0, round(float(avg_r2) * 100, 1)))
    print(f"  => CNN-LSTM Genel Doğruluk: %{accuracy_pct}")
    
    # Kaydet
    lstm_pth = os.path.join(MODEL_DIR, "lstm_model.pth")
    torch.save(model_cnn_lstm.state_dict(), lstm_pth)
    print(f"  CNN-LSTM model weights saved: {lstm_pth}")
    
    all_results.append({
        "name": "CNN-LSTM",
        "accuracy": accuracy_pct,
        "train_time_sec": round(train_time, 1),
        "metrics": results_lstm,
        "model_path": lstm_pth,
    })
 
    # Sonuclari kaydet
    results_path = os.path.join(MODEL_DIR, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Tum sonuclar kaydedildi: {results_path}")

    # Ozet tablo
    print("\n" + "=" * 60)
    print("EGITIM SONUCLARI OZETI")
    print("=" * 60)
    print(f"{'Model':<22} {'Dogruluk':>10} {'MAE(Arac)':>12} {'MAE(Hiz)':>10} {'Sure':>8}")
    print("-" * 62)
    for r in sorted(all_results, key=lambda x: -x["accuracy"]):
        mae_v = r["metrics"]["total_vehicles"]["mae"]
        mae_s = r["metrics"]["city_avg_speed"]["mae"]
        print(f"{r['name']:<22} %{r['accuracy']:>8} {mae_v:>12,.2f} {mae_s:>10.2f} {r['train_time_sec']:>6.1f}s")
    print("=" * 60)
    print("Tum modeller basariyla egitildi ve kaydedildi!")
    print(f"Model klasoru: {MODEL_DIR}")


if __name__ == "__main__":
    main()
