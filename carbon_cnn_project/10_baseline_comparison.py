import numpy as np
import os
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# Layer 10 : Full Baseline Comparison
# ============================================================
# Models: Simple Average, Linear Regression, Random Forest,
#         LSTM, CNN, Transformer
# ============================================================


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    epsilon = 1e-10
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2), "R2": round(r2, 4)}


def run_baselines():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading datasets...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    target_scaler = joblib.load(os.path.join(data_dir, "target_scaler.save"))

    # Inverse transform ground truth once
    y_test_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    results = {}

    # ==========================================
    # 1. Simple Average
    # ==========================================
    print("\n[1/6] Simple Average...")
    y_pred_avg = np.full_like(y_test, fill_value=np.mean(y_train))
    y_pred_avg_true = target_scaler.inverse_transform(y_pred_avg.reshape(-1, 1)).flatten()
    results["Simple Average"] = calculate_metrics(y_test_true, y_pred_avg_true)
    print(f"  MAE: {results['Simple Average']['MAE']}")

    # ==========================================
    # 2. Linear Regression
    # ==========================================
    print("[2/6] Linear Regression...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    lr = LinearRegression()
    lr.fit(X_train_flat, y_train.ravel())
    y_pred_lr = lr.predict(X_test_flat).reshape(-1, 1)
    y_pred_lr_true = target_scaler.inverse_transform(y_pred_lr).flatten()
    results["Linear Regression"] = calculate_metrics(y_test_true, y_pred_lr_true)
    print(f"  MAE: {results['Linear Regression']['MAE']}")

    # ==========================================
    # 3. Random Forest
    # ==========================================
    print("[3/6] Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train.ravel())
    y_pred_rf = rf.predict(X_test_flat).reshape(-1, 1)
    y_pred_rf_true = target_scaler.inverse_transform(y_pred_rf).flatten()
    results["Random Forest"] = calculate_metrics(y_test_true, y_pred_rf_true)
    print(f"  MAE: {results['Random Forest']['MAE']}")

    # ==========================================
    # 4. LSTM
    # ==========================================
    print("[4/6] LSTM (training)...")
    np.random.seed(42)
    tf.random.set_seed(42)

    lstm_model = Sequential([
        LSTM(64, input_shape=X_train.shape[1:]),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    lstm_model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    lstm_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    y_pred_lstm = lstm_model.predict(X_test, verbose=0)
    y_pred_lstm_true = target_scaler.inverse_transform(y_pred_lstm).flatten()
    results["LSTM"] = calculate_metrics(y_test_true, y_pred_lstm_true)
    print(f"  MAE: {results['LSTM']['MAE']}")

    # ==========================================
    # 5. CNN (load pre-trained)
    # ==========================================
    print("[5/6] CNN (pre-trained)...")
    cnn_path = os.path.join(data_dir, "best_cnn_model.keras")
    if os.path.exists(cnn_path):
        cnn_model = load_model(cnn_path)
        y_pred_cnn = cnn_model.predict(X_test, verbose=0)
        y_pred_cnn_true = target_scaler.inverse_transform(y_pred_cnn).flatten()
        results["CNN"] = calculate_metrics(y_test_true, y_pred_cnn_true)
        print(f"  MAE: {results['CNN']['MAE']}")
    else:
        print("  SKIPPED: best_cnn_model.keras not found")

    # ==========================================
    # 6. Transformer (load pre-trained)
    # ==========================================
    print("[6/6] Transformer (pre-trained)...")
    transformer_path = os.path.join(data_dir, "best_transformer_model.keras")
    if os.path.exists(transformer_path):
        transformer_model = load_model(transformer_path)
        y_pred_tf = transformer_model.predict(X_test, verbose=0)
        y_pred_tf_true = target_scaler.inverse_transform(y_pred_tf).flatten()
        results["Transformer"] = calculate_metrics(y_test_true, y_pred_tf_true)
        print(f"  MAE: {results['Transformer']['MAE']}")
    else:
        print("  SKIPPED: best_transformer_model.keras not found")

    # ==========================================
    # Summary Table
    # ==========================================
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R2':>8}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['MAE']:>8.2f} {metrics['RMSE']:>8.2f} {metrics['MAPE']:>7.2f}% {metrics['R2']:>8.4f}")
    print("=" * 70)

    # ==========================================
    # Save Results
    # ==========================================
    out_path = os.path.join(data_dir, "baseline_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAll metrics saved to {out_path}")

    # ==========================================
    # Comparison Bar Chart
    # ==========================================
    model_names = list(results.keys())
    mae_values = [results[m]["MAE"] for m in model_names]
    r2_values = [results[m]["R2"] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison: Baselines vs Deep Learning", fontsize=14)

    colors = ['#95a5a6', '#95a5a6', '#3498db', '#e67e22', '#e74c3c', '#2ecc71']
    colors = colors[:len(model_names)]

    axes[0].barh(model_names, mae_values, color=colors)
    axes[0].set_xlabel("MAE (lower is better)")
    axes[0].set_title("Mean Absolute Error")
    for i, v in enumerate(mae_values):
        axes[0].text(v + 10, i, f"{v:.1f}", va='center')

    axes[1].barh(model_names, r2_values, color=colors)
    axes[1].set_xlabel("R2 (higher is better)")
    axes[1].set_title("R-Squared Score")
    axes[1].set_xlim(0, 1.1)
    for i, v in enumerate(r2_values):
        axes[1].text(v + 0.01, i, f"{v:.4f}", va='center')

    plt.tight_layout()
    chart_path = os.path.join(data_dir, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Comparison chart saved to {chart_path}")


if __name__ == "__main__":
    run_baselines()
