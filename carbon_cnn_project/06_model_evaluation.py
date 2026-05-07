import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# ============================================================
# Layer 6 : Model Evaluation & Plotting (CNN + Transformer)
# ============================================================
# Input  : X_test.npy, y_test.npy, best_cnn_model.keras,
#          best_transformer_model.keras, target_scaler.save
# Output : evaluation_results.png, metrics.txt
# ============================================================


def evaluate_single_model(model, X_test, y_test, target_scaler, model_name):
    """Evaluate a single model and return real-unit predictions + metrics."""
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_true = target_scaler.inverse_transform(y_pred_scaled).flatten()

    mae = mean_absolute_error(y_test_true, y_pred_true)
    rmse = float(np.sqrt(mean_squared_error(y_test_true, y_pred_true)))
    r2 = r2_score(y_test_true, y_pred_true)
    epsilon = 1e-10
    mape = float(np.mean(np.abs((y_test_true - y_pred_true) / (y_test_true + epsilon))) * 100)

    print(f"\n{'=' * 50}")
    print(f"{model_name} METRICS (Real Units)")
    print(f"{'=' * 50}")
    print(f"  MAE   : {mae:.2f} units")
    print(f"  RMSE  : {rmse:.2f} units")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"  R^2   : {r2:.4f}")
    print(f"{'=' * 50}")

    return y_test_true, y_pred_true, {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def evaluate_model(data_dir):
    print("Layer 6: Loading Assets...")

    # ----------------------------------------------------------
    # 1. Load Data & Scalers
    # ----------------------------------------------------------
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    target_scaler = joblib.load(os.path.join(data_dir, "target_scaler.save"))
    print("Scalers loaded successfully.")

    # ----------------------------------------------------------
    # 2. Load & Evaluate Models
    # ----------------------------------------------------------
    models_to_eval = {}

    cnn_path = os.path.join(data_dir, "best_cnn_model.keras")
    if os.path.exists(cnn_path):
        models_to_eval["CNN"] = load_model(cnn_path)
        print("\nCNN model loaded.")

    transformer_path = os.path.join(data_dir, "best_transformer_model.keras")
    if os.path.exists(transformer_path):
        models_to_eval["Transformer"] = load_model(transformer_path)
        print("Transformer model loaded.")

    all_metrics = {}
    all_predictions = {}

    for name, model in models_to_eval.items():
        y_true, y_pred, metrics = evaluate_single_model(model, X_test, y_test, target_scaler, name)
        all_metrics[name] = metrics
        all_predictions[name] = (y_true, y_pred)

    # ----------------------------------------------------------
    # 3. Save Metrics
    # ----------------------------------------------------------
    metrics_path = os.path.join(data_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for name, metrics in all_metrics.items():
            f.write(f"--- {name} ---\n")
            f.write(f"MAE: {metrics['MAE']:.2f}\n")
            f.write(f"RMSE: {metrics['RMSE']:.2f}\n")
            f.write(f"MAPE: {metrics['MAPE']:.2f}\n")
            f.write(f"R2: {metrics['R2']:.4f}\n\n")
    print(f"\nMetrics saved to {metrics_path}")

    # ----------------------------------------------------------
    # 4. Generate Comparison Plots
    # ----------------------------------------------------------
    print("\nGenerating evaluation plots...")
    n_models = len(all_predictions)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 6 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('Layer 6: Model Evaluation (Real Units)', fontsize=16, y=1.01)

    plot_limit = min(150, len(y_test))

    for i, (name, (y_true, y_pred)) in enumerate(all_predictions.items()):
        # Time-series overlay
        axes[i, 0].plot(y_true[:plot_limit], label='Actual', color='blue', linewidth=2)
        axes[i, 0].plot(y_pred[:plot_limit], label='Predicted', color='orange', linestyle='dashed', linewidth=2)
        axes[i, 0].set_title(f'{name}: Time-Series Prediction (First {plot_limit} Days)')
        axes[i, 0].set_xlabel('Time Steps (Days)')
        axes[i, 0].set_ylabel('Carbon Emission')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # Scatter plot
        axes[i, 1].scatter(y_true, y_pred, alpha=0.5, color='purple')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[i, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        r2 = all_metrics[name]['R2']
        axes[i, 1].set_title(f'{name}: Predicted vs Actual (R2={r2:.4f})')
        axes[i, 1].set_xlabel('Actual Emissions')
        axes[i, 1].set_ylabel('Predicted Emissions')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(data_dir, "evaluation_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {output_path}")
    print("\nLayer 6 Complete!")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_model(current_dir)
