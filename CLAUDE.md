# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end ML pipeline for daily carbon emission forecasting using **Transformer Encoder** (primary) and **1D CNN** (baseline) models on 7-day rolling windows of environmental/industrial sensor data. Academic thesis project (SRM University, authors: Rohit Choudhary, Harsh Arora, Dimpesh Ramchandani; guide: Dr. V. Prasanna).

All source code lives in `carbon_cnn_project/`. The `.venv/` directory is the Python virtual environment.

## Environment Setup

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Install dependencies
pip install tensorflow keras numpy pandas scikit-learn matplotlib shap streamlit plotly fastapi uvicorn requests joblib python-docx
```

Requires Python 3.9+. Note: the D: drive venv and C:\Users\ASUS\Downloads venv are separate — always install packages using `.venv/Scripts/python -m pip install`.

## Running the Pipeline

Scripts must run in numbered order — each produces artifacts consumed by the next. Run from the project root (`D:\car racing`):

```bash
cd carbon_cnn_project
python 02_data_synthesis.py       # city_day.csv -> final_df.csv
python 03_eda_visualization.py    # -> eda_results.png
python 04_data_preprocessing.py   # -> X_train/test.npy, y_train/test.npy, *.save scalers
python 05_cnn_model.py            # -> best_cnn_model.keras, final_model.keras, training_history.png
python 05b_transformer_model.py   # -> best_transformer_model.keras, transformer_training_history.png
python 06_model_evaluation.py     # -> metrics.txt, evaluation_results.png (dual CNN + Transformer)
python 08_explainable_ai.py       # -> shap_*.png, shap_values.csv (runs on Transformer, fallback CNN)
python 10_baseline_comparison.py  # -> baseline_metrics.json (6 models compared)
```

After any code change, re-run the affected layer and all downstream layers to validate.

## Running the Apps

```bash
# Streamlit dashboard (5-tab interactive UI)
python -m streamlit run carbon_cnn_project/07_streamlit_app.py

# FastAPI REST server (Swagger UI at http://localhost:8000/docs)
python -m uvicorn carbon_cnn_project.09_fastapi_server:app --reload

# API integration test (server must be running first)
python carbon_cnn_project/09_test_api.py
```

## Architecture

### Data Flow

```
city_day.csv
  -> 02_data_synthesis.py      -> final_df.csv
  -> 04_data_preprocessing.py  -> X_train/test.npy, y_train/test.npy, scalers
  -> 05_cnn_model.py           -> best_cnn_model.keras
  -> 05b_transformer_model.py  -> best_transformer_model.keras
  -> 06_model_evaluation.py    (evaluates both models side-by-side)
  -> 08_explainable_ai.py      (SHAP on Transformer)
  -> 10_baseline_comparison.py (6-model comparison)
```

### Input Shape

Both CNN and Transformer expect `(samples, 7, 6)` tensors:
- `7` = `WINDOW_SIZE` (days of history)
- `6` = features: `temperature`, `humidity`, `wind_speed`, `industry_level`, `vehicle_count`, `energy_usage`
- Target: `carbon_emission` (continuous regression)

### CNN Architecture (`05_cnn_model.py`)

```
Conv1D(64, k=3, relu, padding='same') -> BatchNorm ->
Conv1D(64, k=2, relu, padding='same') -> BatchNorm ->
Flatten -> Dense(64, relu) -> Dropout(0.3) ->
Dense(32, relu) -> Dropout(0.2) -> Dense(1, linear)
```

Compiled with Adam + Huber loss. Callbacks: `EarlyStopping(patience=10)`, `ReduceLROnPlateau`, `ModelCheckpoint`. No `MaxPooling1D` — the 7-step sequence is too short for spatial compression.

### Transformer Architecture (`05b_transformer_model.py`)

```
Input -> Dense(64) projection -> + Sinusoidal Positional Encoding ->
MultiHeadAttention(2 heads, key_dim=32) + Residual + LayerNorm ->
FeedForward(Dense 64 relu -> Dense 64) + Residual + LayerNorm ->
GlobalAveragePooling1D -> Dense(64, relu) -> Dropout(0.3) ->
Dense(32, relu) -> Dropout(0.2) -> Dense(1, linear)
```

Same training setup as CNN (Adam, Huber, same callbacks). Converges in ~13 epochs vs CNN's ~32.

### Preprocessing (`04_data_preprocessing.py`)

- Only the city with the most records is used (single-city model).
- Features and target are independently scaled with `MinMaxScaler` to `[0, 1]`.
- Scalers saved as `feature_scaler.save` / `target_scaler.save` (joblib) — required for all inference.
- Split is **chronological** (no shuffle) at 80/20.

### SHAP Explainability (`08_explainable_ai.py`)

- Loads **Transformer** model by default (falls back to CNN if not found).
- Uses `shap.GradientExplainer` with 100 background samples.
- Generates: `shap_summary.png`, `shap_bar.png`, `feature_importance.txt`, `shap_values.csv`, `shap_force_plot.png`, `explainability_report.txt`.

### FastAPI Inference (`09_fastapi_server.py`)

- Loads **Transformer** as primary model, CNN as fallback.
- `/predict` endpoint tiles a single data point 7 times to form `(1, 7, 6)` — does not use true historical context.
- Models and scalers loaded once at startup via lifespan context manager into `ml_components` dict.
- **Note:** No emoji characters in print statements (Windows cp1252 encoding crashes).

### Streamlit Dashboard (`07_streamlit_app.py`)

5-tab interactive UI:
- **Tab 1 — Live Prediction:** Sliders for 6 features, real-time inference, gauge chart, risk assessment
- **Tab 2 — Training Evaluation:** Interactive Plotly charts (time-series, scatter, residual histograms, error over time)
- **Tab 3 — About Model:** Transformer vs CNN metrics, 6-model comparison bars, training history, summary card
- **Tab 4 — Explainable AI:** SHAP waterfall, impact cards, radar chart, AI-generated insights, global analysis, force plot
- **Tab 5 — 7-Day Forecast:** Rolling forecast with daily breakdown

Model selector in sidebar: users can switch between Transformer and CNN. SHAP runs on whichever model is active.

### Baseline Comparison (`10_baseline_comparison.py`)

Compares 6 models: Simple Average, Linear Regression, Random Forest, LSTM, CNN, Transformer. Results saved to `baseline_metrics.json`.

### Key Artifacts

| File | Purpose |
|------|---------|
| `final_df.csv` | Synthesized training dataset |
| `feature_scaler.save` / `target_scaler.save` | MinMaxScaler state; required for all inference |
| `best_cnn_model.keras` | CNN best checkpoint |
| `best_transformer_model.keras` | Transformer best checkpoint (primary model) |
| `final_model.keras` / `final_transformer_model.keras` | Final epoch models |
| `X_train/test.npy`, `y_train/test.npy` | Preprocessed training arrays |
| `metrics.txt` | CNN + Transformer evaluation metrics |
| `baseline_metrics.json` | All 6 models' metrics |
| `shap_summary.png`, `shap_bar.png` | Global SHAP plots (Transformer-based) |
| `feature_importance.txt` | Ranked feature importance from SHAP |
| `PROJECT_DOCUMENTATION.md` | Full project documentation |

### Known Issues

- Windows cp1252 encoding: avoid emoji characters in any print statements or file writes. Use plain text alternatives.
- The D: drive venv and C: drive venv are separate environments. Always verify packages are installed in the correct venv.
