# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end ML pipeline that trains a 1D CNN to predict daily carbon emissions from 7-day rolling windows of environmental/industrial sensor data. Built for an academic thesis (SRM University).

## Installation

```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib shap streamlit plotly fastapi uvicorn requests joblib
```

Requires Python 3.9+.

## Build and validation
**Very Important: After every code change, validate the code by running the pipeline.**

## Running the Pipeline

Scripts must be executed in numbered order — each layer produces artifacts consumed by the next:

```bash
python 02_data_synthesis.py      # Generates final_df.csv from city_day.csv
python 03_eda_visualization.py   # Outputs eda_results.png
python 04_data_preprocessing.py  # Outputs X_train.npy, X_test.npy, y_train.npy, y_test.npy, *.save scalers
python 05_cnn_model.py           # Trains CNN; outputs best_cnn_model.keras, final_model.keras
python 06_model_evaluation.py    # Outputs metrics.txt, evaluation_results.png
python 08_explainable_ai.py      # Outputs shap_*.png, shap_values.csv
python 10_baseline_comparison.py # Outputs baseline_metrics.json
```

## Running the Apps

```bash
# Streamlit dashboard (5-tab interactive UI)
python -m streamlit run 07_streamlit_app.py

# FastAPI REST server (Swagger docs at http://localhost:8000/docs)
python -m uvicorn 09_fastapi_server:app --reload

# Test the API (run after server is up)
python 09_test_api.py
```

## Architecture

### Data Flow

`city_day.csv` -> `02_data_synthesis.py` -> `final_df.csv` -> `04_data_preprocessing.py` -> `.npy arrays` -> `05_cnn_model.py` -> `.keras models`

### CNN Input Shape

The model expects 3D tensors of shape `(samples, 7, 6)`:
- 7 = `WINDOW_SIZE` (days of history)
- 6 = feature count: `temperature`, `humidity`, `wind_speed`, `industry_level`, `vehicle_count`, `energy_usage`
- Target: `carbon_emission` (continuous, regression)

### Preprocessing Notes (`04_data_preprocessing.py`)

- Only the city with the most records is used for training (single-city model).
- Features and target are independently scaled with `MinMaxScaler` to `[0, 1]`.
- Scalers are persisted as `feature_scaler.save` and `target_scaler.save` (joblib) and must be loaded for any inference to inverse-transform predictions back to original units.
- Split is **chronological** (no shuffle) at 80/20.

### CNN Architecture (`05_cnn_model.py`)

```
Conv1D(64, k=3, relu, padding='same') -> BatchNorm ->
Conv1D(64, k=2, relu, padding='same') -> BatchNorm ->
Flatten -> Dense(64, relu) -> Dropout(0.3) ->
Dense(32, relu) -> Dropout(0.2) -> Dense(1, linear)
```

Compiled with Adam optimizer and Huber loss. Trained with `EarlyStopping(patience=10)`, `ReduceLROnPlateau`, and `ModelCheckpoint`. No `MaxPooling1D` — the 7-step sequence is too short for spatial compression.

### FastAPI Inference (`09_fastapi_server.py`)

The `/predict` endpoint receives a single data point (not a sequence). It tiles the single input 7 times to form a `(1, 7, 6)` tensor — this is intentional but means the API does not use true historical context.

Models and scalers are loaded once at startup via the lifespan context manager and cached in `ml_components` dict.

### Key Artifacts

| File | Purpose |
|------|---------|
| `final_df.csv` | Synthesized training dataset |
| `feature_scaler.save` / `target_scaler.save` | MinMaxScaler state; required for all inference |
| `best_cnn_model.keras` | Best checkpoint (lowest val_loss); used by API |
| `final_model.keras` | Final epoch model |
| `X_train/test.npy`, `y_train/test.npy` | Preprocessed training arrays |
