import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# Layer 4 : Data Preprocessing & Sequence Windowing for CNN
# ============================================================
# Input  : final_df.csv  (from Layer 2)
# Output : X_train.npy, y_train.npy, X_test.npy, y_test.npy
#          feature_scaler.save, target_scaler.save
# ============================================================

WINDOW_SIZE = 7          # Use past 7 days to predict the next day
TRAIN_RATIO = 0.8        # 80 % training, 20 % testing
RANDOM_SEED = 42

FEATURE_COLS = [
    'temperature', 'humidity', 'wind_speed',
    'industry_level', 'vehicle_count', 'energy_usage'
]
TARGET_COL = 'carbon_emission'


def create_sequences(features, target, window_size):
    """
    Slide a window across the time-series data.
    For each window of `window_size` rows of features,
    the label is the target value on the NEXT row.

    Returns
    -------
    X : np.ndarray, shape (samples, window_size, n_features)
    y : np.ndarray, shape (samples,)
    """
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)


def preprocess(input_path, output_dir):
    # ----------------------------------------------------------
    # 1. Load & basic cleanup
    # ----------------------------------------------------------
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['time_index'] = pd.to_datetime(df['time_index'])

    # We train one model per city for cleaner time-series.
    # Pick the city with the most data points as the primary city.
    city_counts = df['City'].value_counts()
    primary_city = city_counts.idxmax()
    print(f"Primary city selected: {primary_city} ({city_counts[primary_city]} records)")

    city_df = df[df['City'] == primary_city].sort_values('time_index').reset_index(drop=True)

    # --- Data Validation ---
    # Check for missing values only in the columns we will use
    relevant_cols = FEATURE_COLS + [TARGET_COL]
    missing_count = city_df[relevant_cols].isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} missing values found in relevant columns. Dropping affected rows...")
        city_df = city_df.dropna(subset=relevant_cols)
    else:
        print("No missing values detected in relevant columns. Data is clean.")

    # Print data types for verification
    print("\nData types:")
    print(city_df.dtypes)

    # Print feature count
    print(f"\nNumber of features: {len(FEATURE_COLS)}")

    # Assert valid window size
    assert WINDOW_SIZE > 1, "WINDOW_SIZE must be greater than 1"

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # ----------------------------------------------------------
    # 2. Feature & Target extraction
    # ----------------------------------------------------------
    features_raw = city_df[FEATURE_COLS].values.astype(np.float64)
    target_raw   = city_df[[TARGET_COL]].values.astype(np.float64)

    print(f"Feature matrix shape : {features_raw.shape}")
    print(f"Target vector shape  : {target_raw.shape}")

    # ----------------------------------------------------------
    # 3. Scaling (StandardScaler -> zero-mean, unit-variance)
    # ----------------------------------------------------------
    feature_scaler = StandardScaler()
    target_scaler  = StandardScaler()

    features_scaled = feature_scaler.fit_transform(features_raw)
    target_scaled   = target_scaler.fit_transform(target_raw).flatten()

    # Save scalers so we can inverse-transform predictions in Layer 6
    joblib.dump(feature_scaler, os.path.join(output_dir, "feature_scaler.save"))
    joblib.dump(target_scaler,  os.path.join(output_dir, "target_scaler.save"))
    print("Scalers saved (feature_scaler.save, target_scaler.save)")

    # ----------------------------------------------------------
    # 4. Sliding Window Sequences
    # ----------------------------------------------------------
    X, y = create_sequences(features_scaled, target_scaled, WINDOW_SIZE)
    print(f"\nAfter windowing (window_size={WINDOW_SIZE}):")
    print(f"  X shape : {X.shape}   ->  (samples, time_steps, features)")
    print(f"  y shape : {y.shape}   ->  (samples,)")

    # ----------------------------------------------------------
    # 5. Chronological Train / Test Split
    # ----------------------------------------------------------
    split_idx = int(len(X) * TRAIN_RATIO)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nChronological split at index {split_idx}:")
    print(f"  Train  ->  X: {X_train.shape}  y: {y_train.shape}")
    print(f"  Test   ->  X: {X_test.shape}  y: {y_test.shape}")

    # ----------------------------------------------------------
    # 6. Save arrays for Layer 5
    # ----------------------------------------------------------
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(output_dir, "y_test.npy"),  y_test)

    print(f"\nLayer 4 Complete! All arrays saved to: {output_dir}")
    print("Files created: X_train.npy, y_train.npy, X_test.npy, y_test.npy")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file  = os.path.join(current_dir, "final_df.csv")

    if os.path.exists(input_file):
        preprocess(input_file, current_dir)
    else:
        print(f"Error: Could not find {input_file}. Please run Layer 2 first.")
