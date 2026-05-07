import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ============================================================
# Layer 5 : CNN Model Building & Training
# ============================================================
# Input  : X_train.npy, y_train.npy, X_test.npy, y_test.npy
# Output : best_cnn_model.keras, final_model.keras,
#          training_history.png
# ============================================================

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


def build_model(input_shape):
    """
    Build a Conv1D regression model.
    Architecture:
        Conv1D(64,k=3) → BatchNorm → Conv1D(128,k=2) → BatchNorm →
        Flatten → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Dense(1)

    MaxPooling1D removed: 7 time steps is too short to afford spatial compression.
    padding='same' preserves sequence length throughout both Conv layers.
    Two Conv1D layers: first detects 3-day trends, second composes them into longer patterns.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               input_shape=input_shape),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # Single output for regression
    ])
    return model


def train_model(data_dir):
    np.random.seed(RANDOM_SEED)

    # ----------------------------------------------------------
    # 1. Load preprocessed data
    # ----------------------------------------------------------
    print("Loading preprocessed numpy arrays...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    print(f"  X_train : {X_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_test  : {y_test.shape}")

    # Dynamic input shape — adapts if window size or features change
    input_shape = X_train.shape[1:]   # (time_steps, features)
    print(f"\nDynamic input shape: {input_shape}")

    # ----------------------------------------------------------
    # 2. Build model
    # ----------------------------------------------------------
    model = build_model(input_shape)

    # Print full architecture table (great for report screenshots)
    model.summary()

    # ----------------------------------------------------------
    # 3. Compile
    # ----------------------------------------------------------
    model.compile(
        optimizer='adam',
        loss='huber',           # Robust to outliers
        metrics=['mae']         # Mean Absolute Error
    )

    # ----------------------------------------------------------
    # 4. Callbacks
    # ----------------------------------------------------------
    checkpoint_path = os.path.join(data_dir, "best_cnn_model.keras")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ----------------------------------------------------------
    # 5. Train
    # ----------------------------------------------------------
    print(f"\nStarting training for up to {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE} | Validation split: {VALIDATION_SPLIT}")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    # ----------------------------------------------------------
    # 6. Evaluate on test set
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating on held-out TEST set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss (Huber) : {test_loss:.6f}")
    print(f"  Test MAE          : {test_mae:.6f}")

    # ----------------------------------------------------------
    # 7. Quick inference test
    # ----------------------------------------------------------
    print("\nInference test — Predicting first 5 test samples:")
    sample_preds = model.predict(X_test[:5], verbose=0)
    for i, (pred, actual) in enumerate(zip(sample_preds.flatten(), y_test[:5])):
        print(f"  Sample {i+1}: Predicted={pred:.4f}  |  Actual={actual:.4f}")

    # ----------------------------------------------------------
    # 8. Save final model
    # ----------------------------------------------------------
    final_model_path = os.path.join(data_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best checkpoint saved to: {checkpoint_path}")

    # ----------------------------------------------------------
    # 9. Plot training history
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Layer 5: CNN Training History', fontsize=14)

    # Loss curve
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss (Huber)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE curve
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    history_path = os.path.join(data_dir, "training_history.png")
    plt.savefig(history_path, dpi=150)
    print(f"Training history plot saved to: {history_path}")

    print("\n" + "=" * 60)
    print("Layer 5 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, "X_train.npy")

    if os.path.exists(train_file):
        train_model(current_dir)
    else:
        print(f"Error: Could not find {train_file}. Please run Layer 4 first.")
