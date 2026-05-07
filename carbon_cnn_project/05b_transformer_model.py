import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ============================================================
# Layer 5b : Transformer Encoder Model Building & Training
# ============================================================
# Input  : X_train.npy, y_train.npy, X_test.npy, y_test.npy
# Output : best_transformer_model.keras, final_transformer_model.keras,
#          transformer_training_history.png
# ============================================================

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


def positional_encoding(seq_len, d_model):
    """Generate sinusoidal positional encoding matrix."""
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)


def build_transformer_model(input_shape):
    """
    Simple Transformer Encoder for time-series regression.

    Architecture:
        Input -> Dense projection -> + Positional Encoding ->
        MultiHeadAttention (2 heads) + Residual + LayerNorm ->
        FeedForward (Dense 64 -> Dense d_model) + Residual + LayerNorm ->
        GlobalAveragePooling1D -> Dense(64) -> Dropout -> Dense(32) -> Dropout -> Dense(1)
    """
    seq_len, n_features = input_shape
    d_model = 64

    inputs = Input(shape=input_shape)

    # Project features to d_model dimensions
    x = Dense(d_model)(inputs)

    # Add positional encoding
    pos_enc = positional_encoding(seq_len, d_model)
    x = x + pos_enc

    # --- Transformer Encoder Block ---
    # Multi-Head Self-Attention
    attn_output = MultiHeadAttention(
        num_heads=2, key_dim=32, dropout=0.1
    )(x, x)
    x = Add()([x, attn_output])  # Residual connection
    x = LayerNormalization()(x)

    # Feed-Forward Network
    ff_output = Dense(64, activation='relu')(x)
    ff_output = Dense(d_model)(ff_output)
    x = Add()([x, ff_output])  # Residual connection
    x = LayerNormalization()(x)

    # --- Output Head ---
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_model(data_dir):
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

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

    input_shape = X_train.shape[1:]  # (7, 6)
    print(f"\nDynamic input shape: {input_shape}")

    # ----------------------------------------------------------
    # 2. Build model
    # ----------------------------------------------------------
    model = build_transformer_model(input_shape)
    model.summary()

    # ----------------------------------------------------------
    # 3. Compile
    # ----------------------------------------------------------
    model.compile(
        optimizer='adam',
        loss='huber',
        metrics=['mae']
    )

    # ----------------------------------------------------------
    # 4. Callbacks
    # ----------------------------------------------------------
    checkpoint_path = os.path.join(data_dir, "best_transformer_model.keras")

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
    print(f"\nStarting Transformer training for up to {EPOCHS} epochs...")
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
    print("\nInference test - Predicting first 5 test samples:")
    sample_preds = model.predict(X_test[:5], verbose=0)
    for i, (pred, actual) in enumerate(zip(sample_preds.flatten(), y_test[:5])):
        print(f"  Sample {i+1}: Predicted={pred:.4f}  |  Actual={actual:.4f}")

    # ----------------------------------------------------------
    # 8. Save final model
    # ----------------------------------------------------------
    final_model_path = os.path.join(data_dir, "final_transformer_model.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best checkpoint saved to: {checkpoint_path}")

    # ----------------------------------------------------------
    # 9. Plot training history
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Layer 5b: Transformer Training History', fontsize=14)

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss (Huber)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    history_path = os.path.join(data_dir, "transformer_training_history.png")
    plt.savefig(history_path, dpi=150)
    print(f"Training history plot saved to: {history_path}")

    print("\n" + "=" * 60)
    print("Layer 5b Complete!")
    print("=" * 60)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, "X_train.npy")

    if os.path.exists(train_file):
        train_model(current_dir)
    else:
        print(f"Error: Could not find {train_file}. Please run Layer 4 first.")
