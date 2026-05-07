import os
import numpy as np
import shap
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================
# Layer 8: Explainable AI - Phase 1 (Global SHAP Explanations)
# ============================================================

def generate_global_explanations(data_dir):
    print("Layer 8 [Phase 1]: Initializing SHAP Explainer...")

    # 1. Load Model (prefer Transformer, fallback to CNN)
    transformer_path = os.path.join(data_dir, "best_transformer_model.keras")
    cnn_path = os.path.join(data_dir, "best_cnn_model.keras")

    if os.path.exists(transformer_path):
        model_path = transformer_path
        model_name = "Transformer Encoder"
        print(f"Using Transformer model: {model_path}")
    elif os.path.exists(cnn_path):
        model_path = cnn_path
        model_name = "1D CNN"
        print(f"WARNING: Transformer not found. Falling back to CNN: {model_path}")
    else:
        print("Error: No trained model found (checked Transformer and CNN).")
        return

    model = load_model(model_path)
    
    # 2. Load Data for Background Sampling
    X_test_path = os.path.join(data_dir, "X_test.npy")
    X_test = np.load(X_test_path)
    
    # The feature labels (MUST match Layer 4 sequence)
    feature_names = ['Temperature', 'Humidity', 'Wind Speed', 'Industry Level', 'Vehicle Count', 'Energy Usage']

    # 3. Background Sampling
    # SHAP needs a background dataset to calculate baseline expected values.
    # For a 3D CNN input (samples, time_steps, features), we use the first 100 samples
    background = X_test[:100]
    
    print(f"Background sample shape: {background.shape}")
    
    # 4. Initialize GradientExplainer
    # GradientExplainer uses native TF2 GradientTape and is fully compatible with Keras
    explainer = shap.GradientExplainer(model, background)
    
    # Calculate SHAP values for the next 100 samples to generate global plots
    test_samples = X_test[100:min(200, len(X_test))]
    print(f"Calculating SHAP values for {len(test_samples)} test samples...")
    
    # shap_values represents the contribution of each feature to the output
    shap_values = explainer.shap_values(test_samples)

    # For a Conv1D model, shap_values is a list. For regression (1 output), it's shap_values[0]
    if isinstance(shap_values, list):
        shap_values_array = shap_values[0]
    else:
        shap_values_array = shap_values

    # Handle variable regression output shapes explicitly (e.g. from (samples, time_steps, features, 1))
    shap_values_array = np.squeeze(shap_values_array)

    # shap_values_array has shape (samples, time_steps, features)
    # To plot global feature importance, we often average or sum across the time_steps window
    # We will compute the mean absolute SHAP value across the 7 time steps for each feature
    shap_aggregated = np.mean(shap_values_array, axis=1) # Shape becomes (samples, features)
    
    # Similarly, we need the aggregated 2D input features for plotting the Summary
    test_samples_aggregated = np.mean(test_samples, axis=1)

    # 5. Generate Global Summary Plot
    print("Generating shap_summary.png...")
    plt.style.use('dark_background')
    plt.figure()
    shap.summary_plot(shap_aggregated, test_samples_aggregated, feature_names=feature_names, show=False)
    summary_path = os.path.join(data_dir, "shap_summary.png")
    plt.savefig(summary_path, bbox_inches='tight', dpi=150, facecolor='#0e1117')
    plt.close()

    # 6. Generate Global Bar Plot
    print("Generating shap_bar.png...")
    plt.figure()
    shap.summary_plot(shap_aggregated, test_samples_aggregated, feature_names=feature_names, plot_type="bar", show=False)
    bar_path = os.path.join(data_dir, "shap_bar.png")
    plt.savefig(bar_path, bbox_inches='tight', dpi=150, facecolor='#0e1117')
    plt.close()

    # 7. Compute and Save Feature Importance Rankings
    print("Generating feature_importance.txt...")
    # Calculate the mean absolute SHAP value for each feature
    mean_abs_shap = np.mean(np.abs(shap_aggregated), axis=0)
    
    # Sort features by importance
    importance_indices = np.argsort(mean_abs_shap)[::-1]
    
    feature_importance_path = os.path.join(data_dir, "feature_importance.txt")
    with open(feature_importance_path, "w", encoding="utf-8") as f:
        f.write("Top Features Affecting Carbon Emissions (Mean Absolute SHAP):\n")
        f.write("-" * 60 + "\n")
        for rank, idx in enumerate(importance_indices):
            f.write(f"{rank + 1}. {feature_names[idx]} ({mean_abs_shap[idx]:.4f})\n")
            
    # 8. Save raw SHAP values to CSV
    print("Generating shap_values.csv...")
    shap_csv_path = os.path.join(data_dir, "shap_values.csv")
    import pandas as pd
    shap_df = pd.DataFrame(shap_aggregated, columns=feature_names)
    shap_df.to_csv(shap_csv_path, index=False)

    # 9. Local Explanation (Force Plot for Sample 0)
    print("Generating shap_force_plot.png...")
    try:
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray) or isinstance(expected_value, list):
            expected_value = expected_value[0]
    except:
        expected_value = np.mean(model.predict(background, verbose=0))
        
    plt.figure()
    shap.force_plot(
        expected_value, 
        shap_aggregated[0], 
        test_samples_aggregated[0], 
        feature_names=feature_names, 
        matplotlib=True, 
        show=False
    )
    force_path = os.path.join(data_dir, "shap_force_plot.png")
    plt.savefig(force_path, bbox_inches='tight', dpi=150, facecolor='#0e1117')
    plt.close()

    # 10. Generate Explainability Report
    print("Generating explainability_report.txt...")
    report_path = os.path.join(data_dir, "explainability_report.txt")
    
    # Calculate dataset comparison (Mean vs Sample 0)
    # Using the raw unscaled sequence arrays
    pred_sample_0 = float(model.predict(np.expand_dims(test_samples[0], axis=0), verbose=0)[0][0])
    mean_dataset_pred = float(np.mean(model.predict(background, verbose=0)))
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== Explainable AI using SHAP ({model_name}) ===\n\n")
        f.write("1. Why needed:\nNeural networks are typically 'Black Boxes'. SHAP allows us to see exactly which features drove the carbon emission predictions up or down.\n\n")
        f.write(f"2. How used:\nWe used shap.GradientExplainer on the trained {model_name}, comparing the test samples against a background baseline of 100 training samples.\n\n")
        f.write("3. Results (Top 3 Drivers):\n")
        for i in range(min(3, len(importance_indices))):
            idx = importance_indices[i]
            f.write(f"  - {feature_names[idx]}\n")
        
        f.write("\n4. Interpretation:\n")
        top_feature = feature_names[importance_indices[0]]
        second_feature = feature_names[importance_indices[1]]
        f.write(f"The model prediction is mainly influenced by {top_feature} and {second_feature}.\n\n")
        
        f.write("5. Dataset Comparison:\n")
        f.write(f"  - Average Dataset Emission (Scaled baseline): {mean_dataset_pred:.4f}\n")
        f.write(f"  - Current Sample Emission (Scaled): {pred_sample_0:.4f}\n")

    print("Phase 2 Complete! Saved:")
    print(f" - {summary_path}")
    print(f" - {bar_path}")
    print(f" - {feature_importance_path}")
    print(f" - {shap_csv_path}")
    print(f" - {force_path}")
    print(f" - {report_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_global_explanations(current_dir)
