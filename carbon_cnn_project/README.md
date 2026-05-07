# 🏭 Carbon Emission Prediction using 1D CNN

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

## Project Overview
This repository contains a full-stack, end-to-end Machine Learning pipeline developed for an academic thesis. It leverages a **1D Convolutional Neural Network (Conv1D)** to analyze temporal sequences of environmental and structural data, successfully forecasting daily carbon emissions with precision. 

The project goes beyond classical predictive modeling by implementing a robust multi-layer pipeline: from data synthesis and preprocessing, right through to an enterprise-grade Streamlit interactive dashboard, Explainable AI (SHAP) integration, 7-day future weather forecasting simulations, and a production FastAPI REST server.

---

## 🚀 Features

- **Deep Temporal Extraction**: The CNN accurately models chronological atmospheric and behavioral shifts over 7-day rolling windows (`(1, 7, 6)` dimensional layout).
- **Explainable AI (XAI)**: Demystifies the "black box" neural network using dynamic SHAP Force Plots to prove exactly which features drove the prediction.
- **7-Day Dynamic Forecast Engine**: A mathematical simulation engine that perturbs inputs to mimic realistic environmental cycles (e.g., weekend industry drops, heatwaves) to map a week-ahead carbon emission trend.
- **Classical Baseline Validation**: Natively evaluates and proves the absolute superiority of the deep learning architecture mathematically against Naive Average and Linear Regression models.
- **Production REST API**: A fully-fledged FastAPI backend ready for IoT integration and high-speed JSON inference.
- **Examiner-Ready Dashboard**: A spectacular 5-tab hyper-polished Streamlit UI designed specifically for a live academic presentation, featuring real-time interactive sliders, performance metrics parsing, and structural architecture summaries.

---

## 📂 Project Pipeline Architecture (Layers 1 to 11)

The codebase is strictly layered chronologically for academic rigor:

- `01_data_collection.py` - Raw dataset parsing.
- `02_data_synthesis.py` - Generation of contextual features (Temp, Wind, Vehicles, Industry Level).
- `03_eda_visualization.py` - Exploratory Data Analysis mapping distributions and correlations.
- `04_data_preprocessing.py` - Normalization (`MinMaxScaler`) and sequential temporal windowing into 3D Numpy arrays.
- `05_cnn_model.py` - The core Conv1D model definition and iterative training loop with `EarlyStopping` and `ReduceLROnPlateau`.
- `06_model_evaluation.py` - Calculates ultimate MAE, RMSE, and R² scores, saving regression plots.
- `07_streamlit_app.py` - **The Crown Jewel:** The massive 5-Tab interactive visualization dashboard.
- `08_explainable_ai.py` - Generates background datasets and initializes the `shap.DeepExplainer`.
- `09_fastapi_server.py` - Boots a live FastAPI REST backend server.
- `09_test_api.py` - Simulates a localized IoT device querying the server payload.
- `10_baseline_comparison.py` - Automates benchmarking of the CNN against traditional simple/linear ML models.

---

## 💻 Installation & Usage

### 1. Requirements
Ensure you have Python 3.9+ installed. Install the dependencies via terminal:
```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib shap streamlit plotly fastapi uvicorn requests joblib
```

### 2. Run the Full Pipeline
If you are starting from a blank directory, you must execute the scripts in order from `01` to `10` to generate the `.npy` arrays, scalable transformers, and trained `.keras` weights.

### 3. Launch the Streamlit Dashboard
To run the interactive UI:
```bash
python -m streamlit run 07_streamlit_app.py
```

### 4. Launch the REST API
To launch the headless production backend:
```bash
python -m uvicorn 09_fastapi_server:app --reload
```
You can view the Swagger API documentation at `http://localhost:8000/docs`.

---

## 🎓 Academic Contribution
**Author:** Rohit Choudhary  
**Institution:** SRM University  

This project was developed strictly to address core research gaps in single-point carbon predictors. By utilizing rolling window topologies and mathematical week-ahead simulations, this 1D CNN establishes a verifiable benchmark for deep-temporal regression modeling in the environmental sector.
