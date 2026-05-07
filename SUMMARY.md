# Project Summary — Carbon Emission Forecasting Using Transformer

## What This Project Does

This project predicts **daily carbon emissions (in kg CO2)** for a city using a **Transformer Encoder** deep learning model. The user provides 6 real-time sensor inputs, and the Transformer analyzes 7 days of patterns to forecast tomorrow's emission level.

The system includes:
- A trained Transformer model that learned emission patterns from historical data
- An interactive Streamlit dashboard with 5 tabs for prediction, evaluation, and explainability
- A FastAPI REST server for real-time inference from IoT devices or mobile apps

---

## How Values Change — The Full Data Flow

### Step 1: User Inputs 6 Sensor Values

| Input | What It Represents | Example |
|-------|-------------------|---------|
| Temperature (°C) | Ambient air temperature | 35.5 |
| Humidity (%) | Relative air humidity | 50.0 |
| Wind Speed (km/h) | Surface wind speed | 5.0 |
| Industry Level (1-5) | Industrial activity intensity | 3 |
| Vehicle Count | Daily traffic volume | 22,000 |
| Energy Usage (MWh) | Daily power consumption | 200 |

### Step 2: Feature Scaling

The raw values are **scaled to [0, 1]** using a MinMaxScaler that was fitted during training. This is critical because neural networks work best with normalized inputs.

Example: If temperature ranges from 10°C to 50°C in the training data:
- 35.5°C becomes → (35.5 - 10) / (50 - 10) = **0.6375**

All 6 features are scaled the same way using their respective min/max ranges.

### Step 3: Building the 7-Day Window

The Transformer expects a **7-day sequence** of data — shape `(1, 7, 6)`. Since the user provides only 1 day of values, the system:
1. Takes the scaled feature vector `[0.64, 0.45, 0.12, 0.50, 0.27, 0.33]`
2. Copies it 7 times to simulate a week of similar conditions
3. Adds small random noise (std=0.05) to Days 1-6 for realistic variation
4. Day 7 keeps the exact user values (no noise)

This creates a `(1, 7, 6)` tensor — 1 sample, 7 timesteps, 6 features.

### Step 4: Transformer Inference

The 7-day tensor passes through the Transformer Encoder:
1. Each day's 6 features are projected to a 64-dimensional space
2. Positional encoding tells the model which day is which (Day 1 vs Day 7)
3. Self-attention lets every day look at every other day to find patterns
4. The model outputs a single scaled prediction (e.g., 0.4501)

### Step 5: Inverse Scaling

The scaled prediction is converted back to real kg CO2 using the target scaler:
- 0.4501 → **4,501 kg CO2**

### Step 6: Display

The prediction flows to the dashboard which shows:
- The emission value (4,501 kg CO2)
- A status badge (Safe / Warning / Danger based on thresholds)
- A gauge showing % of 10,000 kg limit
- SHAP values explaining which features drove the prediction

**When you move any slider, Steps 2-6 re-run instantly**, producing a new prediction.

---

## How Features Are Dependent on Each Other

The Transformer learns these dependencies from training data:

| Relationship | How It Works |
|-------------|-------------|
| Temperature + Energy | Higher temperature → more AC usage → higher energy consumption → more emissions |
| Industry Level + Vehicle Count | Industrial zones attract more vehicles (workers, trucks) → combined emission spike |
| Wind Speed + All Features | High wind disperses pollutants → effectively reduces local emission concentration |
| Humidity + Temperature | High humidity traps heat and pollutants → amplifies emission impact |
| Vehicle Count + Energy | More vehicles = more fuel burned = more energy sector load |

The Transformer's self-attention mechanism captures these cross-feature dependencies automatically — it doesn't need to be told these rules. It discovers them from the data.

---

## Transformer Encoder Algorithm — Detailed Explanation

### Why Transformer?

Traditional models (RNNs, LSTMs) process time steps sequentially — Day 1 then Day 2 then Day 3. This creates a bottleneck: by the time the model reaches Day 7, information from Day 1 may be degraded.

The Transformer processes **all 7 days simultaneously** using a mechanism called **Self-Attention**. Day 7 can directly look at Day 1 without passing through Days 2-6.

### Architecture Step by Step

```
Input: (7 days, 6 features) = (7, 6) tensor

Step 1: Dense Projection
   (7, 6) → Dense(64) → (7, 64)
   Each day's 6 features are projected to a richer 64-dimensional representation.

Step 2: Sinusoidal Positional Encoding
   (7, 64) + PositionalEncoding(7, 64) → (7, 64)
   Since the Transformer has no built-in notion of order, we add unique
   sine/cosine patterns to each day's vector so the model knows
   "this is Day 1" vs "this is Day 7."

   Formula:
   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

   This creates a unique "fingerprint" for each position that the model
   uses to understand temporal order.

Step 3: Multi-Head Self-Attention (2 heads, key_dim=32)
   This is the core innovation.

   For each day, three vectors are computed:
   - Query (Q): "What am I looking for?"
   - Key (K): "What information do I contain?"
   - Value (V): "What information should I pass along?"

   Attention Score = softmax(Q * K^T / sqrt(key_dim)) * V

   Example: If Day 7 has high temperature and Day 3 also had high temperature,
   the attention score between Day 7 and Day 3 will be high, meaning Day 7
   will "pay attention" to Day 3's pattern.

   2 Heads = the model learns 2 different types of relationships:
   - Head 1 might learn temperature-emission patterns
   - Head 2 might learn vehicle-industry patterns

   After attention: Residual Connection (x = x + attention_output)
   Then: LayerNormalization (stabilizes training)

Step 4: Feed-Forward Network
   (7, 64) → Dense(64, ReLU) → Dense(64) → (7, 64)
   A simple neural network applied to each day independently.
   This lets the model learn non-linear feature combinations.

   After FFN: Residual Connection + LayerNormalization (same as Step 3)

Step 5: Global Average Pooling
   (7, 64) → mean across 7 days → (64,)
   Compresses all 7 days into a single 64-dimensional summary vector.

Step 6: Classification Head
   (64) → Dense(64, ReLU) → Dropout(0.3)
        → Dense(32, ReLU) → Dropout(0.2)
        → Dense(1, Linear)  → predicted emission (scaled)

   Dropout randomly deactivates neurons during training to prevent overfitting.
```

### Training Process

| Setting | Value | Why |
|---------|-------|-----|
| Optimizer | Adam | Adapts learning rate per-parameter, fast convergence |
| Loss Function | Huber Loss | Combines MSE (accurate for small errors) and MAE (robust to outliers) |
| Epochs | Up to 100 | Maximum training iterations |
| EarlyStopping | patience=10 | Stops if validation loss doesn't improve for 10 epochs |
| ReduceLROnPlateau | factor=0.5 | Halves learning rate when loss plateaus |
| ModelCheckpoint | best only | Saves the model with lowest validation loss |
| Batch Size | 32 | Processes 32 samples per gradient update |

The Transformer converges in approximately 13 epochs (vs CNN's 32 epochs), meaning it learns the emission patterns faster.

---

## Dashboard Tabs — Graph Explanations

### Tab 1: Live Prediction

**Prediction Card (top left):**
- Shows the predicted CO2 emission in kg for the current slider configuration
- Color coding: Green (Safe, <2000), Orange (Warning, 2000-6000), Red (Danger, >6000)

**Gauge Chart (below prediction card):**
- Semi-circular gauge showing predicted emission as a percentage of 10,000 kg limit
- Visual risk indicator — more fill = higher danger

**Prediction History (top right):**
- Line chart tracking all predictions made in the current session
- Each time you change a slider, a new point is added
- Red dashed line at 6,000 kg = danger threshold
- Shows how emissions change as you adjust different inputs

**Sub-Metric Cards:**
- Temp, Vehicles, Energy — quick reference of current input values

**Tree Equivalent:**
- "Equivalent to cutting down X trees per day" — converts CO2 to a tangible metric (1 tree absorbs ~21 kg CO2/year)

---

### Tab 2: Training Evaluation

**Performance Metric Cards:**
- MAE, RMSE, MAPE, R2 displayed for both CNN and Transformer
- These show how accurate each model is on the test dataset (data the model never saw during training)

**Interactive Time-Series Chart:**
- Blue line = Actual emissions (ground truth from test data)
- Green dashed line = Transformer's predictions
- Orange dashed line = CNN's predictions
- Closer the dashed lines follow the blue line = better model
- You can hover, zoom, and pan to inspect specific days
- The Transformer line tracks the actual values more closely than CNN

**Scatter Plots (Predicted vs Actual):**
- Each dot = one test sample
- X-axis = what the emission actually was
- Y-axis = what the model predicted
- Red dashed diagonal = perfect prediction line
- Dots closer to the diagonal = more accurate predictions
- The Transformer's scatter plot shows tighter clustering around the diagonal (R2 = 0.96 vs CNN's 0.93)

**Residual Histograms:**
- Residual = Actual - Predicted (the error for each sample)
- A well-trained model has residuals centered around zero (narrow bell curve)
- Wide spread = model makes large errors sometimes
- The Transformer's histogram is narrower and more centered than CNN's

**Error Over Time:**
- Shows absolute prediction error for each test day
- Spikes indicate days that were hard to predict (unusual conditions)
- Transformer's spikes are generally lower than CNN's

---

### Tab 3: About Model

**Dual Metric Cards (Transformer vs CNN):**
- Transformer row highlighted in green with delta indicators (e.g., "+27.1% vs CNN" for MAE improvement)
- CNN row in grey as baseline reference

**Grouped Bar Charts (6 Models):**
- Left chart: MAE and RMSE bars for all 6 models (Simple Average, Linear Regression, Random Forest, LSTM, CNN, Transformer)
- Transformer bars in green, others in blue
- Lower bars = better model
- Simple Average has the tallest bars (worst), Transformer among the shortest (best)

**R2 Score Bar Chart:**
- R2 measures how much variance the model explains (1.0 = perfect)
- Simple Average has negative R2 (worse than guessing the mean)
- Transformer and LSTM tied at 0.96, proving deep learning captures complex patterns

**Full Metrics Table:**
- All 6 models' MAE, RMSE, MAPE, R2 in a sortable table
- Transformer row highlighted with green background

**Training History Comparison:**
- Left: CNN training curves (loss and MAE over epochs)
- Right: Transformer training curves
- Both show training (blue) and validation (orange) lines
- Transformer converges faster (fewer epochs) and reaches lower loss

**Summary Card:**
- Green bordered box with auto-calculated improvement percentages
- States: "Transformer Encoder achieves X% lower MAE, Y% lower RMSE, Z% higher R2 than CNN"

---

### Tab 4: Explainable AI (SHAP)

**Model Indicator Banner:**
- Green badge showing "Currently Explaining: Transformer Model"
- Confirms SHAP is running on the Transformer, not CNN

**"What is SHAP?" Expander:**
- Educational explanation of SHAP for non-technical audience
- Covers game theory origins and why explainability matters

**Waterfall Chart:**
- Shows how the prediction builds up from a baseline value
- Each bar = one feature's contribution
- Red bars push prediction higher (increase emissions)
- Green bars push prediction lower (decrease emissions)
- The final bar = total prediction
- Example: Vehicle Count adds +800 kg, Wind Speed reduces -200 kg

**Feature Impact Cards:**
- 6 styled cards (one per feature)
- Shows: feature name, contribution in kg CO2, direction (up/down), percentage of total impact
- Red border = increases emission, Green border = decreases emission
- Updates live when you change sliders

**Radar Chart:**
- Polar chart showing absolute SHAP impact % for all 6 features
- Creates a visual "fingerprint" of what the Transformer focuses on
- Larger area toward a feature = that feature has more influence
- Typically Industry Level and Vehicle Count dominate

**AI-Generated Insight:**
- Dynamic text that changes with slider values
- Identifies: strongest emission driver, strongest mitigator
- Provides actionable advice (e.g., "consider reducing industrial activity")

**Global vs Local Analysis:**
- Left column: Global feature importance bar chart (average across ALL test samples)
- Right column: SHAP beeswarm summary plot
  - Each dot = one test sample
  - Color: red = high feature value, blue = low
  - Position on x-axis = SHAP impact
  - Example: If Industry Level dots are red and far right, high industry activity consistently increases emissions

**Interactive Force Plot:**
- Horizontal bar showing features pushing prediction left (lower) or right (higher)
- Red segments = features increasing emission
- Blue segments = features decreasing emission

---

### Tab 5: 7-Day Forecast

**Forecast Line Chart:**
- Projects emissions for the next 7 days based on current inputs
- Each day adds slight variation to simulate realistic conditions
- Shows trend direction (rising, falling, or stable)

**Daily Breakdown:**
- Individual cards for each forecasted day
- Shows predicted emission value and risk level

---

## Key Results

| Metric | Transformer | CNN | Improvement |
|--------|------------|-----|-------------|
| MAE | 218.18 kg | 299.28 kg | 27.1% lower error |
| RMSE | 272.73 kg | 371.07 kg | 26.5% lower error |
| MAPE | 3.83% | 5.57% | 31.2% more accurate |
| R2 | 0.9599 | 0.9257 | 3.7% more variance explained |

The Transformer's self-attention mechanism captures temporal dependencies across all 7 timesteps simultaneously, unlike the CNN's local convolutional kernels which only see 2-3 day windows at a time.
