import pandas as pd
import numpy as np
import os

def synthesize_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Renaming and formatting time_index
    df.rename(columns={'Date': 'time_index'}, inplace=True)
    df['time_index'] = pd.to_datetime(df['time_index'])
    
    # Sort chronologically (Critokical for CNN Time-Series later)
    df = df.sort_values(['City', 'time_index']).reset_index(drop=True)
    
    n_samples = len(df)
    np.random.seed(42)
    
    print("Generating realistic synthetic variables...")
    
    # 2. Extract out temporal properties for realistic weather generation
    months = df['time_index'].dt.month
    
    # Simulate Temperature (C) with AR(1) temporal persistence
    # Real weather changes gradually — AR(0.85) gives ~2-day half-life
    base_temp = 30 + 10 * np.sin((months - 5) * np.pi / 6)
    AR_COEF_TEMP = 0.85
    temp_list = [float(base_temp.iloc[0]) + np.random.normal(0, 1)]
    for i in range(1, n_samples):
        steady = float(base_temp.iloc[i])
        shock = np.random.normal(0, 1)
        curr = AR_COEF_TEMP * temp_list[-1] + (1 - AR_COEF_TEMP) * steady + shock
        temp_list.append(round(max(10, min(48, curr)), 1))
    df['temperature'] = temp_list
    
    # Simulate Humidity (%) with AR(1) temporal persistence
    base_humidity = 50 + 25 * np.sin((months - 8) * np.pi / 6)
    AR_COEF_HUM = 0.85
    hum_list = [float(base_humidity.iloc[0]) + np.random.normal(0, 2)]
    for i in range(1, n_samples):
        steady = float(base_humidity.iloc[i])
        shock = np.random.normal(0, 2)
        curr = AR_COEF_HUM * hum_list[-1] + (1 - AR_COEF_HUM) * steady + shock
        hum_list.append(round(max(10, min(100, curr)), 1))
    df['humidity'] = hum_list
    
    # Simulate Wind Speed (km/h) with AR(1) temporal persistence
    AR_COEF_WIND = 0.8
    wind_list = [np.random.lognormal(mean=2.0, sigma=0.5)]
    for i in range(1, n_samples):
        shock = np.random.normal(0, 1.5)
        curr = AR_COEF_WIND * wind_list[-1] + (1 - AR_COEF_WIND) * 7.5 + shock
        wind_list.append(round(max(0.1, curr), 1))
    df['wind_speed'] = wind_list
    
    # 3. Industry Level (1-5) with Markov persistence
    # Real industries maintain stable operational levels; 90% chance of staying same
    ind_list = [np.random.randint(1, 6)]
    for i in range(1, n_samples):
        if np.random.random() < 0.9:
            ind_list.append(ind_list[-1])
        else:
            change = np.random.choice([-1, 1])
            new_val = max(1, min(5, ind_list[-1] + change))
            ind_list.append(new_val)
    df['industry_level'] = ind_list
    
    # 4. Vehicle Count with temporal autocorrelation (AR(1) process)
    # Logic: High industry -> More commercial traffic -> High vehicles
    # AR(1): today's count is 70% of yesterday's + new shock
    AR_COEF_VEH = 0.85
    vehicle_list = []
    prev = 15000 + (int(df['industry_level'].iloc[0]) * 5000)
    for i in range(n_samples):
        shock = np.random.normal(loc=0, scale=3000)
        industry_base = df['industry_level'].iloc[i] * 5000
        curr = AR_COEF_VEH * prev + (1 - AR_COEF_VEH) * (15000 + industry_base) + shock
        curr = max(curr, 1000)
        vehicle_list.append(int(curr))
        prev = curr
    df['vehicle_count'] = vehicle_list
    
    # 5. Energy Usage (MWh) with temporal autocorrelation (AR(1) process)
    # Logic: High industry -> High energy AND High temp (>30C) -> High energy (AC Usage)
    # AR(1): energy demand persists day-to-day (industrial schedules, grid inertia)
    AR_COEF_ENE = 0.88
    ac_load = np.where(df['temperature'] > 30, (df['temperature'] - 30) * 15, 0)
    energy_list = []
    industry_arr = df['industry_level'].values
    prev_e = 150 + (float(industry_arr[0]) * 120) + float(ac_load[0])
    for i in range(n_samples):
        steady_state = 150 + (industry_arr[i] * 120) + ac_load[i]
        shock = np.random.normal(loc=0, scale=20)
        curr_e = AR_COEF_ENE * prev_e + (1 - AR_COEF_ENE) * steady_state + shock
        energy_list.append(round(curr_e, 2))
        prev_e = curr_e
    df['energy_usage'] = energy_list
    
    # 6. Carbon Emission (Target)
    # Realistic relation:
    # High Energy -> High Emission
    # High Vehicles -> High Emission
    # High Wind Speed -> Blows emission away (Negative correlation)
    
    # Ensure real pollutants are numeric (in case of dirty 'str' data) and fill NAs
    temp_co = pd.to_numeric(df['CO'], errors='coerce')
    temp_co = temp_co.fillna(temp_co.median())
    
    temp_no2 = pd.to_numeric(df['NO2'], errors='coerce')
    temp_no2 = temp_no2.fillna(temp_no2.median())
    
    # Carbon emission with AR(1) — real pollutant levels persist in atmosphere
    # All 6 CNN features must have direct, learnable effects on emission
    base_emission = (
        (df['energy_usage'] * 3.5) +           # Energy dominates
        (df['vehicle_count'] * 0.08) +         # Vehicles contribute heavily
        (df['temperature'] * 50.0) +           # Hotter -> more AC/combustion emission
        (df['industry_level'] * 200.0) +       # Higher industry -> more emission
        (df['humidity'] * 8.0) +               # Moderate positive effect
        (temp_co * 5.0) +                      # Positive correlation with other pollutants
        (temp_no2 * 2.0) -                     # Positive correlation with NO2
        (df['wind_speed'] * 80.0)              # Wind strongly disperses pollution
    )
    AR_COEF_EMISSION = 0.6
    emission_list = [float(base_emission.iloc[0]) + np.random.normal(0, 10)]
    for i in range(1, n_samples):
        curr = (AR_COEF_EMISSION * emission_list[-1] +
                (1 - AR_COEF_EMISSION) * float(base_emission.iloc[i]) +
                np.random.normal(0, 10))
        emission_list.append(round(max(50, curr), 2))
    df['carbon_emission'] = emission_list
    
    # 7. Select final columns for Layer 2 Output
    # The CNN will ingest these specific features
    final_cols = [
        'City', 'time_index', 'temperature', 'humidity', 'wind_speed', 
        'industry_level', 'vehicle_count', 'energy_usage', 'carbon_emission'
    ]
    
    # We keep the original pollutants as well in case we want to use them later,
    # but the requested columns are explicitly guaranteed to be here.
    final_df = df
    
    final_df.to_csv(output_path, index=False)
    print(f"Layer 2 Complete! Saved 'final_df' to {output_path}")
    print("\nPreview of final_df columns for the CNN:")
    summary_cols = ['time_index', 'temperature', 'industry_level', 'vehicle_count', 'energy_usage', 'carbon_emission']
    print(final_df[summary_cols].head(10))

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "city_day.csv")
    output_file = os.path.join(current_dir, "final_df.csv") 
    
    if os.path.exists(input_file):
        synthesize_data(input_file, output_file)
    else:
        print(f"Error: Could not find {input_file}.")
