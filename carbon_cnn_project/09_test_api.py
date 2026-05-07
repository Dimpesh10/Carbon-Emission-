import requests
import json
import time

# ==========================================================
# Layer 9 API Client Simulation
# Represents a mobile app or IoT dashboard calling the server
# ==========================================================

API_URL = "http://localhost:8000"

def test_api():
    print("1. Pinging DevOps Health Check...")
    try:
        health_resp = requests.get(f"{API_URL}/health")
        print(f"Health Response: {health_resp.json()}\n")
    except requests.exceptions.ConnectionError:
        print("[FAIL] Server is offline. Please run: uvicorn 09_fastapi_server:app --reload")
        return

    print("2. Simulating IoT Sensor Payload (Normal Scenario)...")
    
    # Mock JSON array representing real-time sensor measurements
    normal_payload = {
        "temperature": 25.5,
        "humidity": 60.2,
        "wind_speed": 12.0,
        "industry_level": 2,
        "vehicle_count": 5000,
        "energy_usage": 150.5
    }
    
    # Second extreme scenario
    danger_payload = {
        "temperature": 48.0,
        "humidity": 90.0,
        "wind_speed": 2.0,
        "industry_level": 5,
        "vehicle_count": 80000,
        "energy_usage": 900.0
    }
    
    def run_prediction(payload, scenario_name):
        print(f"\n--- Running {scenario_name} ---")
        print(f"Sending strictly typed JSON: {json.dumps(payload, indent=2)}")
        
        start_time = time.time()
        predict_resp = requests.post(f"{API_URL}/predict", json=payload)
        end_time = time.time()
        
        if predict_resp.status_code == 200:
            data = predict_resp.json()
            print("[OK] API Inference Successful!")
            print(f"Network Latency + Inference Time: {round((end_time - start_time) * 1000, 2)} ms")

            print("\nExtracted Values for UI:")
            print(f"  Predicted CO2: {data['predicted_emission']} kg")
            print(f"  Status Level:  {data['status_level']}")
            print(f"  AI Confidence: {data['confidence']}")
        else:
            print(f"[FAIL] API Failed with status code: {predict_resp.status_code}")
            print(predict_resp.text)
            
    run_prediction(normal_payload, "Normal Safe Scenario")
    run_prediction(danger_payload, "Extreme Danger / Anomaly Scenario")

if __name__ == "__main__":
    test_api()
