import joblib
import numpy as np
import os

try:
    path = "model/students_sleep_scaler.joblib"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        exit()
        
    scaler = joblib.load(path)
    print(f"Scaler type: {type(scaler)}")
    print(f"Expected features: {getattr(scaler, 'n_features_in_', 'Unknown')}")
    print(f"Means: {getattr(scaler, 'mean_', 'Unknown')}")
    print(f"Scale: {getattr(scaler, 'scale_', 'Unknown')}")
    
    # Test transform
    try:
        test_input = np.array([[25, 1, 6, 2, 3, 15]]) # 6 features
        print(f"Test transform (6 features): {scaler.transform(test_input)}")
    except Exception as e:
        print(f"Test transform failed: {e}")

except Exception as e:
    print(f"Error: {e}")
