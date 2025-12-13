
import joblib
import numpy as np
import sys
import os

# Try to import real classes first
try:
    from predict_mental_health import (
        RewardModel, VariationalPreferenceLearning, ActiveLearner,
        EnhancedQuantumTransform, SuperEnsemble95
    )
    print("✓ Imported classes from predict_mental_health")
    
    # Hack: If models were saved from a script where these were in __main__,
    # we need to alias them in this script's __main__ for joblib to find them.
    # Check if we are running as __main__
    if __name__ == "__main__":
        import sys
        current_module = sys.modules[__name__]
        setattr(current_module, 'SuperEnsemble95', SuperEnsemble95)
        setattr(current_module, 'EnhancedQuantumTransform', EnhancedQuantumTransform)
        setattr(current_module, 'RewardModel', RewardModel)
        setattr(current_module, 'VariationalPreferenceLearning', VariationalPreferenceLearning)
        setattr(current_module, 'ActiveLearner', ActiveLearner)
        print("✓ Aliased classes to __main__ for pickle compatibility")
        
except ImportError as e:
    print(f"⚠ Could not import from predict_mental_health: {e}")
    # Define dummies if import fails (fallback)
    class RewardModel:
        def __init__(self, n_features): pass
    class VariationalPreferenceLearning:
        def __init__(self, n_features, n_latent): pass
    class ActiveLearner:
        def __init__(self, n_queries=20): pass
    class EnhancedQuantumTransform:
        def __init__(self, n_features): pass
        def transform(self, X): return X
    class SuperEnsemble95:
        def __init__(self, n_features, n_classes): pass


def inspect_model(name, path):
    print(f"\n--- Inspecting {name} ---")
    try:
        model = joblib.load(path)
        print(f"Type: {type(model)}")
        if hasattr(model, 'estimators_'):
            print(f"Estimators: {len(model.estimators_)}")
        if hasattr(model, 'classes_'):
            print(f"Classes: {model.classes_}")
        return model
    except Exception as e:
        print(f"Error loading {name}: {e}")
        # Detailed debug
        import traceback
        traceback.print_exc()
        return None

def test_determinism(model, scaler, input_data):
    print(f"Testing determinism for {model}...")
    try:
        if scaler:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data
            
        preds = []
        probas = []
        for i in range(5):
            p = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]
            preds.append(p)
            probas.append(prob)
            
        print(f"Predictions (5 runs): {preds}")
        # print(f"Probabilities: {probas}")
        
        if len(set(preds)) > 1:
            print("!!! WARNING: Model is non-deterministic (Random Output) !!!")
        else:
            print("Model is deterministic.")
            
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    MODEL_DIR = "model"
    
    print("Starting Model Diagnostics...")
    
    # Check if model dir exists
    if not os.path.exists(MODEL_DIR):
        print(f"Error: {MODEL_DIR} directory not found!")
        sys.exit(1)

    # 1. Sleep Model
    # Input: [age, gender_enc, screen_time, night_screen, social_media, stress]
    sleep_model = inspect_model("Sleep Model", f"{MODEL_DIR}/students_sleep_model.joblib")
    sleep_scaler = inspect_model("Sleep Scaler", f"{MODEL_DIR}/students_sleep_scaler.joblib")
    
    if sleep_model and sleep_scaler:
        test_input = np.array([[20, 0, 6.5, 3.0, 4.5, 22]]) 
        test_determinism(sleep_model, sleep_scaler, test_input)

    # 2. Depression Model
    # Input: [age, gender, role, social_media, platform, phq9, gad7]
    dep_model = inspect_model("Depression Model", f"{MODEL_DIR}/depression_model.joblib")
    dep_scaler = inspect_model("Depression Scaler", f"{MODEL_DIR}/depression_scaler.joblib")
    
    if dep_model and dep_scaler:
        test_input = np.array([[25, 1, 2, 5.0, 1, 14, 12]])
        test_determinism(dep_model, dep_scaler, test_input)
