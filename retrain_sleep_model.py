
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('dataset/students_sleep_screen.csv')

# 2. Preprocess
print("Preprocessing...")
# Define Target: 1 = Good Sleep (>= 7.5), 0 = Poor Sleep (< 7.5)
# This aligns with app.py logic (1 -> Green/Good)
df['sleep_target'] = (df['sleep_quality_score'] >= 7.5).astype(int)

# Encode Gender: Female=0, Male=1 (Matches app.py)
df['gender_enc'] = df['gender'].apply(lambda x: 0 if x.lower() == 'female' else 1)

# Select Features (6 features used in App)
feature_cols = ['age', 'gender_enc', 'screen_time_hours', 
                'screen_time_night_hours', 'social_media_hours', 'stress_score']

X = df[feature_cols]
y = df['sleep_target']

print(f"Features: {feature_cols}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 3. Scale Features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train Model
print("Training Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Save
print("Saving artifacts...")
joblib.dump(model, 'model/students_sleep_model.joblib')
joblib.dump(scaler, 'model/students_sleep_scaler.joblib')

print("✓ Success! Model and Scaler have been regenerated with 6 features.")
print("  - model/students_sleep_model.joblib")
print("  - model/students_sleep_scaler.joblib")
