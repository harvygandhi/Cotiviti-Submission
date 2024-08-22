import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Simulate data
np.random.seed(42)
n_patients = 1000

data = {
    'age': np.random.randint(20, 80, size=n_patients),
    'gender': np.random.choice(['Male', 'Female'], size=n_patients),
    'insurance': np.random.choice(['Insurance_A', 'Insurance_B', 'Insurance_C'], size=n_patients),
    'admission_type': np.random.choice(['Emergency', 'Elective'], size=n_patients),
    'treatment_type': np.random.choice(['Treatment_A', 'Treatment_B', 'Treatment_C'], size=n_patients),
    'dosage': np.random.uniform(10, 100, size=n_patients),
    'predicted_los': np.random.uniform(3, 14, size=n_patients),
    'actual_los': np.random.uniform(3, 14, size=n_patients),
    'baseline_los': np.random.uniform(3, 14, size=n_patients)
}

df = pd.DataFrame(data)

# Define categorical columns
categorical_cols = ['gender', 'insurance', 'admission_type', 'treatment_type']

# Create Column Transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Define features and targets for baseline LOS model
features = categorical_cols + ['age', 'dosage']
target_baseline = 'baseline_los'
target_predicted = 'predicted_los'

# Split data for baseline LOS
X_baseline = df[features]
y_baseline = df[target_baseline]
X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(X_baseline, y_baseline, test_size=0.3, random_state=42)

# Train Baseline LOS model
pipeline_baseline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
pipeline_baseline.fit(X_train_baseline, y_train_baseline)

# Evaluate Baseline LOS model
baseline_predictions = pipeline_baseline.predict(X_test_baseline)
print(f'Baseline LOS Model Mean Squared Error: {mean_squared_error(y_test_baseline, baseline_predictions)}')

# Split data for predicted LOS
X_predicted = df[features]
y_predicted = df[target_predicted]
X_train_predicted, X_test_predicted, y_train_predicted, y_test_predicted = train_test_split(X_predicted, y_predicted, test_size=0.3, random_state=42)

# Train Predicted LOS model
pipeline_predicted = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
pipeline_predicted.fit(X_train_predicted, y_train_predicted)

# Evaluate Predicted LOS model
predicted_predictions = pipeline_predicted.predict(X_test_predicted)
print(f'Predicted LOS Model Mean Squared Error: {mean_squared_error(y_test_predicted, predicted_predictions)}')

# Function to predict LOS for new data
def predict_los_for_new_data(new_data, pipeline):
    new_data_encoded = new_data.copy()
    predictions = pipeline.predict(new_data_encoded)
    return predictions

# Example new patient data
new_patient_data = pd.DataFrame({
    'age': [50],
    'gender': ['Male'],
    'insurance': ['Insurance_B'],
    'admission_type': ['Elective'],
    'treatment_type': ['Treatment_A'],
    'dosage': [60],
    'predicted_los': [7]
})

# Predict baseline LOS for new patient
baseline_los_prediction = predict_los_for_new_data(new_patient_data, pipeline_baseline)
print(f'Predicted Baseline LOS for new patient: {baseline_los_prediction[0]:.2f} days')

# Predict LOS based on treatment for new patient
predicted_los_prediction = predict_los_for_new_data(new_patient_data, pipeline_predicted)
print(f'Predicted LOS for new patient with treatment: {predicted_los_prediction[0]:.2f} days')
