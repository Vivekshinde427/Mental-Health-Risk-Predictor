import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
import os

# Automatically get the path of the current script
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'mental_health.csv')

# Now load the CSV
df = pd.read_csv(file_path)

expected_columns = [
    'Age', 'Gender', 'family_history', 'work_interfere', 'remote_work',
    'no_employees', 'mental_health_consequence', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'anonymity',
    'leave', 'mental_health_interview', 'phys_health_interview'
]

df = df[[col for col in expected_columns if col in df.columns]].copy()

# Binary column encoding
def map_binary(col):
    return col.map({'Yes': 1, 'No': 0}).fillna(0)

binary_cols = [
    'family_history', 'remote_work', 'mental_health_consequence',
    'benefits', 'care_options', 'wellness_program', 'seek_help',
    'anonymity', 'mental_health_interview', 'phys_health_interview'
]

for col in binary_cols:
    if col in df.columns:
        df[col] = map_binary(df[col])

# Gender encoding
df['Gender'] = df['Gender'].apply(lambda x: 1 if 'male' in str(x).lower() else 0)

# Leave encoding
leave_map = {
    'Very easy': 1, 'Somewhat easy': 0.7, 'Don’t know': 0.3,
    'Somewhat difficult': 0.2, 'Very difficult': 0
}
df['leave'] = df['leave'].map(leave_map).fillna(0.3)

# Work interfere encoding
work_map = {'Often': 1, 'Sometimes': 0.6, 'Rarely': 0.3, 'Never': 0}
df['work_interfere'] = df['work_interfere'].map(work_map).fillna(0.3)

# Employee size mapping
emp_map = {
    '1-5': 1, '6-25': 2, '26-100': 3, '100-500': 4,
    '500-1000': 5, 'More than 1000': 6
}
df['no_employees'] = df['no_employees'].map(emp_map).fillna(3)

# Age cleaning
df['Age'] = df['Age'].apply(lambda x: x if 18 <= x <= 70 else np.nan)
df['Age'] = df['Age'].fillna(df['Age'].median())

# 🧠 Enhanced risk score formula (normalized and weighted)
df['risk_score'] = (
    df['family_history'] * 20 +
    df['mental_health_consequence'] * 15 +
    df['work_interfere'] * 10 +
    (1 - df['benefits']) * 10 +
    (1 - df['leave']) * 10 +
    (1 - df['care_options']) * 10 +
    df['remote_work'] * 3 +
    df['seek_help'] * 4 +
    df['anonymity'] * 5 +
    df['wellness_program'] * 2 +
    df['mental_health_interview'] * 3 +
    df['phys_health_interview'] * 3 +
    df['Gender'] * 1 +
    df['Age'] * 0.3 +
    df['no_employees'] * 2
)

# Prepare features and label
features = df.drop(columns=['risk_score'])
target = df['risk_score']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features_imputed, target, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R² Score:", r2)

# Save artifacts
joblib.dump(model, 'predictor.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(list(features.columns), 'features.pkl')

with open("metrics.txt", "w") as f:
    f.write(f"MSE: {mse:.4f}\nR2 Score: {r2:.4f}")
