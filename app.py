import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit page config
st.set_page_config(page_title="Mental Health Risk Predictor", page_icon="🧠", layout="centered")

# Load ML model and transformers
model = joblib.load("predictor.pkl")
imputer = joblib.load("imputer.pkl")
features = joblib.load("features.pkl")

# Title Section
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 Mental Health Risk Predictor</h1>
    <p style='text-align: center; color: #666;'>Estimate your risk score based on workplace and lifestyle factors</p>
""", unsafe_allow_html=True)

st.markdown("---")
st.subheader("📋 Enter Your Details")

# Collect user input
input_data = {}
for feature in features:
    if feature == "Age":
        input_data[feature] = st.slider("Age", 18, 70)
    elif feature == "Gender":
        input_data[feature] = st.selectbox("Gender", ["Choose...", "Male", "Female"])
    elif feature == "no_employees":
        input_data[feature] = st.selectbox("Company Size", ["Choose...", '1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'])
    elif feature == "leave":
        input_data[feature] = st.selectbox("Ease of Taking Mental Health Leave", ["Choose...", "Very easy", "Somewhat easy", "Don’t know", "Somewhat difficult", "Very difficult"])
    elif feature == "work_interfere":
        input_data[feature] = st.selectbox("How often does mental health interfere with work?", ["Choose...", "Never", "Rarely", "Sometimes", "Often"])
    else:
        input_data[feature] = st.radio(f"{feature.replace('_', ' ').capitalize()}", ["Yes", "No"], index=None)

# Prediction
if st.button("🔍 Predict Risk Score"):
    # Validation
    if "Choose..." in input_data.values():
        st.error("⚠️ Please fill all fields before predicting.")
    else:
        df_input = pd.DataFrame([input_data])

        # Preprocessing
        df_input['Gender'] = df_input['Gender'].apply(lambda x: 1 if 'male' in x.lower() else 0)
        leave_map = {'Very easy': 1, 'Somewhat easy': 0.7, 'Don’t know': 0.3, 'Somewhat difficult': 0.2, 'Very difficult': 0}
        df_input['leave'] = df_input['leave'].map(leave_map)

        work_map = {'Often': 1, 'Sometimes': 0.6, 'Rarely': 0.3, 'Never': 0}
        df_input['work_interfere'] = df_input['work_interfere'].map(work_map)

        emp_map = {'1-5': 1, '6-25': 2, '26-100': 3, '100-500': 4, '500-1000': 5, 'More than 1000': 6}
        df_input['no_employees'] = df_input['no_employees'].map(emp_map)

        binary_cols = [col for col in df_input.columns if col not in ['Age', 'Gender', 'leave', 'work_interfere', 'no_employees']]
        for col in binary_cols:
            df_input[col] = df_input[col].map({'Yes': 1, 'No': 0}).fillna(0)

        # Impute and predict
        df_input_imputed = imputer.transform(df_input[features])
        prediction = model.predict(df_input_imputed)[0]

        # Color-coded result
        color = "#28a745" if prediction < 40 else "#dc3545"
        st.markdown(f"""
        <h3 style='color:{color}; text-align:center;'>🧠 Your Predicted Risk Score: {prediction:.2f}</h3>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Input Overview")

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(x=list(df_input.columns), y=df_input.values[0], ax=ax, palette="coolwarm")
        ax.set_title("Your Input Feature Distribution", fontsize=14)
        ax.set_ylabel("Value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

        if hasattr(model, 'coef_'):
            st.subheader("🔍 Feature Importance (Model Weights)")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.barplot(x=model.coef_, y=features, ax=ax2, palette="viridis")
            ax2.set_xlabel("Coefficient")
            ax2.set_title("Model Feature Contribution")
            st.pyplot(fig2)
