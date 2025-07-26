# 🧠 Mental Health Risk Predictor for Working Professionals

This project predicts the **Mental Health Risk Score** of working professionals using a **Linear Regression** model trained on various workplace-related features.

---

## 🚀 Demo Screenshot

![Streamlit UI](screenshots/streamlit_ui.png)

---

## 📌 Features

- Predicts **Mental Health Risk Score** (scale: 0 to 100)
- Input factors: Age, Gender, Remote Work, Leave Policy, Company Size, and more
- Uses **Linear Regression**
- Built using **Streamlit** for a real-time interactive UI
- Visual output and feature importance charts

---

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- streamlit
- joblib

---

## 📂 Files Included

- `model.py`: Code to train the ML model
- `app.py`: Streamlit app for prediction
- `predictor.pkl`: Trained model
- `imputer.pkl`, `features.pkl`: Transformers used in prediction
- `mental_health.csv`: Cleaned dataset
- `mental_health_banner.png`: Banner for presentation
- `screenshots/streamlit_ui.png`: UI screenshot

---

## 📈 Model Accuracy

- **Mean Squared Error:** Near 0  
- **R² Score:** 1.0 (on training set)

---

## 🧪 How to Run

```bash
# Clone the repo
git clone https://github.com/VivekShinde427/Mental-Health-Risk-Predictor.git

# Move into the folder
cd Mental-Health-Risk-Predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
