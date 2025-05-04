import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Disease Predictor", layout="centered")

st.title("🧠 AI-Powered Disease Prediction")
st.markdown("Transforming healthcare with AI-powered disease prediction based on patient data")

# Helper function
def yes_no_to_binary(choice):
    return 1 if choice == "Yes" else 0

# Disease selection
disease_option = st.selectbox("Select Disease to Predict", ["Heart Disease", "Diabetes"])

# Define variables outside the condition to avoid NameError
df = None
X = None
y = None
user_input = None
disease_name = ""

# Disease-specific logic
if disease_option == "Heart Disease":
    url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    df = pd.read_csv(url)
    X = df.drop("target", axis=1)
    y = df["target"]
    disease_name = "Heart Disease"

    def get_user_input():
        sex_input = st.selectbox("Sex", ["Male", "Female"])
        sex = 1 if sex_input == "Male" else 0
        return pd.DataFrame({
            "age": [st.slider("Age", 29, 77, 55)],
            "sex": [sex],
            "cp": [st.slider("Chest Pain Type (0–3)", 0, 3, 1)],
            "trestbps": [st.slider("Resting BP", 94, 200, 130)],
            "chol": [st.slider("Cholesterol", 126, 564, 246)],
            "fbs": [yes_no_to_binary(st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"]))],
            "restecg": [st.slider("Rest ECG (0–2)", 0, 2, 1)],
            "thalach": [st.slider("Max Heart Rate", 71, 202, 150)],
            "exang": [yes_no_to_binary(st.selectbox("Exercise Induced Angina", ["Yes", "No"]))],
            "oldpeak": [st.slider("ST Depression", 0.0, 6.2, 1.0)],
            "slope": [st.slider("ST Slope", 0, 2, 1)],
            "ca": [st.slider("Vessels Colored by Fluoroscopy", 0, 4, 0)],
            "thal": [st.slider("Thalassemia (0–2)", 0, 2, 1)]
        })

elif disease_option == "Diabetes":
    url = "https://raw.githubusercontent.com/Helmy2/Diabetes-Health-Indicators/main/diabetes_binary_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(url)
    df = df.sample(n=5000, random_state=1)
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]
    disease_name = "Diabetes"

    def get_user_input():
        return pd.DataFrame({
            "HighBP": [yes_no_to_binary(st.selectbox("High Blood Pressure", ["Yes", "No"]))],
            "HighChol": [yes_no_to_binary(st.selectbox("High Cholesterol", ["Yes", "No"]))],
            "CholCheck": [yes_no_to_binary(st.selectbox("Cholesterol Check Done", ["Yes", "No"]))],
            "BMI": [st.slider("BMI", 12.0, 80.0, 30.0)],
            "Smoker": [yes_no_to_binary(st.selectbox("Smoker", ["Yes", "No"]))],
            "Stroke": [yes_no_to_binary(st.selectbox("History of Stroke", ["Yes", "No"]))],
            "HeartDiseaseorAttack": [yes_no_to_binary(st.selectbox("Heart Disease or Attack", ["Yes", "No"]))],
            "PhysActivity": [yes_no_to_binary(st.selectbox("Physical Activity", ["Yes", "No"]))],
            "Fruits": [yes_no_to_binary(st.selectbox("Eats Fruits", ["Yes", "No"]))],
            "Veggies": [yes_no_to_binary(st.selectbox("Eats Vegetables", ["Yes", "No"]))],
            "HvyAlcoholConsump": [yes_no_to_binary(st.selectbox("Heavy Alcohol Use", ["Yes", "No"]))],
            "AnyHealthcare": [yes_no_to_binary(st.selectbox("Has Healthcare", ["Yes", "No"]))],
            "NoDocbcCost": [yes_no_to_binary(st.selectbox("No Doctor Due to Cost", ["Yes", "No"]))],
            "GenHlth": [st.slider("General Health (1–5)", 1, 5, 3)],
            "MentHlth": [st.slider("Mental Health Days", 0, 30, 5)],
            "PhysHlth": [st.slider("Physical Health Days", 0, 30, 5)],
            "DiffWalk": [yes_no_to_binary(st.selectbox("Difficulty Walking", ["Yes", "No"]))]
        })

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# User input
st.subheader("🔍 Enter Patient Data:")
user_data = get_user_input()

# Align user_data to model input features
try:
    user_data = user_data.reindex(columns=X.columns, fill_value=0)
except Exception as e:
    st.error(f"⚠️ Error aligning user input with model features: {e}")
    st.stop()

# Prediction
st.subheader("🧪 Prediction:")
prediction = model.predict(user_data)
prediction_proba = model.predict_proba(user_data)

result = f"🟢 No {disease_name}" if prediction[0] == 0 else f"🔴 At Risk of {disease_name}"
st.success(result)

# Probability
st.subheader("📊 Prediction Probability:")
st.write(f"{disease_name} Risk: {round(prediction_proba[0][1] * 100, 2)} %")
