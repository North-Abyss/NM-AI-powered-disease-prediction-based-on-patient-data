import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Disease Predictor", layout="centered")

st.title("🧠 AI-Powered Disease Prediction")
st.markdown("Transforming healthcare with AI-powered disease prediction based on patient data")

# Dataset selection
disease_option = st.selectbox("Select Disease to Predict", ["Heart Disease", "Diabetes"])

# Load selected dataset
if disease_option == "Heart Disease":
    url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    df = pd.read_csv(url)
    X = df.drop("target", axis=1)
    y = df["target"]
    disease_name = "Heart Disease"

    # User Inputs
    def user_input():
        return pd.DataFrame({
            "age": [st.slider("Age", 29, 77, 55)],
            "sex": [st.selectbox("Sex", [0, 1])],
            "cp": [st.slider("Chest Pain Type", 0, 3, 1)],
            "trestbps": [st.slider("Resting Blood Pressure", 94, 200, 130)],
            "chol": [st.slider("Cholesterol", 126, 564, 246)],
            "fbs": [st.selectbox("Fasting Blood Sugar > 120", [0, 1])],
            "restecg": [st.slider("Rest ECG", 0, 2, 1)],
            "thalach": [st.slider("Max Heart Rate", 71, 202, 150)],
            "exang": [st.selectbox("Exercise Induced Angina", [0, 1])],
            "oldpeak": [st.slider("ST Depression", 0.0, 6.2, 1.0)],
            "slope": [st.slider("ST Slope", 0, 2, 1)],
            "ca": [st.slider("Vessels Colored", 0, 4, 0)],
            "thal": [st.slider("Thalassemia (0–2)", 0, 2, 1)]
        })

elif disease_option == "Diabetes":
    url = "https://raw.githubusercontent.com/Helmy2/Diabetes-Health-Indicators/main/diabetes_binary_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(url)
    df = df.sample(n=5000, random_state=1)  # Use 5K for speed
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]
    disease_name = "Diabetes"

    def user_input():
        return pd.DataFrame({
            "HighBP": [st.selectbox("High BP", [0, 1])],
            "HighChol": [st.selectbox("High Cholesterol", [0, 1])],
            "CholCheck": [st.selectbox("Cholesterol Check Done", [0, 1])],
            "BMI": [st.slider("BMI", 12.0, 80.0, 30.0)],
            "Smoker": [st.selectbox("Smoker", [0, 1])],
            "Stroke": [st.selectbox("History of Stroke", [0, 1])],
            "HeartDiseaseorAttack": [st.selectbox("Heart Disease/Attack", [0, 1])],
            "PhysActivity": [st.selectbox("Physical Activity", [0, 1])],
            "Fruits": [st.selectbox("Eats Fruits", [0, 1])],
            "Veggies": [st.selectbox("Eats Vegetables", [0, 1])],
            "HvyAlcoholConsump": [st.selectbox("Heavy Alcohol Use", [0, 1])],
            "AnyHealthcare": [st.selectbox("Has Healthcare", [0, 1])],
            "NoDocbcCost": [st.selectbox("No Doctor Due to Cost", [0, 1])],
            "GenHlth": [st.slider("General Health (1–5)", 1, 5, 3)],
            "MentHlth": [st.slider("Mental Health Days", 0, 30, 5)],
            "PhysHlth": [st.slider("Physical Health Days", 0, 30, 5)],
            "DiffWalk": [st.selectbox("Difficulty Walking", [0, 1])]
        })

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Get user input and predict
st.subheader("🔍 Enter Patient Data:")
user_data = user_input()

st.subheader("🧪 Prediction:")
prediction = model.predict(user_data)
prediction_proba = model.predict_proba(user_data)

result = f"🟢 No {disease_name}" if prediction[0] == 0 else f"🔴 At Risk of {disease_name}"
st.success(result)

st.subheader("📊 Prediction Probability:")
st.write(f"{disease_name} Risk: {round(prediction_proba[0][1]*100, 2)} %")
