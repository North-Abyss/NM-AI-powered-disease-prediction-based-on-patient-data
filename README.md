# 🧠 NM - AI-Powered Disease Prediction Based on Patient Data

Transforming healthcare with AI by enabling early prediction of diseases like Heart Disease and Diabetes using machine learning models trained on patient data.

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Completed-blue)

## 🚀 Live Demo
🔗 [Click here to try the app on Streamlit Cloud](https://nmksyaipredictionbasedonpatientdata.streamlit.app/)

## 📌 Project Highlights

- Supports **Heart Disease** and **Diabetes** prediction
- Real-time prediction using **Random Forest Classifier**
- User-friendly Streamlit interface
- Visualizes risk probability
- Built with **pandas**, **scikit-learn**, and **Streamlit**
- Easily deployable on **Streamlit Cloud**

## 🧬 Datasets Used

| Disease        | Source                                                                                   | Records | Features |
|----------------|-------------------------------------------------------------------------------------------|---------|----------|
| Heart Disease  | [Heart UCI Dataset](https://github.com/sharmaroshan/Heart-UCI-Dataset/blob/master/heart.csv) | ~300    | 13+      |
| Diabetes       | [Diabetes BRFSS Dataset](https://github.com/Helmy2/Diabetes-Health-Indicators/blob/main/diabetes_binary_health_indicators_BRFSS2015.csv) | ~25,000 | 17       |

## 🛠️ Features

- 🔄 Toggle between Heart Disease and Diabetes prediction
- 📋 Interactive patient input form
- 🧠 Real-time ML prediction with probabilities
- 📊 Visual results to support decision making

## 🧪 How It Works

```mermaid
flowchart TD
    A[Select Dataset] --> B[Input Patient Data]
    B --> C[Preprocessing]
    C --> D[Model Prediction]
    D --> E[Show Risk Level]
