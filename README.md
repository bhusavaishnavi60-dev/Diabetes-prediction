**🩺 Diabetes Prediction System using Machine Learning**

An AI & Machine Learning based web application that predicts whether a person is at risk of diabetes using health parameters such as age, BMI, HbA1c level, blood glucose level, hypertension, heart disease, gender, and smoking history.

## 🚀 Live Demo

Streamlit App: https://diabetes-prediction-4xjabnfechfhmfcykj8dl8.streamlit.app/

## 📌 Project Overview

This project uses a trained Machine Learning model to predict diabetes risk based on patient health information. The application provides an easy-to-use web interface built with Streamlit.

## ✨ Features

* 🔐 Login and Signup interface.
* 📝 User-friendly patient health form.
* 🤖 Diabetes prediction using Machine Learning.
* 📊 Prediction confidence percentage.
* ❤️ Health tips and precautions based on prediction.
* 📱 Responsive web application.

## 🧠 Machine Learning Model

* Algorithm: Logistic Regression
* Libraries: Scikit-learn, NumPy, Pandas
* Model Files:

  * `model.pkl`
  * `scaler.pkl`
  * `gender_encoder.pkl`
  * `smoking_encoder.pkl`

## 📋 Input Parameters

* Gender
* Age
* Hypertension
* Heart Disease
* Smoking History
* BMI
* HbA1c Level
* Blood Glucose Level

## 🛠️ Tech Stack

* Python
* Streamlit
* Scikit-learn
* NumPy
* Pandas
* Joblib
* Git & GitHub

## ▶️ How to Run Locally

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## 📂 Project Structure

```text
Diabetes-Prediction/
│── app.py
│── model.pkl
│── scaler.pkl
│── gender_encoder.pkl
│── smoking_encoder.pkl
│── requirements.txt
│── README.md
```

## 🎯 Future Improvements

* PDF prediction report.
* BMI health calculator.
* Diabetes risk meter.
* Patient history dashboard.

