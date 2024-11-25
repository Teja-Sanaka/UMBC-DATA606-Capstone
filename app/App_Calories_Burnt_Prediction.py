# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
import warnings

warnings.filterwarnings('ignore')

st.write("## Calories Burnt Prediction")
st.write("""
Here we will be predicting calories burned based on some personal parameters 
such as Age, Gender, Weight, Height, Duration, Heart Rate, and Body Temperature.
""")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/premium-photo/mat-with-hand-weights-sport-concept-3d-illustration-copy-space-fitness_522591-609.jpg?semt=ais_hybrid"); 
        background-size: cover !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("User Input Parameters:")

def user_input_features():
    age = st.sidebar.slider("Age:", 10, 100, 30)
    weight = st.sidebar.slider("Weight (kg):", 30, 200, 60)
    height = st.sidebar.slider("Height (cm):", 40, 200, 150)
    duration = st.sidebar.slider("Duration (min):", 0, 120, 15)
    heart_rate = st.sidebar.slider("Heart Rate:", 50, 150, 80)
    body_temp = st.sidebar.slider("Body Temperature (C):", 35, 50, 38)
    gender_button = st.sidebar.radio("Gender:", ("Male(0)", "Female(1)"))

    gender = 0 if gender_button == "Male(0)" else 1

    data_model = {
        "age": age,
        "weight": weight,
        "height": height,
        "duration": duration,
        "heart_rate": heart_rate,
        "body_temp": body_temp,
        "gender": gender
    }

    features = pd.DataFrame(data_model, index=[0])
    return features, age, weight, height, duration, heart_rate, body_temp

df, age, weight, height, duration, heart_rate, body_temp = user_input_features()

# File upload
st.sidebar.write("### Upload Datasets")
calories_file = st.sidebar.file_uploader("Upload Calories CSV", type=["csv"])
exercise_file = st.sidebar.file_uploader("Upload Exercise CSV", type=["csv"])

if not calories_file or not exercise_file:
    st.error("Please upload both datasets to proceed!")
else:
    calories = pd.read_csv(calories_file)
    exercise = pd.read_csv(exercise_file)

    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)

    # Train-test split
    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
    exercise_train_data = exercise_train_data[["Gender", "Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_test_data = exercise_test_data[["Gender", "Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

    # One-hot encoding for Gender
    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]
    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]

    # XGBoost model
    xgb_model = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)

    # Prediction on user input with progress bar
    st.write("---")
    st.header("Prediction:")
    latest_iteration = st.empty()
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)

    prediction = xgb_model.predict(df)
    st.write(round(prediction[0], 2), " **kilocalories**")
