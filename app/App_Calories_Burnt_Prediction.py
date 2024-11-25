# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import time
import warnings

# Error handling for missing xgboost
# try:
#     from xgboost import XGBRegressor
# except ModuleNotFoundError:
#     st.error("The 'xgboost' library is not installed. Please install it using 'pip install xgboost'.")

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

# Display user parameters with progress bar
st.write("---")
st.header("Your Parameters:")
latest_iteration = st.empty()
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Calculate BMI
height_m = height / 100  # convert cm to meters
bmi = weight / (height_m ** 2)
st.write(f"**BMI:** {bmi:.2f}")

# Define relative paths for CSV files
calories_path = "calories.csv"
exercise_path = "exercise.csv"

# # Check if files exist
# if not os.path.exists(calories_path):
#     st.error(f"The file {calories_path} does not exist. Please provide the correct path.")
#     st.stop()

# if not os.path.exists(exercise_path):
#     st.error(f"The file {exercise_path} does not exist. Please provide the correct path.")
#     st.stop()

# # Load dataset
# calories = "calories.csv"
# exercise = "exercise.csv"

exercise_df = exercise_path.merge(calories_path, on="User_ID")
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
try:
    xgb_model = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)
except Exception as e:
    st.error(f"An error occurred while training the model: {e}")
    st.stop()

# Prediction on user input with progress bar
st.write("---")
st.header("Prediction:")
latest_iteration = st.empty()
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

try:
    prediction = xgb_model.predict(df)
    st.write(round(prediction[0], 2), " **kilocalories**")
except Exception as e:
    st.error(f"An error occurred while predicting: {e}")
    st.stop()

# Display similar results with progress bar
st.write("---")
st.header("Similar Results:")
latest_iteration = st.empty()
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

prediction_range = [prediction[0] - 10, prediction[0] + 10]
similar_results = exercise_df[(exercise_df["Calories"] >= prediction_range[0]) & (exercise_df["Calories"] <= prediction_range[-1])]
st.write(similar_results.sample(5))

# General information
st.write("---")
st.header("General Information:")

boolean_age = (exercise_df["Age"] < age).tolist()
boolean_weight = (exercise_df["Weight"] < weight).tolist()
boolean_height = (exercise_df["Height"] < height).tolist()
boolean_duration = (exercise_df["Duration"] < duration).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < body_temp).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < heart_rate).tolist()

st.write("You are older than ", round(sum(boolean_age) / len(boolean_age) * 100, 2), "% of other people.")
st.write("Your weight is higher than ", round(sum(boolean_weight) / len(boolean_weight) * 100, 2), "% of other people.")
st.write("Your exercise duration is longer than ", round(sum(boolean_duration) / len(boolean_duration) * 100, 2), "% of other people.")
st.write("Your heart rate is higher than ", round(sum(boolean_heart_rate) / len(boolean_heart_rate) * 100, 2), "% of other people during exercise.")
st.write("Your body temperature is higher than ", round(sum(boolean_body_temp) / len(boolean_body_temp) * 100, 2), "% of other people during exercise.")
