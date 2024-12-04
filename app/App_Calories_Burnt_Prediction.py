import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Cache data loading to avoid reloading datasets on every rerun
@st.cache_data
def load_data():
    calories_url = "https://raw.githubusercontent.com/Teja-Sanaka/UMBC-DATA606-Capstone/refs/heads/main/app/calories.csv"
    exercise_url = "https://raw.githubusercontent.com/Teja-Sanaka/UMBC-DATA606-Capstone/refs/heads/main/app/exercise.csv"
    calories = pd.read_csv(calories_url)
    exercise = pd.read_csv(exercise_url)
    merged = exercise.merge(calories, on="User_ID")
    merged.drop(columns="User_ID", inplace=True)
    return merged

# Cache model training to avoid re-training on every rerun
@st.cache_resource
def train_model(data):
    train_data, _ = train_test_split(data, test_size=0.2, random_state=1)
    train_data = train_data[["Gender", "Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    train_data = pd.get_dummies(train_data, drop_first=True)
    X_train = train_data.drop("Calories", axis=1)
    y_train = train_data["Calories"]

    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

# Load datasets and train the model
exercise_data = load_data()
xgb_model = train_model(exercise_data)

# App UI
st.write("## Calories Burnt Prediction")
st.write("""
Predict calories burned based on personal parameters such as Age, Gender, Weight, Height, 
Duration, Heart Rate, and Body Temperature.
""")

st.sidebar.header("User Input Parameters:")
def user_input_features():
    age = st.sidebar.slider("Age:", 10, 100, 30)
    weight = st.sidebar.slider("Weight (kg):", 30, 200, 60)
    height = st.sidebar.slider("Height (cm):", 40, 200, 150)
    duration = st.sidebar.slider("Duration (min):", 0, 120, 15)
    heart_rate = st.sidebar.slider("Heart Rate:", 50, 150, 80)
    body_temp = st.sidebar.slider("Body Temperature (C):", 35, 50, 38)
    gender = st.sidebar.radio("Gender:", ("Male(0)", "Female(1)"))
    gender_encoded = 0 if gender == "Male(0)" else 1
    data = {
        "Age": age,
        "Weight": weight,
        "Height": height,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": 1 if gender_encoded == 0 else 0
    }
    return pd.DataFrame(data, index=[0])

# Input Features
df = user_input_features()

# Display User Parameters
st.write("---")
st.write("### Your Parameters")
st.write(df)

# Predict Calories Burned
st.write("---")
st.write("### Prediction")
prediction = xgb_model.predict(df)
st.write(f"**Calories Burned:** {round(prediction[0], 2)} kilocalories")

# Display Similar Results
st.write("---")
st.write("### Similar Results")
prediction_range = [prediction[0] - 10, prediction[0] + 10]
similar_results = exercise_data[
    (exercise_data["Calories"] >= prediction_range[0]) &
    (exercise_data["Calories"] <= prediction_range[1])
]
st.write(similar_results.sample(min(5, len(similar_results))))

# General Information
st.write("---")
st.write("### General Information")
age, weight, height, duration, heart_rate, body_temp = df.iloc[0][["Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp"]]
stats = {
    "Age": (exercise_data["Age"] < age).mean(),
    "Weight": (exercise_data["Weight"] < weight).mean(),
    "Duration": (exercise_data["Duration"] < duration).mean(),
    "Heart Rate": (exercise_data["Heart_Rate"] < heart_rate).mean(),
    "Body Temperature": (exercise_data["Body_Temp"] < body_temp).mean()
}

for stat, value in stats.items():
    st.write(f"Your {stat.lower()} is higher than **{value:.1%}** of others.")
