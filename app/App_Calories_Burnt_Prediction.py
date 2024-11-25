import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
import time

warnings.filterwarnings("ignore")

# App Title and Description
st.title("Calories Burnt Prediction App")
st.markdown("""
This app predicts the number of calories burned based on personal and exercise parameters.
""")

# Sidebar for User Input Parameters
st.sidebar.header("User Input Parameters:")


def user_input_features():
    age = st.sidebar.slider("Age:", 10, 100, 30)
    weight = st.sidebar.slider("Weight (kg):", 30, 200, 60)
    height = st.sidebar.slider("Height (cm):", 40, 200, 150)
    duration = st.sidebar.slider("Duration (min):", 0, 120, 15)
    heart_rate = st.sidebar.slider("Heart Rate:", 50, 150, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C):", 35, 50, 38)
    gender = st.sidebar.radio("Gender:", ["Male", "Female"])
    gender_binary = 0 if gender == "Male" else 1

    data = {
        "Age": age,
        "Weight": weight,
        "Height": height,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender": gender_binary,
    }
    return pd.DataFrame(data, index=[0])


# File Upload Section
st.sidebar.subheader("Upload Your Data Files:")

uploaded_calories = st.sidebar.file_uploader("Upload `calories.csv`", type=["csv"])
uploaded_exercise = st.sidebar.file_uploader("Upload `exercise.csv`", type=["csv"])


@st.cache_data
def load_data(calories_file, exercise_file):
    try:
        calories = pd.read_csv(calories_file)
        exercise = pd.read_csv(exercise_file)
        return exercise.merge(calories, on="User_ID")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Preprocess Data
@st.cache_data
def preprocess_data(data):
    data = data.drop(columns=["User_ID"], errors="ignore")
    data = pd.get_dummies(data, columns=["Gender"], drop_first=True)
    return data


# Model Training
@st.cache_resource
def train_model(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
    X_train = train_data.drop("Calories", axis=1)
    y_train = train_data["Calories"]

    model = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model, train_data


# Main Application Logic
if uploaded_calories and uploaded_exercise:
    data = load_data(uploaded_calories, uploaded_exercise)

    if data is not None:
        # Preprocess the data
        processed_data = preprocess_data(data)

        # Train the model
        xgb_model, train_data = train_model(processed_data)

        # User Input
        user_data = user_input_features()

        # Prediction
        prediction = xgb_model.predict(user_data)[0]
        st.subheader("Predicted Calories Burnt:")
        st.write(f"**{prediction:.2f} kilocalories**")

        # Display Similar Results
        prediction_range = [prediction - 10, prediction + 10]
        similar_results = processed_data[
            (processed_data["Calories"] >= prediction_range[0]) &
            (processed_data["Calories"] <= prediction_range[1])
        ]
        st.subheader("Similar Records from Dataset:")
        st.write(similar_results.sample(min(len(similar_results), 5)))

        # Additional Insights
        st.subheader("Additional Insights:")
        bmi = user_data["Weight"].iloc[0] / ((user_data["Height"].iloc[0] / 100) ** 2)
        st.write(f"**Your BMI:** {bmi:.2f}")
    else:
        st.stop()
else:
    st.sidebar.warning("Please upload both `calories.csv` and `exercise.csv` to proceed.")
    st.stop()

