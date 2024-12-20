import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up the app
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

# Function to handle user input
def user_input_features():
    age = st.sidebar.slider("Age:", 10, 100, 30)
    weight = st.sidebar.slider("Weight (kg):", 30, 200, 60)
    height = st.sidebar.slider("Height (cm):", 40, 200, 150)
    duration = st.sidebar.slider("Duration (min):", 0, 120, 15)
    heart_rate = st.sidebar.slider("Heart Rate:", 50, 150, 80)
    body_temp = st.sidebar.slider("Body Temperature (C):", 35, 50, 38)
    gender_button = st.sidebar.radio("Gender:", ("Male(0)", "Female(1)"))

    gender = 0 if gender_button == "Male(0)" else 1

    # Return the features in the format the model expects
    data_model = {
        "Age": age,
        "Weight": weight,
        "Height": height,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": 1 if gender == 0 else 0  # Encoding 'Male' as 1 and 'Female' as 0
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

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
height_m = df['Height'][0] / 100  # convert cm to meters
bmi = df['Weight'][0] / (height_m ** 2)
st.write(f"**BMI:** {bmi:.2f}")

# Load datasets from GitHub
calories_url = "https://raw.githubusercontent.com/Teja-Sanaka/UMBC-DATA606-Capstone/refs/heads/main/app/calories.csv"
exercise_url = "https://raw.githubusercontent.com/Teja-Sanaka/UMBC-DATA606-Capstone/refs/heads/main/app/exercise.csv"

# Load the datasets
calories = pd.read_csv(calories_url)
exercise = pd.read_csv(exercise_url)

# Merge datasets on 'User_ID'
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# Train-test split
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
exercise_train_data = exercise_train_data[["Gender", "Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

# One-hot encoding for Gender
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Prepare features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the XGBoost model
try:
    from xgboost import XGBRegressor
    xgb_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
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

# Prediction
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

boolean_age = (exercise_df["Age"] < df["Age"][0]).tolist()
boolean_weight = (exercise_df["Weight"] < df["Weight"][0]).tolist()
boolean_height = (exercise_df["Height"] < df["Height"][0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"][0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"][0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"][0]).tolist()

st.write("You are older than ", round(sum(boolean_age) / len(boolean_age) * 100, 2), "% of other people.")
st.write("Your weight is higher than ", round(sum(boolean_weight) / len(boolean_weight) * 100, 2), "% of other people.")
st.write("Your exercise duration is longer than ", round(sum(boolean_duration) / len(boolean_duration) * 100, 2), "% of other people.")
st.write("Your heart rate is higher than ", round(sum(boolean_heart_rate) / len(boolean_heart_rate) * 100, 2), "% of other people during exercise.")
st.write("Your body temperature is higher than ", round(sum(boolean_body_temp) / len(boolean_body_temp) * 100, 2), "% of other people during exercise.")
