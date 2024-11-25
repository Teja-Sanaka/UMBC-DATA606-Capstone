
# coding: utf-8

# In[26]:


# !pip install streamlit


# In[25]:


# !pip install plotly


# In[1]:


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

from matplotlib import style
style.use("seaborn")
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

st.write("## Calories burned Prediction")
st.image("https://assets.considerable.com/wp-content/uploads/2019/07/03093250/ExerciseRegimenPano.jpg" , use_column_width=True)
st.write("In this WebApp you will be able to observe your predicted calories burned in your body.Only thing you have to do is pass your parameters such as `Age` , `Gender` , `BMI` , etc into this WebApp and then you will be able to see the predicted value of kilocalories that burned in your body.")


st.sidebar.header("User Input Parameters : ")

def user_input_features():
    global age , bmi , duration , heart_rate , body_temp
    age = st.sidebar.slider("Age : " , 10 , 100 , 30)
    bmi = st.sidebar.slider("BMI : " , 15 , 40 , 20)
    duration = st.sidebar.slider("Duration (min) : " , 0 , 35 , 15)
    heart_rate = st.sidebar.slider("Heart Rate : " , 60 , 130 , 80)
    body_temp = st.sidebar.slider("Body Temperature (C) : " , 36 , 42 , 38)
    gender_button = st.sidebar.radio("Gender : ", ("Male" , "Female"))

    if gender_button == "Male":
        gender = 1
    else:
        gender = 0

    data = {
    "age" : age,
    "bmi" : bmi,
    "duration" : duration,
    "heart_rate" : heart_rate,
    "body_temp" : body_temp,
    "gender" : ["Male" if gender_button == "Male" else "Female"]
    }

    data_model = {
    "age" : age,
    "bmi" : bmi,
    "duration" : duration,
    "heart_rate" : heart_rate,
    "body_temp" : body_temp,
    "gender" : gender
    }

    features = pd.DataFrame(data_model, index=[0])
    data = pd.DataFrame(data, index=[0])
    return features , data

df , data = user_input_features()

st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)
st.write(data)

calories = pd.read_csv("C:/Users/teja/Desktop/Data 606/calories.csv")
exercise = pd.read_csv("C:/Users/teja/Desktop/Data 606/exercise.csv")

exercise_df = exercise.merge(calories , on = "User_ID")
# st.write(exercise_df.head())
exercise_df.drop(columns = "User_ID" , inplace = True)

exercise_train_data , exercise_test_data = train_test_split(exercise_df , test_size = 0.2 , random_state = 1)

for data in [exercise_train_data , exercise_test_data]:         # adding BMI column to both training and test sets
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"] , 2)

exercise_train_data = exercise_train_data[["Gender" , "Age" , "BMI" , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_test_data = exercise_test_data[["Gender" , "Age" , "BMI"  , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first = True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first = True)

X_train = exercise_train_data.drop("Calories" , axis = 1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories" , axis = 1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators = 1000 , max_features = 3 , max_depth = 6)
random_reg.fit(X_train , y_train)
random_reg_prediction = random_reg.predict(X_test)

#st.write("RandomForest Mean Absolute Error(MAE) : " , round(metrics.mean_absolute_error(y_test , random_reg_prediction) , 2))
#st.write("RandomForest Mean Squared Error(MSE) : " , round(metrics.mean_squared_error(y_test , random_reg_prediction) , 2))
#st.write("RandomForest Root Mean Squared Error(RMSE) : " , round(np.sqrt(metrics.mean_squared_error(y_test , random_reg_prediction)) , 2))
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

st.write(round(prediction[0] , 2) , "   **kilocalories**")

st.write("---")
st.header("Similar Results : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

# range = [prediction[0] - 10 , prediction[0] + 10]
prediction_range = [prediction[0] - 10 , prediction[0] + 10]
ds = exercise_df[(exercise_df["Calories"] >= prediction_range[0]) & (exercise_df["Calories"] <= prediction_range[-1])]

# ds = exercise_df[(exercise_df["Calories"] >= range[0]) & (exercise_df["Calories"] <= range[-1])]
st.write(ds.sample(5))

st.write("---")
st.header("General Information : ")

boolean_age = (exercise_df["Age"] < age).tolist()
boolean_duration = (exercise_df["Duration"] < duration).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < body_temp).tolist()
boolean_heart_rate= (exercise_df["Heart_Rate"] < heart_rate).tolist()

st.write("You are older than %" , round(sum(boolean_age) / len(boolean_age) , 2) * 100 , "of other people.")
st.write("Your had higher exercise duration than %" , round(sum(boolean_duration) / len(boolean_duration) , 2) * 100 , "of other people.")
st.write("You had more heart rate than %" , round(sum(boolean_heart_rate) / len(boolean_heart_rate) , 2) * 100 , "of other people during exercise.")
st.write("You had higher body temperature  than %" , round(sum(boolean_body_temp) / len(boolean_body_temp) , 2) * 100 , "of other people during exercise.")


# In[24]:


# streamlit run C:\Users\teja\Anaconda3\lib\site-packages\ipykernel_launcher.py


# In[28]:


# !pip install --upgrade numpy


# In[31]:


# !pip install --upgrade pip


# In[29]:


# !pip install --upgrade --force-reinstall pyarrow


# In[24]:


# import numpy as np

# # Adding noise to the numerical columns (Duration, Age, Weight, Height, etc.)
# # Define the level of noise (percentage of original value)
# noise_level = 0.40 # Adjust this value to increase/decrease noise

# # Apply noise to selected numerical columns
# numerical_cols = ['Duration', 'Age', 'Weight', 'Height', 'Heart_Rate', 'Body_Temp']

# # Adding noise by randomly perturbing the values
# for col in numerical_cols:
#     noise = np.random.normal(loc=0, scale=noise_level * calories_data[col].std(), size=calories_data[col].shape)
#     calories_data[col] = calories_data[col] + noise

# # Display the head of the DataFrame to see the noise-added data
# calories_data.head()


# In[2]:


# import os

# notebook_path = os.getcwd()
# print(notebook_path)


# In[3]:


# import os
# current_directory = os.getcwd()
# print(current_directory)

