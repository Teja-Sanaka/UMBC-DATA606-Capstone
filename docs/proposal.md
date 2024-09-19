# Project Title: Calories Burnt Prediction

- *Author Name:* Teja Sanaka
- *Semester:* Fall'24
- *Prepared for:* UMBC Data Science Master's Degree Capstone by Dr. Chaojie (Jay) Wang
- *GitHub:* [https://github.com/Teja-Sanaka/UMBC-DATA606-Capstone](https://github.com/Teja-Sanaka/UMBC-DATA606-Capstone)
- *LinkedIn profile:* [https://www.linkedin.com/in/teja-sanaka-6598771ba/](https://www.linkedin.com/in/teja-sanaka-6598771ba/)

- ### 2. Background

**What is the project about?**

This project is about predicting how many calories are burned during exercise based on factors like how long the exercise lasts, what kind of exercise it is, and personal details like age, weight, and gender. The goal is to use machine learning to better understand how these factors affect calorie burning.

**Why is it important?**

Knowing how many calories are burned is important for staying healthy, managing weight, and improving fitness. Many people want to meet their fitness goals, whether it's losing weight, building muscle, or staying healthy. Accurately predicting calorie burn helps people customize their exercise plans, eat better, and improve their training.

This research can also be useful for fitness trainers, individuals, and companies that create fitness devices or apps. Better calorie predictions can help them give better advice for reaching health goals.

**What questions are we trying to answer?**

1. **What are the most important factors that affect calorie burning during exercise?**
   - This question looks at which factors, like age, weight, or exercise type, have the biggest impact on calories burned.

2. **Can machine learning predict how many calories someone burns during exercise?**
   - This looks at whether machine learning models can accurately guess the number of calories burned.

3. **Which machine learning model is best for predicting calorie burn?**
   - This question compares different models to see which one works the best in terms of accuracy and speed.

4. **Does customizing the prediction based on personal details like age and weight improve accuracy?**
   - This looks at whether adding personal information helps make better predictions.




### 3. Data

**Datasets**

The datasets used in this project focus on exercise data and calorie expenditure. These datasets will help answer research questions about predicting calorie burn during different types of physical activities.

- **Data Sources**: The datasets include "exercise.csv" and "calories.csv" files, which provide the necessary information for building a calorie prediction model based on exercise details.
  
- **Data Size**: 
  - **exercise.csv**: [662 KB]
  - **calories.csv**: [225 KB]
  
- **Data Shape**: 
  - **exercise.csv**: 15000 rows and 8 columns
  - **calories.csv**: 15000 rows and 2 columns

- **Time Period**: The datasets do not include explicit time-bound data (no clear time periods provided from file names). If time data is available, further analysis may be needed.

- **Each Row Represents**: 
  - Each row likely represents a single exercise session or activity, performed by an individual participant, with details such as exercise duration, participant characteristics, and the corresponding calories burned.

**Data Dictionary**:  
The table below outlines the columns, data types, definitions, and potential values in the datasets.

| **Column Name**       | **Data Type**  | **Definition**                                      | **Potential Values**                 |
|-----------------------|----------------|----------------------------------------------------|--------------------------------------|
| **Duration (mins)**    | Numerical      | The duration of the exercise session in minutes     | Numeric values (e.g., 30, 60, etc.) |
| **Age**               | Numerical      | The age of the participant                          | Numeric values (e.g., 25, 40, etc.) |
| **Weight (kg)**        | Numerical      | The weight of the participant in kilograms          | Numeric values (e.g., 70, 85, etc.) |
| **Height (cm)**        | Numerical      | The height of the participant in centimeters        | Numeric values                      |
| **Gender**            | Categorical    | Gender of the participant                           | Male, Female                        |
| **Exercise Type**     | Categorical    | Type of exercise performed                          | Walking, Running, Cycling, etc.     |
| **Calories Burned**   | Numerical      | The number of calories burned during the exercise   | Numeric values (target variable)    |

**Target/Label**:
- **Target Variable**: The target variable in the machine learning model will be **"Calories Burned"**. This is the output the model will predict based on the input features.

**Features/Predictors**:
- The following columns may be selected as features/predictors in the machine learning model:
  - **Duration (mins)**
  - **Age**
  - **Weight (kg)**
  - **Height (cm)**
  - **Gender**
  - **Exercise Type**

These features will help the model learn the relationship between exercise activities and the number of calories burned.

**Dataset Details**:

- **calories.csv**:
  - **Shape**: 15,000 rows and 2 columns
  - **Columns**: 
    - **User_ID**: Unique identifier for the participant
    - **Calories**: The number of calories burned during the exercise session (target variable)

- **exercise.csv**:
  - **Shape**: 15,000 rows and 8 columns
  - **Columns**:
    - **User_ID**: Unique identifier for the participant
    - **Gender**: Gender of the participant (categorical: Male, Female)
    - **Age**: Age of the participant (numerical)
    - **Height**: Height of the participant in centimeters (numerical)
    - **Weight**: Weight of the participant in kilograms (numerical)
    - **Duration**: Duration of the exercise session in minutes (numerical)
    - **Heart_Rate**: Participant's heart rate during the exercise (numerical)
    - **Body_Temp**: Participant's body temperature during the exercise (numerical)

In this project, the target variable will be **Calories** from the "calories.csv" file, and potential features for the machine learning model will include:
- **Gender**
- **Age**
- **Height**
- **Weight**
- **Duration**
- **Heart_Rate**
- **Body_Temp**.

These features will be used to predict the number of calories burned for each exercise session.
```
