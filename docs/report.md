# Calories Burnt Prediction - Machine Learning Project

## 1. Problem Overview
The objective of this project is to predict the number of calories burnt by individuals during exercise based on a variety of features, including user demographics and exercise details. Using machine learning, we aim to build a model that accurately predicts calorie expenditure from input features.

## 2. Data Overview

### a. **Calories Dataset**:
- Contains two columns:
  - `User_ID`: Unique identifier for each user.
  - `Calories`: The number of calories burnt by the user.

### b. **Exercise Dataset**:
- Contains several columns related to user characteristics and exercise metrics:
  - `User_ID`: Unique identifier for each user.
  - `Gender`: Male or female.
  - `Age`: The age of the individual in years.
  - `Height`: User's height in centimeters.
  - `Weight`: User's weight in kilograms.
  - `Duration`: Duration of the exercise session in minutes.
  - `Heart_Rate`: The average heart rate during the exercise session.
  - `Body_Temp`: Body temperature during the session (in degrees Celsius).

## 3. Data Preprocessing

### a. **Data Merging**:
- The `calories.csv` and `exercise.csv` datasets are merged based on the `User_ID` column to form a complete dataset containing both user characteristics and calorie data.

### b. **Handling Missing Values**:
- Ensure there are no missing values in critical columns like `Duration`, `Heart_Rate`, or `Calories`. Any missing values are either imputed or removed depending on the context.

### c. **Encoding Categorical Variables**:
- The `Gender` column is converted from categorical to numerical format (0 for Female, 1 for Male).

### d. **Feature Scaling**:
- Features like `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, and `Body_Temp` may need to be scaled using either normalization or standardization for models like linear regression to ensure uniformity.

## 4. Exploratory Data Analysis (EDA)

### a. **Distribution of Calories Burnt**
A histogram or distribution plot helps visualize the distribution of the target variable, `Calories`. This plot can show whether the calories burnt are normally distributed, skewed, or have outliers.

<img src="Distribution of Calories Burned.png" />

- You can create this plot using:
  ```python
  import seaborn as sns
  sns.histplot(calories_df['Calories'], kde=True)
