# 1.Project Title: Calories Burnt Prediction

- *Author Name:* Teja Sanaka
- *Semester:* Fall'24
- *Prepared for:* UMBC Data Science Master's Degree Capstone by Dr. Chaojie (Jay) Wang
- *GitHub:* [https://github.com/Teja-Sanaka/UMBC-DATA606-Capstone](https://github.com/Teja-Sanaka/UMBC-DATA606-Capstone)
- *LinkedIn profile:* [https://www.linkedin.com/in/teja-sanaka-6598771ba/](https://www.linkedin.com/in/teja-sanaka-6598771ba/)

# 2.Background

**What is the project about?**

This project is about predicting how many calories are burned during exercise based on factors like how long the exercise lasts, what kind of exercise it is, and personal details like age, weight, and gender. The goal is to use machine learning to better understand how these factors affect calorie burning.

**Why is it important?**

Knowing how many calories are burned is important for staying healthy, managing weight, and improving fitness. Many people want to meet their fitness goals, whether it's losing weight, building muscle, or staying healthy. Accurately predicting calorie burn helps people customize their exercise plans, eat better, and improve their training.

This research can also be useful for fitness trainers, individuals, and companies that create fitness devices or apps. Better calorie predictions can help them give better advice for reaching health goals.

**What questions are we trying to answer?**

1. **What are the most important factors that affect calorie burning during exercise?**

2. **- Can we accurately predict calories burnt using exercise and personal data?**
    
3. **Which machine learning model is best for predicting calorie burn?**

**#3. Data**

**Datasets**

The datasets used in this project focus on exercise data and calorie expenditure. These datasets will help answer research questions about predicting calorie burn during different types of physical activities.

- **Data Sources**: The datasets include "exercise.csv" and "calories.csv" files, which provide the necessary information for building a calorie prediction model based on exercise details.
  
- **Data Size**: 
  - **exercise.csv**: [662 KB]
  - **calories.csv**: [225 KB]
  
- **Data Shape**: 
  - **exercise.csv**: 15000 rows and 8 columns
  - **calories.csv**: 15000 rows and 2 columns

- **Time Period**: The datasets do not include explicit time-bound data.

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
| **Calories Burned**   | Numerical      | The number of calories burned after the exercise   | Numeric values (target variable)    |
| **Body Temperature**   | Numerical      | The body temperature after the exercise   | Numeric values     |
| **Heart rate**   | Numerical      | The heart rate after the exercise   | Numeric values     |

**Target/Label**:
- **Target Variable**: The target variable in the machine learning model will be **"Calories Burned"**. This is the output the model will predict based on the input features.

**Features/Predictors**:
- The following columns may be selected as features/predictors in the machine learning model:
  - **Duration (mins)**
  - **Age**
  - **Weight (kg)**
  - **Height (cm)**
  - **Gender**

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
    - **Heart_Rate**: Participant's heart rate after the exercise (numerical)
    - **Body_Temp**: Participant's body temperature after the exercise (numerical)


 **Data Preprocessing**:
- Merging the `calories.csv` and `exercise.csv` datasets based on the `User_ID`.
- Encoding categorical variables (`Gender`), and scaling numerical features for consistency in model training.
  
---

## 4. Exploratory Data Analysis (EDA)

### a. **Distribution of Calories Burnt**:
A histogram to visualize the distribution of the target variable, `Calories`. This plot shows whether the calories burnt are normally distributed or skewed.

![Calories Burnt Distribution](./calories_distribution.png)

### b. **Pairplot**:
This helps us to visualize the distribution of every attribute with the other
![Pairplot](./pairplot.png)

### c. **Correlation Matrix**:
A heatmap of correlations between variables helps identify which features are most strongly related to the target variable (`Calories`). Features with high correlation, such as `Duration` and `Heart_Rate`, are expected to play a significant role in prediction.

![Correlation Matrix](./correlation_matrix.png)



## 5. Model Training
Before building machine learning models, the dataset is split into training and testing sets. Typically, an 80-20 split is used, where 80% of the data is used for training the model and 20% is held back for testing the model's performance.

```python
from sklearn.model_selection import train_test_split

# Define the features (X) and the target variable (y)
X = data.drop(columns=["Calories"])  # Assuming 'Calories' is the target column
y = data["Calories"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
```

---


## 5. Model: Linear Regression

### Model Fitting
A linear regression model was trained on the dataset. The R² score was used as a measure of how well the model's predictions match the actual data.

- **Model**: Linear Regression
- **Evaluation Metric**: R-squared (R²)

###  Performance Metrics

#### Training Data
- **Training R²**: 0.864  
This value indicates that the model explains 86.4% of the variance in the training data.

#### Testing Data
- **Testing R²**: 0.859  
This value shows that the model generalizes well to unseen data, explaining 85.9% of the variance in the test set.

### Observations
- The close similarity between the training and test R² scores (0.864 and 0.859, respectively) suggests that the model is not overfitting. It performs consistently on both the training and test datasets, indicating a good balance between bias and variance.

---

## Conclusion
The linear regression model has demonstrated strong performance on the dataset, achieving an R² score of 0.864 on the training data and 0.859 on the test data. These metrics suggest that the model is reliable and has good predictive power with minimal overfitting.

### Recommendations
To further improve the model:
- I would like to consider testing other models which can help handle potential multicollinearity in the dataset.
- I would like to explore feature engineering to enhance the model's performance by adding meaningful transformations or interactions between variables.

---

## 5. Future Work
In future iterations, I would like to explore advanced models, or more extensive feature selection to optimize performance and reduce residual prediction error and also make use of streamlit library for creating an interface for the project

---

## 6. Code Snippet
Below is the key code snippet used to calculate the R² score for both training and test sets:

```python
from sklearn.metrics import r2_score

# R-squared score for the test set
linear_acc_test = r2_score(linear_result_test, Y_test)
print("Testing R²:", linear_acc_test)

# R-squared score for the training set
linear_acc_train = r2_score(linear_result_train, Y_train)
print("Training R²:", linear_acc_train)
```
