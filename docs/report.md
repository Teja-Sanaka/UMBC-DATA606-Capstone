
# Calories Burnt Prediction - Machine Learning Project

## 1. Problem Overview
The objective of this project is to predict the number of calories burnt by individuals during exercise based on several factors such as age, gender, and exercise details (e.g., heart rate, duration). Using machine learning, we aim to build a model that accurately predicts calorie expenditure from input features.

---

## 2. Data Overview

### a. **Calories Dataset**:
- Contains two columns:
  - `User_ID`: Unique identifier for each user.
  - `Calories`: The number of calories burnt by the user during the exercise session.

### b. **Exercise Dataset**:
- Contains features related to user demographics and exercise metrics:
  - `User_ID`: Unique identifier for each user.
  - `Gender`: Male or female.
  - `Age`: The age of the individual in years.
  - `Height`: User's height in centimeters.
  - `Weight`: User's weight in kilograms.
  - `Duration`: Duration of the exercise session in minutes.
  - `Heart_Rate`: The average heart rate during the exercise.
  - `Body_Temp`: Body temperature during the session (in degrees Celsius).

### c. **Data Preprocessing**:
- Merging the `calories.csv` and `exercise.csv` datasets based on the `User_ID`.
- Encoding categorical variables (`Gender`), and scaling numerical features for consistency in model training.
  
---

## 3. Exploratory Data Analysis (EDA)

### a. **Distribution of Calories Burnt**:
A histogram to visualize the distribution of the target variable, `Calories`. This plot shows whether the calories burnt are normally distributed or skewed.

![Calories Burnt Distribution](./calories_distribution.png)

### b. **Pairplot**:
This helps us to visualize the distribution of every attribute with the other
![Pairplot](./pairplot.png)

### c. **Correlation Matrix**:
A heatmap of correlations between variables helps identify which features are most strongly related to the target variable (`Calories`). Features with high correlation, such as `Duration` and `Heart_Rate`, are expected to play a significant role in prediction.

![Correlation Matrix](./correlation_matrix.png)

### d. **Boxplot of Numerical Features**:
A boxplot for all numerical features (`Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`, `Calories`) helps in detecting outliers and understanding the spread of data across these variables.

![Boxplot of Numerical Features](./boxplot_numerical_features.png)


---

## 4. Data Splitting
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
