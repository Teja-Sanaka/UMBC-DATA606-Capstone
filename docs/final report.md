# Final Report: Calories Burnt Prediction

## 1. Title and Author
**Project Title**: Calories Burnt Prediction  
**Prepared for**: UMBC Data Science Master's Degree Capstone by Dr. Chaojie (Jay) Wang  
**Author**: Teja Sanaka GS40818  
- [GitHub Repository](https://github.com/Teja-Sanaka/UMBC-DATA606-Capstone)  
- [LinkedIn Profile](https://www.linkedin.com/in/teja-sanaka/)  
- [PowerPoint Presentation](file:///mnt/data/teja_ppt.pptx)  
- [YouTube Video](#)  

---

## 2. Background
**What is it about?**  
The project predicts calories burnt during physical activity based on personal and exercise-related parameters.  

**Why does it matter?**  
Understanding calorie expenditure is essential for fitness goals, weight management, and maintaining a healthy lifestyle.  

**Research Questions:**  
- Can we accurately predict calories burnt using exercise and personal data?  
- Which features contribute most to calorie prediction?  

---

## 3. Data
**Data Sources:** Kaggle  
- **Files**: `exercise.csv` (662 KB, 15,000 rows × 8 columns), `calories.csv` (225 KB, 15,000 rows × 2 columns).  
- **Time Period**: Not time-bound.  
- **Row Representation**: Each row represents an exercise session.  

**Data Dictionary:**  
- **Key Variables**:  
  - `Calories` (target): Number of calories burned.  
  - `Age`, `Weight`, `Height`, `Duration`, `Heart Rate`, `Body Temp`, `Gender` (predictors).  

---

## 4. Exploratory Data Analysis (EDA)
**Summary Statistics:**  
- Highlighted key variable distributions and correlations.  
- Target variable (`Calories`) shows variation based on exercise duration and intensity.  

**Key Visualizations:**  
- Distribution plots for features and target.  
- Correlation matrix: Strong correlations with `Duration`, `Heart Rate`, and `Body Temp`.  

**Data Preparation:**  
- Handled missing values and irrelevant columns.  
- Normalized numerical features; one-hot encoded categorical variables (`Gender`).  

---

## 5. Model Training
**Models Used:**  
- Linear Regression  
- Lasso Regression  
- Random Forest  
- XGBoost  

**Training Approach:**  
- Split data into 80% training and 20% testing.  
- Used cross-validation for parameter tuning.  

**Python Libraries:**  
- `pandas`, `numpy`: Data preprocessing.  
- `scikit-learn`: Model building and evaluation.  
- `xgboost`: Model training.  

**Development Environments:**  
- Google Colab and Jupyter Notebook.  

**Performance Metrics:**  
- XGBoost provided the best accuracy, handling feature interactions effectively.  

---

## 6. Application of the Trained Models
**Web App**: Built using Streamlit.  
**Features:**  
1. **Calorie Burn Prediction**: Users input personal details to receive predictions.  
2. **BMI Calculation**: Computes BMI based on user-provided weight and height.  
3. **Insights**: Compares user predictions with similar data points for better understanding.  

**Advantages:**  
- Intuitive user interface.  
- Real-time predictions with visual feedback.  

---

## 7. Conclusion
**Achievements:**  
- Developed a functional calorie prediction app using XGBoost and Streamlit.  
- Delivered a user-friendly interface to make predictions accessible.  

**Limitations:**  
- Dataset diversity is limited.  
- Model accuracy may vary for underrepresented groups.  

**Future Directions:**  
- Incorporate advanced models like neural networks.  
- Expand datasets with more varied demographic and activity data.  
- Integrate real-time wearable device data for enhanced predictions.  

---

## 8. References
- Kaggle Datasets: [Exercise and Calories](https://www.kaggle.com/datasets)  
- Python Libraries Documentation: `scikit-learn`, `xgboost`, `streamlit`  
- Academic Papers and Blogs on Feature Engineering and Predictive Modeling.  
