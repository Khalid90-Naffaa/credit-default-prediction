# 💳 Credit Default Prediction (Random Forest)

A practical Machine Learning project designed to analyze customer behavior and predict the probability of credit default.

## 💡 The Core Idea
Instead of manual reviews, this project uses historical data (such as income, age, and family status) to build a predictive model. This helps financial institutions make data-driven decisions and identify high-risk applicants before approving loans.

## 🤖 Algorithm: Random Forest
The **Random Forest Classifier** was chosen for this project because:
* **High Accuracy:** It combines the results of hundreds of "Decision Trees" to reach the most reliable prediction.
* **Robustness:** It handles missing values and complex tabular data exceptionally well.
* **Feature Importance:** It allows us to identify exactly which 5 factors (features) influenced the decision most, making the model transparent and explainable.

## 💻 Logic & Execution
The script follows a clean 3-step workflow:
1. **Preprocessing:** Cleaning the data and encoding categorical text into a format the model can understand.
2. **Training:** Splitting the dataset to train the algorithm and then testing it on new data to ensure reliability.
3. **Analysis:** Outputting the final **Accuracy Score (approx. 92%)** and a list of the **Top 5 most important features**.
