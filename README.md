# Loan Default Prediction using Gradient Boosting

## Project Overview
This project involves building a predictive model using the **Gradient Boosting Classifier** to predict loan defaults. The model was trained on a loan default dataset, following a structured approach to data preprocessing, exploratory data analysis (EDA), model building, evaluation, and feature importance analysis.

## Workflow
The project follows these key steps:
1. **Data Understanding and Preprocessing**:
   - The dataset is inspected for missing values and basic statistics.
   - Missing values are imputed (mean for numeric and most frequent for categorical columns).
   - Numeric features are scaled, and categorical features are one-hot encoded.

2. **Exploratory Data Analysis (EDA)**:
   - Correlation heatmaps are generated to visualize relationships between numerical features.
   - Histograms are plotted to show the distribution of numeric features.

3. **Model Building**:
   - The data is split into training and testing sets (80-20 split).
   - A **Gradient Boosting Classifier** is built within a pipeline that handles preprocessing.
   - **GridSearchCV** is used to perform hyperparameter tuning.

4. **Model Evaluation**:
   - The model is evaluated using accuracy, precision, recall, F1 score, and ROC-AUC on the test set.
   - Predictions and evaluation metrics are calculated and displayed.

5. **Feature Importance**:
   - The importance of each feature is calculated and visualized in a bar plot, providing insights into which features are most influential in predicting loan defaults.

## Setup and Installation
To run this project, you need the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install the dependencies using:
pip install -r requirements.txt


## Usage

1. **Clone this repository** and navigate to the project directory.
2. Place your dataset (`Loan_default.csv`) in the directory.
3. Run the `SK.py` script to preprocess the data, build the model, perform hyperparameter tuning, and evaluate the model.

## Hyperparameters and Model Tuning
- **Hyperparameter Tuning** was performed using **GridSearchCV**, testing different values for the number of estimators, learning rate, and maximum depth.
- The best hyperparameters found during the grid search are used to build the final model.

## Results
- The model is evaluated on the test set, and metrics such as accuracy, precision, recall, F1 score, and ROC-AUC are displayed.
- Feature importance is analyzed, providing insights into the most significant predictors of loan default.

## Example Output

Best Hyperparameters: {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3} Test Accuracy: 0.85 Precision: 0.87 Recall: 0.81 F1 Score: 0.84 ROC-AUC Score: 0.91
