## Overview of the Analysis

In this analysis, the aim is to build a machine learning model to predict the risk of loan defaults using logistic regression. The financial information in the dataset includes various features related to lending and borrowing activities, and the goal was to predict the `loan_status`, where a value of 0 indicates a healthy loan and a value of 1 indicates a high-risk loan.

I utilized the `lending_data.csv` file, which contains historical loan data. Our target variable was `loan_status`, and the features included various financial metrics. Here are some basic statistics about the `loan_status` variable:

```python
import pandas as pd
from pathlib import Path

# Load the data
file_path = Path('Resources/lending_data.csv')
data = pd.read_csv(file_path)

# Display value counts for the loan_status
value_counts = data['loan_status'].value_counts()
print(value_counts)

The machine learning process involved the following stages:

Data Loading: Read the CSV file into a Pandas DataFrame.
Data Preparation: Split the data into features (X) and labels (y).
Data Splitting: Split the data into training and testing sets using train_test_split.
Model Training: Fit a logistic regression model using the training data.
Model Prediction: Make predictions on the testing data.
Model Evaluation: Evaluate the model's performance using a confusion matrix and classification report.
I used the LogisticRegression algorithm from the sklearn library to build our model.

## Results
Logistic Regression Model:
Accuracy: 0.99 (99%)
Precision: 0.99 (99%)
Recall: 0.99 (99%)

## Summary
The logistic regression model performed exceptionally well, achieving an accuracy, precision, and recall score of 0.99 (99%) across all metrics. This indicates that the model is highly effective at predicting both healthy loans (0) and high-risk loans (1).

Given the balanced performance across accuracy, precision, and recall, the logistic regression model seems to perform the best. This high level of performance is crucial because it ensures that we correctly identify both healthy and high-risk loans, minimizing potential financial risks.

In this context, predicting high-risk loans (1) accurately is particularly important to mitigate financial losses. The model's high precision and recall for predicting high-risk loans make it a valuable tool for financial institutions.

Therefore, I recommend using the logistic regression model for predicting loan risks, given its strong performance metrics and balanced predictive capabilities.

