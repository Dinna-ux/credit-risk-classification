# credit-risk-classification
# Loan Risk Prediction Using Logistic Regression

## Overview of the Analysis

In this project, we aim to predict the risk of loan defaults using a logistic regression model. The dataset used contains various features related to lending and borrowing activities. The primary goal is to predict the `loan_status` of loans, where a value of 0 indicates a healthy loan and a value of 1 indicates a high-risk loan.

### Purpose of the Analysis

The purpose of this analysis is to build a machine learning model to identify high-risk loans. This can help financial institutions mitigate potential financial risks by accurately predicting loans that are likely to default.

### Financial Information in the Data

The data includes several financial metrics, such as the borrower's annual income, debt-to-income ratio, loan amount, and other relevant features that can impact loan status.

### Variables to Predict

The primary variable we aim to predict is the `loan_status`:
- 0: Healthy loan
- 1: High-risk loan

Here is a basic overview of the `loan_status` variable:

```python
import pandas as pd
from pathlib import Path

# Load the data
file_path = Path('Resources/lending_data.csv')
data = pd.read_csv(file_path)

# Display value counts for the loan_status
value_counts = data['loan_status'].value_counts()
print(value_counts)

##Stages of the Machine Learning Process
Data Loading: Read the CSV file into a Pandas DataFrame.
Data Preparation: Split the data into features (X) and labels (y).
Data Splitting: Split the data into training and testing sets using train_test_split.
Model Training: Fit a logistic regression model using the training data.
Model Prediction: Make predictions on the testing data.
Model Evaluation: Evaluate the model's performance using a confusion matrix and classification report.
Methods Used
The primary method used in this analysis is logistic regression, implemented using the LogisticRegression class from the sklearn library.

Results
The logistic regression model's performance is summarized as follows:

Accuracy: 0.99 (99%)
Precision: 0.99 (99%)
Recall: 0.99 (99%)

How to Run the Project
git clone <repository-url>
cd <project-directory>
pip install -r requirements.txt
jupyter notebook loan_risk_prediction.ipynb
python loan_risk_prediction.py

Files in the Repository
README.md: This file, providing an overview of the project.
Resources/lending_data.csv: The dataset used for the analysis.
loan_risk_prediction.ipynb: Jupyter notebook containing the full analysis.
loan_risk_prediction.py: Python script with the same analysis steps as the notebook.
requirements.txt: List of required Python packages.
Dependencies
Python 3.x
Pandas
Numpy
Scikit-learn
Pathlib

You can install the required packages using:
pip install -r requirements.txt

Contact
For any questions or comments, please reach out to [Dinna] at [witnessdinna@gmail.com].
