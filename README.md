# Diabetes Detection

## Type of problem : Binary Classification problem

## Dataset Description:

- Diabetes Dataset: This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.

- Description: This dataset consists of ten baseline variables, such as age, BMI (Body Mass Index), blood pressure, and six blood serum measurements, for diabetes patients.

- Purpose: Commonly used for predicting diabetes progression based on various health-related features.

## The proposed framework is summarized in the following steps:
1- Data exploration 🔎 : Check the number of features and rows, missing values, duplicate values and generating descriptive statistics for this data 
2- Data visualization 📊 : 
- use histplot from Seaborn library to see the distribution of data for each feature and try to reach insight from it.
- comparing each feature with the dependent feature and try to reach insight from it.
- Checking if there is high correlation between each attributes.
3- Data preprocessing 🛠 : Data Splitting ,Data Cleaning (Handling Missing Values & Outliers Values), Handling Imbalanced Data (Oversampling),Feature Engineering (Creating New Features), Handing Categorical Data(One Hot Encoding).
4- Selecting and comparing models 🎯 : After many experiments and attempts with most of the famous models in classification problems, I found that the Random Forest Classifier is the best model that was able to learn from this data and was able to predict the validate data. 
5- Fine tuning 🪄 : I used Optuna (optimization framework for machine learning models) to choose the best hyperparameters for the Random Forest Classifier that give me the best performance for the model.
6- Evaluation using test set ⚖ 
7- Saving model 🗃