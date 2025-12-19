📊 A Comparative Analysis of Machine Learning Models for Sales Forecasting
🔍 Project Overview

Sales forecasting is a critical task in retail management, helping businesses optimize inventory, staffing, and strategic planning.
This project performs a comparative analysis of machine learning models to predict weekly sales using the Walmart Sales dataset.

Three popular regression models are implemented and evaluated:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

🎯 Objectives

Predict weekly sales accurately using machine learning

Compare multiple regression models

Evaluate models using standard performance metrics

Identify the best-performing model for sales forecasting

📦 Dataset Description

The dataset contains historical Walmart sales data with the following features:

Feature	Description
Store	Store number
Date	Weekly sales date
Holiday_Flag	Indicates holiday week
Temperature	Temperature in the region
Unemployment	Unemployment rate
Weekly_Sales	Target variable

Additional features were engineered:

Month

Year

⚙️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

🧠 Machine Learning Models

Linear Regression – Baseline statistical model

Decision Tree Regressor – Captures non-linear patterns

Random Forest Regressor – Ensemble model for improved accuracy

📈 Evaluation Metrics

Models were evaluated using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Accuracy (%) = R² × 100

🏆 Results Summary
Model	MAE	RMSE	R² Score	Accuracy (%)
Random Forest	Best	Lowest	Highest	⭐ Best
Decision Tree	Moderate	Moderate	Moderate	Good
Linear Regression	Higher	Higher	Lower	Baseline

👉 Random Forest Regressor achieved the highest accuracy, making it the most effective model for this dataset.
