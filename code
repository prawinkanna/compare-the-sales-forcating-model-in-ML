#A Comparative Analysis of Machine Learning Models for Sales Forecasting
# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("Lets start our A Comparative Analysis of Machine Learning Models for Sales Forecasting")
#  Step 2: Load the Dataset
file_path = "/content/Walmart_Sales (1).csv"
data = pd.read_csv(file_path)

#  Step 3: Basic Info and First Look
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

#  Step 4: Preprocess the Date Column
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

#  Step 5: Define Features and Target (Dropped CPI and Fuel_Price)
feature_columns = ['Store', 'Holiday_Flag', 'Temperature', 'Unemployment', 'Month', 'Year']
target_column = 'Weekly_Sales'

X = data[feature_columns]
y = data[target_column]

#  Step 6: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 7: Define Machine Learning Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Step 8: Train Models and Evaluate Performance
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)                  # Train the model
    predictions = model.predict(X_test)          # Predict on test set

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Store results
    results[name] = {
        "Mean Absolute Error": mae,
        "Root Mean Squared Error": rmse,
        "R2 Score": r2
    }

#  Step 9: Create and Display Comparison Table
comparison_df = pd.DataFrame(results).T
comparison_df['Accuracy (%)'] = comparison_df['R2 Score'] * 100
comparison_df = comparison_df.round(2)
comparison_df = comparison_df.sort_values(by='Accuracy (%)', ascending=False).reset_index()
comparison_df.rename(columns={'index': 'Model'}, inplace=True)

print("\n Model Comparison Table:")
print(comparison_df)
#  Step 10: Visualize Model Accuracy using Bar Chart
import matplotlib.pyplot as plt

# Bar chart for model accuracy
plt.figure(figsize=(8,5))
plt.bar(comparison_df['Model'], comparison_df['Accuracy (%)'], color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Machine Learning Model')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)  # Optional: set y-axis limit to 100%
for i, v in enumerate(comparison_df['Accuracy (%)']):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')  # Display values on bars
plt.show()
