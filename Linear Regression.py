# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Simulated data for chemical cost and weight
chemical_data = {
    'Chemical1': {'cost': 10, 'weight': 5},
    'Chemical2': {'cost': 15, 'weight': 8},
    'Chemical3': {'cost': 20, 'weight': 10},
    'Chemical4': {'cost': 25, 'weight': 12},
    'Chemical5': {'cost': 30, 'weight': 15}
}

# Extracting features (weight) and target variable (cost)
X = np.array([data['weight'] for data in chemical_data.values()]).reshape(-1, 1)
y = np.array([data['cost'] for data in chemical_data.values()])

# Linear Regression
lr = LinearRegression()
lr.fit(X, y)

# Make predictions
y_pred = lr.predict(X)

# Calculate Mean Squared Error
mse = mean_squared_error(y, y_pred)

print("Mean Squared Error:", mse)

# Plotting actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
plt.xlabel('Actual Cost')
plt.ylabel('Predicted Cost')
plt.title('Actual vs Predicted Cost')
plt.show()