import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate example data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Polynomial Regression
degree = 3  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Predictions
y_pred_linear = linear_reg.predict(X_test)
y_pred_poly = poly_reg.predict(X_test_poly)

# Calculate Mean Squared Error
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# Plot the results
plt.scatter(X, y, s=20, label="Data")
plt.plot(X_test, y_pred_linear, color='r', label=f'Linear Reg. (MSE={mse_linear:.2f})')
plt.plot(X_test, y_pred_poly, color='g', label=f'Poly Reg. (MSE={mse_poly:.2f})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
