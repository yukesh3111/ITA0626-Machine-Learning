import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the sales dataset (replace with your dataset)
sales_data = pd.read_csv('sales_data.csv')
sales_data['Date'] = pd.to_datetime(sales_data['Date'])
sales_data.set_index('Date', inplace=True)

# Resample the data to daily frequency
daily_sales = sales_data.resample('D').sum()

# Split the data into training and testing sets
train_size = int(len(daily_sales) * 0.8)
train_data, test_data = daily_sales[:train_size], daily_sales[train_size:]

# Fit the ARIMA model
model = ARIMA(train_data, order=(5,1,0))
model_fit = model.fit(disp=0)

# Make predictions
predictions = model_fit.forecast(steps=len(test_data))

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print("Root Mean Squared Error:", rmse)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(test_data, label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.legend()
plt.title("Future Sales Prediction using ARIMA")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()
