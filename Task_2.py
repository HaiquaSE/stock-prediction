import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Download Historical Data
stock = "AAPL"
data = yf.download(stock, start = "2015-01-01", end = "2025-01-01")
print(data.head())

# Prepare Features and Target
data["Next_Close"] = data["Close"].shift(-1)

# Remove last row (because Next_Close is NaN)
data = data[:-1]
x = data[["Open" , "High" , "Low" , "Volume"]]
y = data["Next_Close"]

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = False
)

# Train Model through Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_preds = lr.predict(x_test)

# Evaluate
print("Linear Regression MAE:" , mean_absolute_error(y_test, lr_preds))

# Plot Actual vs Predicted
plt.figure(figsize = (10, 5))
plt.plot(y_test.values, label = "Actual Close")
plt.xlabel ("Days")
plt.ylabel("Price")
plt.title(f"{stock} - Actual vs Predicted Closing Price")
plt.legend()
plt.show()