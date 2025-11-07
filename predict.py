import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import math

#import necessary, fully prepared variables from your data utility file
from fetch_data import prepare_stock_data, TIME_STEP, ticker_symbol

X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler = prepare_stock_data()

model=load_model('stock_lstm_model.h5')


#generate predictions
#use the model to predict on the unseen X_test
print("Generating predictions on the test data")
predictions = model.predict(X_test)

#inverse transformartion (De-Normalize)
predicted_prices = scaler.inverse_transform(predictions)

actual_prices = scaler.inverse_transform(Y_test.reshape(-1,1))

#evaluate
#calculate root mean squared error
rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"\n Test RMSE (Root Mean Squared Error): ${rmse:.2f}")

#visualize the results
print("Plotting the results")
plt.figure(figsize=(14,7))
plt.plot(actual_prices, color='blue', label='Actual Price')
plt.plot(predicted_prices,color='red',alpha=0.7,label='Predicted_Price')
plt.title(f'{ticker_symbol} Stock Price Prediction (LSTM)')
plt.xlabel('Time (Days in Test set)')
plt.ylabel('Price USD')
plt.legend()
plt.grid(True)
plt.show()

