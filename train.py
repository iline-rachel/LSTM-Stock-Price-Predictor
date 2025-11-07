#import necessary libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE


from fetch_data import prepare_stock_data, TIME_STEP

X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler = prepare_stock_data()

#Model Architecture and training
#build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()

    #first LSTM layer
    model.add(LSTM(
        units = 50,
        return_sequences = True,
        input_shape = input_shape
    ))
    model.add(Dropout(0.2))

    #second LSTM layer
    model.add(LSTM(units= 50))
    model.add(Dropout(0.2))

    #output layer
    model.add(Dense(units=1))

    #compile the model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[RMSE(name='rmse')]
    )
    return model

if __name__ == '__main__' :
    #get the input shape for the model
    input_shape = (X_train.shape[1],X_train.shape[2])

    #build the model
    lstm_model=build_lstm_model(input_shape)
    lstm_model.summary()

    #train the model
    epochs=100
    batch_size=32

    print(f"\n Strating training for {epochs} epochs......")

    history = lstm_model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val,Y_val),
        verbose=1
    )
    print("Training completed successfully.")

    #save the trained model
    lstm_model.save('stock_lstm_model.h5')
    print("\nModel saved successfully ")


