#import necessary libraries
import yfinance as yf
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#define the ticker symbol and the desired period
ticker_symbol='AAPL'
YEARS=5
TIME_STEP=60  #look back period of 60 days

def prepare_stock_data():

    #data acquisition
    end_date=datetime.date.today()
    start_date=end_date - datetime.timedelta(days=YEARS*365) #last 5 years


    #fetch the data using the installed library yfinance
    stock_data=yf.download(
    ticker_symbol,
    start=start_date,
    end=end_date,
    auto_adjust=True #ensures prices are adjusted for splits and dividends
    )

    #dispaly the columns to see what was fetched
    print("\n DataFrane columns: ")
    print(stock_data.columns)
    
    #select the adjusted closing price stored in 'close' column. this single feature will be used to train the LLM
    closing_prices=stock_data['Close']
    
    print("\n First 5 rows of closing prices: ")
    print(closing_prices.head())

    #normalization and reshaping the data
    #MinMaxScaler requires the input to be 2D:(samples,datasets)
    #Currently closing_prices is a pandas series (1D). We reshape it (N,1)
    dataset=closing_prices.values.reshape(-1,1)
    print(f"\n Data is successfully reshaped for scaling. New Shape: {dataset.shape}")
    
    #Normalization
    #initialize the MinMaxScaler to scale data between 0 and 1
    scaler=MinMaxScaler(feature_range=(0,1))
    
    #fit the scaler and transform the data
    normalized_data=scaler.fit_transform(dataset)
    print(f"Data sucessfully normalized ")
    print(normalized_data[:5])
   
    #Sequence transformation
    def create_lstm_sequences(data: np.ndarray,time_step: int):
        X,Y = [], [] #transforms the 2D normalized data into 3D sequences X and 1D Y targets required for LSTM training
        for i in range(time_step,len(data)):
            X.append(data[i - time_step:i,0])
            Y.append(data[i,0])
        print(f"\n Created  {len(X)} training sequences (X/Y pairs).")
        
        #convert lists to numpy arrays
        X,Y=np.array(X),np.array(Y)
        
        #final reshaping to 3D for LSTM 
        X=X.reshape(X.shape[0],X.shape[1],1)
        return X, Y
    
    #execute the sequence creation
    X_sequences, Y_targets = create_lstm_sequences(normalized_data,TIME_STEP)
    print(f"Final LSTM Input Shape (X): {X_sequences.shape}")
    print(f"Final LSTM Target Shape (Y): {Y_targets.shape}")
    
    #split the data into training, test and validation sets
    total_size= X_sequences.shape[0]
    
    #80% train, 10% validation,10% test
    train_size = int(total_size * 0.8) 
    val_size=int(total_size * 0.1)
    print(f"\n Total sequences: {total_size}")
    print(f"Training size: {train_size}, Validation size: {val_size}, Test_size: {total_size - train_size - val_size}")
    
    #perform splitting using array slicing
    #Training set
    X_train = X_sequences[:train_size]
    Y_train = Y_targets[:train_size]
    #Validation set
    X_val = X_sequences[train_size:train_size + val_size]
    Y_val = Y_targets[train_size:train_size + val_size]

    #test set
    X_test = X_sequences[train_size + val_size:]
    Y_test = Y_targets[train_size + val_size:]
    print("Data is splitted into training, validation and test sets successfully.")

    return X_train,Y_train,X_val,Y_val,X_test,Y_test,scaler

#Excecute the data prepration pipeline

if __name__ == "__main__":
    print("Starting data Prepration utility.....")

    X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler = prepare_stock_data()

    print(f"\n Final training input shape (X): {X_train.shape}")
    print(f"Final Test input shape (X): {X_test.shape}")
    print(f"Scaler object is ready for inverse transformation")
