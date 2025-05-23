from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import LSTM

days=20000

today = date.today()
end_date = today.strftime("%Y-%m-%d")

start_date = today - timedelta(days=days)
start_date = start_date.strftime("%Y-%m-%d")




data = yf.download('BTC-USD', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
data["Date"] = data.index
data.reset_index(drop=True, inplace=True)




def create_dataset(serie, window_size=20):
    dataX, dataY = [], []
    for i in range(len(serie)-window_size-1):
        a = serie[i:(i+window_size), 0]  
        dataX.append(a)
        dataY.append([serie[i + window_size, 0]])
    return np.array(dataX), np.array(dataY)





window_size=20

closedf=data[['Close']] # import the data from the Close 
scaler=MinMaxScaler(feature_range=(0,1)) # scale to range 0-1 
closedf=scaler.fit_transform(closedf) # the new Dataset Fited 
  
X, y = create_dataset(closedf, window_size) # x,y data 


# Split the Data 


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0)



model = Sequential()
model.add(Input(shape=(window_size, 1))) 
model.add(LSTM(units=32,  dropout=0.1, activation="relu"))
model.add(Dense(1 , activation="relu"))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse'])



history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=100, batch_size=8)




pred_steps = 10  # How Many Days to Predict 
predicted_prices=[]
X_pred=[X[-1]]
X_pred = np.array(X_pred)
for _ in range(pred_steps):
    
    prediction = model.predict([X_pred])
    
    price=prediction[0]
    
    predicted_prices.append(price)
    
    X_pred = np.append(X_pred, [price], axis=1)
    
    X_pred = X_pred[:, 1:]

predicted_prices = np.array(predicted_prices, dtype=object)
predicted_prices = scaler.inverse_transform(predicted_prices)
print(predicted_prices)
