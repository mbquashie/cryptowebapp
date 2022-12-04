import streamlit as st
import time
import yfinance as yf
from datetime import date as dt
import numpy as np
import pandas as pd
from PIL import Image
import requests
import matplotlib as plt
import plotly.express as px
from plotly import graph_objs as go
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import metrics 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error



st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>     Cryptocurrency Prediction  Using  Machine Learning        </h1>", unsafe_allow_html=True)
image = Image.open('logo.jpg')
col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write(' ')

with col2:
    st.image(image, width = 1100)

with col3:
    st.write(' ')

#st.dataframe(df, height=800)

#data= yf.download ("BTC-USD",start="2014-09-17",end="2020-08-05")


selected=option_menu(
   menu_title=None,
   options=["Home","Indicator","How much should I Invest?","Long Short-Term Memory", "Artificial Neural Networks","Conclusion"],
   icons=["house","bar-chart-fill","currency-bitcoin","list-task","arrows-fullscreen","info-square"],
   menu_icon="cast",
   default_index=0,
   orientation="horizontal",
)

start='2014-09-17'
end='2020-08-05'
today=dt.today().strftime("%Y-%m-%d")


if selected=="Home":
 st.title("Till Date")
 cryptos=( "XRP-USD","MATIC-USD","ETH-USD","DOT-USD","BTC-USD")
 targetcrypto=st.selectbox("Select the Cryptocurrency you want to predict",cryptos)
 def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data
 data=load_data(targetcrypto)
 st.subheader("Data Since 2007")
 st.write(data.tail())
 def org_graph():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Till Date",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 org_graph()
 
 
 

if selected=="Indicator":
 st.write("A simple moving average (SMA) calculates the average of a selected range of prices, usually closing prices, by the number of periods in that range.")
 cryptos=( "XRP-USD","MATIC-USD","ETH-USD","DOT-USD","BTC-USD")
 targetcrypto=st.selectbox("Select the Cryptocurrency you want to predict",cryptos)
 def load_data(ticker):
    data=yf.download(ticker,start='2007-01-01')
    return data
 data=load_data(targetcrypto)
 data['MA20'] = data['Adj Close'].rolling(20).mean()
 data['MA50'] = data['Adj Close'].rolling(50).mean()
 data['MA100'] = data['Adj Close'].rolling(100).mean()
 data = data[['Adj Close','MA20','MA50','MA100']]
 data = data.dropna()
 st.write(data)
 st.subheader("Simple Moving Average")
 st.write("Here we make use of one of the famous indicator known as SMA i.e Simple Moving Average here the past prices are considered and the average btw the values is calculated for the period of time and this process continues")

 def sma_graph():
    fig1=go.Figure()
    fig1.add_trace(go.Scatter(x=data['MA20'].index,y=data['MA20'],name='MA20'))
    fig1.add_trace(go.Scatter(x=data['MA50'].index,y=data['MA50'],name='MA50'))
    fig1.add_trace(go.Scatter(x=data['MA100'].index,y=data['MA100'], name='MA100'))
    fig1.add_trace(go.Scatter(x=data['Adj Close'].index,y=data['Adj Close'], name='Adj Close'))
    fig1.layout.update(title_text="Moving average",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)
 sma_graph()
 st.subheader("CONCLUSION")
 st.write("We observe that considering 20 days moving average shows relatively close result than 50 days moving average and 100 days moving average.hence if the period is less for calculating moving average the more accurate results will be drawn")
 
 
   
if selected=="How much should I Invest?":
    url_base='https://api.coingecko.com/api/v3/simple/price?ids='
    url_end='&vs_currencies=usd'
    crypt_dictionary = {"Bitcoin": "bitcoin", "Ethereum": "ethereum", "Ripple": "xrp", "Bitcoin Cash": "bitcoin-cash",
                    "Litecoin": "litecoin", "Stellar": "stellar", "Tether": "tether-gold", "Bitcoin SV": "bitcoin-sv", "Dash": "dash"}
    crypt_arr = ['Bitcoin', 'Ethereum', 'Ripple', 'Bitcoin Cash',
             'Litecoin', 'Stellar', 'Tether', 'Bitcoin SV', 'Dash']
    targetcrypto=st.selectbox("ENTER THE CRYPTO TO BE PREDICTED",crypt_arr)
    url=url_base+targetcrypto+url_end
    myrequest=requests.get(url)
    data_temp=myrequest.json()
    data_temp=pd.DataFrame.from_dict(data_temp)
    price=data_temp.to_numpy()
    currprice= float(price)
    money=st.number_input("ENTER THE MONEY YOU WANT TO INVEST IN DOLLARS",min_value=100.00,step=10.00)
    amount=money/currprice
    st.write("You will make an ",amount," of ",targetcrypto,".")


if selected=="Long Short-Term Memory":
 st.title("Till Date")
 cryptos=( "XRP-USD","MATIC-USD","ETH-USD","DOT-USD","BTC-USD")
 targetcrypto=st.selectbox("Select the Cryptocurrency you want to predict",cryptos)
 def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data
 data=load_data(targetcrypto)
 np.sum(data['Close'] - data['Adj Close'])
 st.write(data.tail())
 def org_graph():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Till Date",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 org_graph()
 
 
 def lstm_graph():
    fig=go.Figure()
    fig.add_trace(go.Box(y=data['Close'],name='stock_open'))
    fig.add_trace(go.Box(y=data['Open'],name='stock_close'))
    fig.layout.update(title_text="Box Plot",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 lstm_graph()
 
 Op=data["Open"]
 Cl=data["Close"]
 data1=pd.DataFrame(Op.values,columns=["Open"])
 data1["Close"]=Cl.values
 data1.index=Cl.index
 scaler = MinMaxScaler(feature_range=(0, 1))
 scaled_arr = scaler.fit_transform(data1[:])
 scaled_data=pd.DataFrame(scaled_arr)
 
 Scaled_Open=scaled_data[0].values
 Scaled_Close=scaled_data[1].values
 
 ## we will use 80% for training
 train_size = int(len(Scaled_Close) * 0.8) 

 ## We will use 20% for testing
 test_size = len(Scaled_Close) - train_size 

 # Split the data manually 
 trainO, testO = Scaled_Open[0:train_size], Scaled_Open[train_size:len(Scaled_Open)]
 trainC, testC = Scaled_Close[0:train_size], Scaled_Close[train_size:len(Scaled_Close)]
 
 def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

 look_back = 10

 # Split the data into training features and training targets
 trainX, trainY = create_dataset(trainC, look_back)

 # Split the data into testing features and testing targets
 testX, testY = create_dataset(testC, look_back)
 
 trainXkeras = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
 testXkeras = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
 
 ## Defining our model
 model = Sequential()

 # Adding an LSTM cell.
 model.add(LSTM(10, batch_input_shape=(None,10,1), stateful=False))

 # We have to add a fully connected layer of the LSTM to output the predictions. Here we only have 1 node because our time-series prediction is expecting one output. 
 model.add(Dense(1))

 # We define our loss function. Here since we are dealing with numerical data, one option is to use mean squared error and we will use adam as the optimization function.
 model.compile(loss='mean_squared_error', optimizer='adam')
 
 
 start = time.time()
 result= model.fit(trainXkeras, trainY, epochs=100, batch_size=20,validation_split=0.2, verbose = 0, shuffle=False)
 
 
 def lstm_graph():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=result.epoch,y=result.history['loss'],name='Train loss'))
    fig.add_trace(go.Scatter(x=result.epoch,y=result.history['val_loss'],name='Validation loss'))
    fig.layout.update(title_text="Prediction of one day ahead with 10 days lookback",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 lstm_graph()
 
 
 # Estimate model performance
 batch_size=10
 trainScore = model.evaluate(trainXkeras, trainY, batch_size=batch_size, verbose=0)
 model.reset_states()

 # generate predictions for training
 trainPredict = model.predict(trainXkeras, batch_size=batch_size)
 testPredict = model.predict(testXkeras, batch_size=batch_size)

 # shift train predictions for plotting
 trainPredictPlot = np.empty_like(Scaled_Open)
 trainPredictPlot[:] = np.nan
 trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict.reshape(1,-1)

 # shift test predictions for plotting
 testPredictPlot = np.empty_like(Scaled_Open)
 testPredictPlot[:] = np.nan
 testPredictPlot[len(trainPredict)+(look_back*2)+1:len(Scaled_Close)-1] = testPredict.reshape(1,-1)

 openforcomparison = np.empty_like(Scaled_Open)
 openforcomparison[:] = np.nan
 openforcomparison[len(trainO):len(Scaled_Open)] = testO
 
 

 def lstm_graph():
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=Scaled_Close))
    fig.add_trace(go.Scatter(y=trainPredictPlot))
    fig.add_trace(go.Scatter(y=testPredictPlot))
    fig.layout.update(title_text="Comparison of predicted and actual closing Bitcoin Price",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 lstm_graph()
 
 openforcomparison = np.empty_like(Scaled_Open)
 openforcomparison[:] = np.nan
 openforcomparison[len(trainO):len(Scaled_Open)] = testO
 
 
 def lstm_graph():
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=openforcomparison, name='Predicted for open'))
    fig.add_trace(go.Scatter(y=testPredictPlot,name='Test predict plot'))
    fig.layout.update(title_text="Comparison of predicted and actual closing Bitcoin Price",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 lstm_graph()
 
 
 def mean_absolute_percentage_error(y_true,y_pred):
  y_true,y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


 Error1= mean_squared_error(testY,testPredict)
 RMSE = np.sqrt(Error1)
 Error2 = mean_absolute_error(testY,testPredict)
 Error3 = mean_absolute_percentage_error(testY,testPredict)
 

 st.write("The Mean Squared Error is: ",Error1)
 st.write("The Root Mean Squared Error is: ",RMSE)
 st.write("The Mean Absolute Error is: ",Error2)
 st.write("The Mean Absolute Percentage Error is: ",Error3)
 



if selected=="Artificial Neural Networks":
 st.title("Till Date")
 cryptos=( "XRP-USD","MATIC-USD","ETH-USD","DOT-USD","BTC-USD")
 targetcrypto=st.selectbox("Select the Cryptocurrency you want to predict",cryptos)
 def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data
 data=load_data(targetcrypto)
 np.sum(data['Close'] - data['Adj Close'])
 st.write(data.tail())
 dataset= data.filter(['Close'])
 Data=dataset.values
 scaler = MinMaxScaler(feature_range=(0, 1))
 scaled_data = scaler.fit_transform(Data)
 training_data_len = math.ceil(len(Data)*.7)
 
 
 Train_data = scaled_data[0:training_data_len, :]

 ## Split the data into X_train and Y_train datasets
 X_train = []
 Y_train = []
 for i in range (10, len(Train_data)):
     X_train.append(Train_data[i-10:i, 0])
     Y_train.append(Train_data[i,0])
     
  
 X_train, Y_train = np.array(X_train), np.array(Y_train)   
 
 def network_model():
     model = Sequential()
     model.add(Dense(10, input_dim=10, activation='linear'))
     model.add(Dense(6, activation='linear'))
     model.add(Dense(1, activation='linear'))
     # Compile the model
     model.compile(loss='mean_squared_error', optimizer='adam',metrics=[metrics.mse])
     return model 
     
 model=network_model()
 start = time.time()
 Result = model.fit(X_train, Y_train, epochs=300, batch_size=120, validation_split=0.2,verbose=1)
     
     
 def org_graph():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=Result.epoch,y=Result.history['loss'],name='Train loss'))
    fig.add_trace(go.Scatter(x=Result.epoch,y=Result.history['val_loss'],name='Validation loss'))
    fig.layout.update(title_text="Plot of the training loss and validation loss",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 org_graph()
 
 ### create a new array containing scaled values from index  1445 to 2149
 Test_data = scaled_data[training_data_len-10:, :]

 ### Create the datasets X_test and Y_test
 X_test = []
 Y_test = Data[training_data_len:, :]
 for i in range (10, len(Test_data)):
     X_test.append (Test_data[i-10:i, 0])
     
     
 x_test = np.array(X_test)
 y_test=np.array(Y_test)
 test_prediction = model.predict(x_test)
 Test_prediction=scaler.inverse_transform(test_prediction)
 


 fig1=px.line(Test_prediction,title="Predicted Prices")
 fig1.update_traces(line=dict(color="Red", width=2.0))
 st.plotly_chart(fig1)
 
 fig2=px.line(Y_test,title="Actual Prices")
 fig2.update_traces(line=dict(color="Blue", width=2.0))
 st.plotly_chart(fig2)
 
 

 def mean_absolute_percentage_error(y_true,y_pred):
     y_true,y_pred = np.array(y_true), np.array(y_pred)
     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
 
 
 Error1= mean_squared_error(y_test,Test_prediction)
 RMSE = np.sqrt(Error1)
 Error2 = mean_absolute_error(y_test,Test_prediction)
 Error3 = mean_absolute_percentage_error(y_test,Test_prediction)

 st.write("The Mean Squared Error is: ",Error1)
 st.write("The Root Mean Squared Error is: ",RMSE)
 st.write("The Mean Absolute Error is: ",Error2)
 st.write("The Mean Absolute Percentage Error is: ",Error3)

 if selected=="Conclusion":
    st.write('''We concentrated on predicting the final value of cryptocurrency.
          comparing the performance of ANN and LSTM, two machine learning models,n order
          to determine which model is best able to forecast bitcoin prices. \n The LSTM model 
          performed better on the unseen data, according to our findings. Because the values 
          of the accuracy metrics for LSTM are quite low in comparison to the ANN, we saw that
          LSTM performed better than the ANN. We can infer from our findings that LSTM can be used 
          to forecast bitcoin prices.''')







