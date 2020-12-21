import numpy as np #for  numerical analytics
import pandas as pd #for data analytics
import category_encoders as ce #encoding
from sklearn.preprocessing import LabelEncoder #label encoder

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn.preprocessing

from sklearn.metrics import roc_curve, precision_recall_curve, auc,roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, style
import itertools

from tqdm import tqdm_notebook

from core import *
from imports import *
from db_core import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
from keras.layers import Dense,Dropout, SimpleRNN, LSTM
from keras.models import Sequential
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

def setup():
    pd.set_option('display.max_rows', 3000)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    sns.set_context("paper", font_scale=1.3)
    sns.set_style('white')

#ingest data
def ingest_data():
    df = get_dataframe('sql_tensorflow_qry')
    data = df[['readtime_date','cons_int_0', 'cons_int_1', 'cons_int_2', 'cons_int_3', 'cons_int_4',
   'cons_int_5', 'cons_int_6', 'cons_int_7', 'cons_int_8', 'cons_int_9',
   'cons_int_10', 'cons_int_11', 'cons_int_12', 'cons_int_13',
   'cons_int_14', 'cons_int_15', 'cons_int_16', 'cons_int_17',
   'cons_int_18', 'cons_int_19', 'cons_int_20', 'cons_int_21',
   'cons_int_22', 'cons_int_23']]
    ddata = pd.melt(data, id_vars=['readtime_date'], 
                  var_name="Date", value_name="Value")

    convert_type = {'Value': float
                   } 
    ddata = ddata.astype(convert_type)
    
    di = {'cons_int_0': "00:00:00",'cons_int_1': "01:00:00", 'cons_int_2': "02:00:00",'cons_int_3': "03:00:00",'cons_int_4': "04:00:00",
      'cons_int_5': "05:00:00", 'cons_int_6': "06:00:00",'cons_int_7': "07:00:00",'cons_int_8': "08:00:00",
      'cons_int_9': "09:00:00", 'cons_int_10': "10:00:00",'cons_int_11': "11:00:00",'cons_int_12': "12:00:00",
      'cons_int_13': "13:00:00", 'cons_int_14': "14:00:00",'cons_int_15': "15:00:00",'cons_int_16': "16:00:00",
      'cons_int_17': "17:00:00", 'cons_int_18': "18:00:00",'cons_int_19': "19:00:00",'cons_int_20': "20:00:00",
      'cons_int_21': "21:00:00", 'cons_int_22': "22:00:00",'cons_int_23': "23:00:00"}
    ddata['Date'].map(di)
    ddata.head()
    s = ddata['Date']

    ddata['hours'] = s.map(di)
    ddata["datetime"] =  ddata['readtime_date'].astype(str) + ' ' + ddata["hours"]

    ddata1 = ddata[['datetime','Value']]
    
    ddata1['datetime'] = pd.to_datetime(ddata1['datetime'], format='%Y/%m/%d %H:%M:%S') 
    ddata1.Value = ddata1.Value.fillna(method='ffill')
    if ddata1.Value.isnull().sum() != 0:
        pass

    return ddata1

# split train/test
def train_test_split(ddata1,split_factor):
    dataset = ddata1.Value.values #numpy.ndarray
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * split_factor)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return train, test, scaler


# convert an array of values into a dataset matrix
# split sequences - 
def split_sequence(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# LSTM solver

def lstm_build(train,test,scaler):
    # reshape into X=t and Y=t+1
    n_steps=24#25# or n_steps
    trainX, trainY = split_sequence(train, n_steps)
    testX, testY = split_sequence(test, n_steps)

    trainX = np.reshape(trainX, (trainX.shape[0],  trainX.shape[1],1))
    testX = np.reshape(testX, (testX.shape[0],  testX.shape[1],1))


    # create and fit the LSTM network
    n_features = 1 # consumption is the feature here
    
    # instantiate the Sequential class which is the model class
    model = Sequential()
    # add LSTM Layer to the model class
    # return parameter is true as we add more layers to the model
    model.add(LSTM(50, return_sequences=True, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    # add dropout layer to reduce overfitting
    model.add(Dropout(0.2))
    
    # add 3 more dropout and LSTM layers
    model.add(LSTM(units=20, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=20, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=20))
    model.add(Dropout(0.2))
    
    # add a dense layer to make the model robust
    # no of neurons 1 at dense layer as we want to predict a single value in the output
    model.add(Dense(1))
    
    # compile the LSTM model before training on train data
    model.compile(loss='mse', optimizer='adam')
    
    # train the model
    history = model.fit(trainX, trainY, epochs=200, batch_size=70, validation_data=(testX, testY), 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    # Training Phase
    print(model.summary())
    
    # make predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)
    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    trainY = scaler.inverse_transform([trainY])
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform([testY])

    print('Train Mean Absolute Error:', mean_absolute_error(trainY[0], train_predict[:,0]))
    print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(trainY[0], train_predict[:,0])))
    print('Test Mean Absolute Error:', mean_absolute_error(testY[0], test_predict[:,0]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(testY[0], test_predict[:,0])))
    
    return model, history, testY, test_predict

def plot_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.ioff()
    plt.savefig('train_valid_loss', dpi=600)
    plt.close()

    
def plot_predict(testY, test_predict):
    
    aa=[x for x in range(200)]
    plt.figure(figsize=(17,7))
    plt.plot(aa, testY[0][:200], marker='.', label="actual")
    plt.plot(aa, test_predict[:,0][:200], 'r', label="prediction")
    plt.tight_layout()
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('power consumption', size=15)
    plt.xlabel('Time step, n_step=25 , 0, 25, 50, 75 ..', size=15)
    plt.legend(fontsize=15)
    plt.ioff()
    plt.savefig('plot_predict', dpi=600)
    plt.close()
    
    

    
if __name__ == '__main__':
    setup()
    ddata1 = ingest_data()
    print(ddata1.shape)
    print(ddata1)
    train, test, scaler = train_test_split(ddata1, .67)
    print(train.shape, test.shape, scaler)
    print(train)
    model, history, testY, test_predict = lstm_build(train,test,scaler)
    plot_loss(history)
    plot_predict(testY, test_predict)
