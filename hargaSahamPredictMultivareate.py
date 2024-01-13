'''
dataset : https://query1.finance.yahoo.com/v7/finance/download/GE?period1=-252288000&period2=1656288000&interval=1d&events=history&includeAdjustedClose=true

env     :
1. Python   : 3.7
2. Packages : numpy, pandas, matplotlib, scikit-learn, seaborn, tensorflow==2.8.0

'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from keras.models import load_model


df = pd.read_csv('/content/drive/MyDrive/dataset/bigdata/Jan3_1962.csv') #membaca dataframe dari file csv
print(df.head())

#Mengambil kolom 'Date' untuk plot 
train_dates = pd.to_datetime(df['Date'])
print(train_dates.tail(15)) #mengecek 15 nilai date terakhir

#Variabel untuk training 
cols = list(df)[1:6]
print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Close']

#dataframe konversi menjadi float
df_for_training = df[cols].astype(float)

#normalisasi dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
print(df_for_training)
print('====================')

#transformasi dataset
df_for_training_scaled = scaler.transform(df_for_training)
print(df_for_training_scaled)

from sklearn.model_selection import train_test_split
#list untuk dataset training
trainX = []
trainY = []

n_future = 1   # Jumlah hari yang akan diprediksi (variabel dependen).
n_past = 14  # Jumlah hari yang menjadi variabel independen.

#sliding window
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])


trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print('==========================')

from keras.utils.vis_utils import plot_model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# fit the model
history = model.fit(trainX, trainY, epochs=300, batch_size=16, validation_split=0.1, verbose=1)

model.save("/content/drive/MyDrive/SEMESTER 6/SPK/multivarite300epoch16batch.hdf5")

plt.figure(figsize=(16, 9), dpi = 150)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

#Predicting...
#Library untuk mendapatkan bussiness dy di US.
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

n_past = 1 #hari inisiasi
n_days_for_prediction= 500 #jumlah hari yang harus diprediksi ke depan

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()
print(predict_period_dates)


modelCoba = load_model("/content/drive/MyDrive/SEMESTER 6/SPK/multivarite300epoch16batch.hdf5") #load model

val_loss = modelCoba.evaluate(trainX, trainY, batch_size=16) 
print("Loaded model, loss: {}".format(val_loss))

#Membuat prediksi 
prediction = modelCoba.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction

#Data dengan 1 kolom selanjutnya di-inverse transform agar menjadi ke data awal
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]


# Konversi timestamp menjadi date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

original = df[['Date', 'Open']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2020-5-1'] #'2020-5-1

sns.set(rc = {'figure.figsize':(16,9)})
sns.lineplot(original['Date'], original['Open'])
sns.lineplot(df_forecast['Date'], df_forecast['Open'])

print(df_forecast['Date'].astype(str) +" || "+ df_forecast["Open"].astype(str))