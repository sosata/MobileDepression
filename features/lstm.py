
# coding: utf-8

# In[20]:

import pandas as pd
import numpy as np

data = pd.read_csv('/home/sohrob/Dropbox/Data/CS120/1022235/fus.csv',sep='\t',header=None)
target = pd.read_csv('/home/sohrob/Dropbox/Data/CS120/1022235/emm.csv',sep='\t',header=None)


# In[71]:

# adapting data to neural net

deltat = 600

x = np.array([])
y = np.array([])
for t in np.arange(data.loc[0,0], data.loc[data.shape[0]-1,0], deltat):
    # capturing sensor data
    ind = np.where(data[0].between(t, t+deltat, inclusive=True))[0]
    if ind.size:
        lat = np.nanmean(data[1][ind])
        lng = np.nanmean(data[2][ind])
    else:
        lat = np.nan
        lng = np.nan
    # capturing targets
    ind = np.where(target[0].between(t, t+deltat, inclusive=True))[0]
    if ind.size:
        mood = np.nanmean(target[1][ind])
    else:
        mood = np.nan
    
    if t==data.loc[0,0]:
        x = np.array([lat, lng]).reshape(1,2)
    else:
        x = np.append(x, np.array([lat, lng]).reshape(1,2), axis=0)
    y = np.append(y, mood)
    
# adding bias
x = x.reshape([x.shape[0],x.shape[1],1])


# In[76]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Convolution2D, Flatten, LSTM

model = Sequential()
model.add(LSTM(4, input_dim=1))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# In[77]:

model.fit(x, y, batch_size=1, verbose=2)


# In[78]:

# prediction on trianing
x_test = x[:1000,:,:]
y_test = y[:1000]

y_pred = model.predict(x_test)


# In[79]:

y_pred

