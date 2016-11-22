
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

data = pd.read_csv('/home/sohrob/Dropbox/Data/CS120/1022235/fus.csv',sep='\t',header=None)
target = pd.read_csv('/home/sohrob/Dropbox/Data/CS120/1022235/emm.csv',sep='\t',header=None)


# In[2]:

# adapting data to neural net

deltat = 600
win = 26*3600

x = np.array([])
y = np.array([])

for (i,t1) in enumerate(target[0]):
    for t2 in np.arange(t1-win, t1, deltat):
        # capturing sensor data
        ind = np.where(data[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            lat = np.nanmean(data[1][ind])
            lng = np.nanmean(data[2][ind])
        else:
            lat = np.nan
            lng = np.nan

        if t2==t1-win:
            x_sample = np.array([lat, lng]).reshape(1,2)
        else:
            x_sample = np.append(x_sample, np.array([lat, lng]).reshape(1,2), axis=0)
    
    mood = target.loc[i,1]
    y = np.append(y, mood)
    
    if i==0:
        x = x_sample.reshape(1,x_sample.shape[0],x_sample.shape[1])
    else:
        x = np.append(x, x_sample.reshape(1,x_sample.shape[0],x_sample.shape[1]), axis=0)


# In[3]:

# remove samples that contain nan

ind_del = []
for i in range(x.shape[0]):
    if np.sum(np.sum(np.isnan(x[i,:,:])))>0:
        ind_del.append(i)
x = np.delete(x, ind_del, axis=0)
y = np.delete(y, ind_del)


# In[7]:

x.shape


# In[ ]:

# shaping the array [samples, timesteps, features]
# x = x.reshape([1,x.shape[0],x.shape[1]])
# x = x.reshape([1,x.shape[0],x.shape[1]])


# In[59]:

# building the network

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, TimeDistributed

model = Sequential()
model.add(LSTM(2, input_dim=2, return_sequences=False, activation='tanh'))
model.add(Dense(1))
# model.add(TimeDistributed(Dense(output_dim=1)))

model.compile(optimizer='adam', loss='mse')


# In[60]:

# training the network

# normalizing input
x = x - np.ones([x.shape[0],x.shape[1],1])*np.mean(np.mean(x,axis=0),axis=0)

model.fit(x, y, batch_size=1, verbose=2, nb_epoch=10)


# In[56]:

# prediction on training

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

x_test = x
y_test = y

y_pred = model.predict(x_test)

plt.figure(figsize=[7,3])
plt.plot(y_test, color=(0,.5,0))
plt.plot(y_pred, color=(0,0,1))


# In[51]:

y.shape

