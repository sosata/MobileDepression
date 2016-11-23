
# coding: utf-8

# In[19]:

import pandas as pd
import numpy as np

data_fus = pd.read_csv('/home/sohrob/Dropbox/Data/CS120/1022235/fus.csv',sep='\t',header=None)
data_coe = pd.read_csv('/home/sohrob/Dropbox/Data/CS120/1022235/coe.csv',sep='\t',header=None)
target = pd.read_csv('/home/sohrob/Dropbox/Data/CS120/1022235/emm.csv',sep='\t',header=None)


# In[20]:

# adapting data to neural net

deltat = 600
win = 12*3600

for (i,t1) in enumerate(target[0]):
    for t2 in np.arange(t1-win, t1, deltat):
        # capturing gps data
        ind = np.where(data_fus[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            lat = np.nanmean(data_fus[1][ind])
            lng = np.nanmean(data_fus[2][ind])
        else:
            lat = np.nan
            lng = np.nan
        # capturing communication data
        ind = np.where(data_coe[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            sms = np.sum(data_coe[3][ind]=='SMS')
            phone = np.sum(data_coe[3][ind]=='PHONE')
            incoming = np.sum(data_coe[4][ind]=='INCOMING')
            outgoing = np.sum(data_coe[4][ind]=='OUTGOING')
            missed = np.sum(data_coe[4][ind]=='MISSED')
        else:
            sms = 0
            phone = 0
            incoming = 0
            outgoing = 0
            missed = 0
        hour = np.mod(t1,86400)/3600.0
        dow = np.mod(t1,86400*7)/86400.0
        stress = target.loc[i,2]

        ft = np.array([lat, lng, hour, dow, stress, sms, phone, incoming, outgoing, missed])
        ft = ft.reshape(1,ft.size)
        if t2==t1-win:
            x_sample = ft
        else:
            x_sample = np.append(x_sample, ft, axis=0)
    
    mood = target.loc[i,1]
    mood = mood.reshape(1,mood.size)
    
    if i==0:
        x = x_sample.reshape(1,x_sample.shape[0],x_sample.shape[1])
        y = mood
    else:
        x = np.append(x, x_sample.reshape(1,x_sample.shape[0],x_sample.shape[1]), axis=0)
        y = np.append(y, mood, axis=0)


# In[21]:

# remove samples that contain nan
ind_del = []
for i in range(x.shape[0]):
    if np.sum(np.sum(np.isnan(x[i,:,:])))>0:
        ind_del.append(i)
x = np.delete(x, ind_del, axis=0)
y = np.delete(y, ind_del, axis=0)


# In[28]:

# building the network

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, TimeDistributed
from keras import regularizers, optimizers

reg = regularizers.WeightRegularizer(l1=0, l2=.1)
optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = Sequential()
model.add(LSTM(output_dim=20, input_dim=x.shape[2], return_sequences=False, activation='tanh', W_regularizer=reg,              dropout_W=0.5,dropout_U=0.5))
model.add(Dense(20, activation='tanh'))
model.add(Dense(y.shape[1], activation='linear'))
# model.add(TimeDistributed(Dense(output_dim=1)))

model.compile(optimizer=optim, loss='mse')


# In[31]:

# training

x_train = x[:x.shape[0]/2,:,:]
y_train = y[:y.shape[0]/2,:]

x_test = x[x.shape[0]/2:,:,:]
y_test = y[y.shape[0]/2:,:]

# centering the input
x_train = x_train - np.ones([x_train.shape[0],x_train.shape[1],1])*np.mean(np.mean(x_train,axis=0),axis=0)
x_test = x_test - np.ones([x_test.shape[0],x_test.shape[1],1])*np.mean(np.mean(x_test,axis=0),axis=0)

# y = y - np.mean(y)

model.fit(x_train, y_train, batch_size=1, verbose=1, nb_epoch=30, validation_data=(x_test,y_test))


# In[32]:

# prediction

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

y_pred_train = model.predict(x_train)
y_pred = model.predict(x_test)

plt.figure(figsize=[5,2])
plt.plot(y_train, color=(0,.5,0))
plt.plot(y_pred_train, color=(0,0,0))
plt.text(y_train.shape[0],6,'mse=%.2f' % np.mean(np.power(y_pred_train-y_train,2)))

plt.figure(figsize=[5,2])
plt.plot(y_test, color=(0,.5,0))
plt.plot(y_pred, color=(0,0,0))
plt.text(y_test.shape[0],6,'mse=%.2f' % np.mean(np.power(y_pred-y_test,2)))


# In[18]:

x[:,0,-1]

