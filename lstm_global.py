
# coding: utf-8

# In[14]:

import pandas as pd
import numpy as np
from scipy import stats
import pickle

with open('data_lstm.dat') as f:
    data = pickle.load(f)
f.close()


# In[18]:

# concatenating sensor and target data

n_core = 12
x = data[0]
y = data[1]
for i in range(n_core-1):
    x = np.concatenate([x,data[(i+1)*2]], axis=0)
    y = np.concatenate([y,data[(i+1)*2+1]], axis=0)


# In[21]:

# remove samples that contain nan

ind_del = []
for i in range(x.shape[0]):
    if np.sum(np.sum(np.isnan(x[i,:,:])))>0:
        ind_del.append(i)
print '{}/{} samples removed'.format(len(ind_del),x.shape[0])
x = np.delete(x, ind_del, axis=0)
y = np.delete(y, ind_del, axis=0)


# In[30]:

# build the network

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, TimeDistributed
from keras import regularizers, optimizers

reg = regularizers.WeightRegularizer(l1=0, l2=0)
optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# optim = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# optim = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False)

model = Sequential()
model.add(LSTM(output_dim=100, input_dim=x.shape[2], return_sequences=False, activation='tanh', W_regularizer=reg,              dropout_W=0.2, dropout_U=0.2))
# model.add(Dense(20, activation='tanh'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(1, activation='linear'))
# model.add(TimeDistributed(Dense(output_dim=1)))

model.compile(optimizer=optim, loss='mse')


# In[37]:

# training

split = int(round(x.shape[0]*0.9))

x_train = x[:split,:,:]
y_train = y[:split,1] # only mood

x_test = x[split:,:,:]
y_test = y[split:,1] # only mood

# centering the input
x_train = x_train - np.ones([x_train.shape[0],x_train.shape[1],1])*np.mean(np.mean(x_train,axis=0),axis=0)
x_test = x_test - np.ones([x_test.shape[0],x_test.shape[1],1])*np.mean(np.mean(x_test,axis=0),axis=0)

# y = y - np.mean(y)

model.fit(x_train, y_train, batch_size=1000, verbose=1, nb_epoch=20, validation_data=(x_test,y_test))


# In[38]:

# prediction

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

y_pred_train = model.predict(x_train).reshape([y_train.shape[0],])
y_pred = model.predict(x_test).reshape([y_test.shape[0],])

plt.figure(figsize=[10,2])
plt.plot(y_train, color=(0,.5,0))
plt.plot(y_pred_train, color=(0,0,0))
plt.xlim([0, y_train.shape[0]-1])
plt.ylim([-1, 10])
plt.text(y_train.shape[0],9,'R2=%.2f' % (1-np.sum(np.power(y_pred_train-y_train,2))/np.sum(np.power(np.mean(y_train)-y_train,2))))
plt.title('train')

plt.figure(figsize=[10,2])
plt.plot(y_test, color=(0,.5,0))
plt.plot(y_pred, color=(0,0,0))
plt.xlim([0, y_test.shape[0]-1])
plt.ylim([-1, 10])
plt.text(y_test.shape[0],9,'R2=%.2f' % (1-np.sum(np.power(y_pred-y_test,2))/np.sum(np.power(np.mean(y_train)-y_test,2))))
plt.title('test')


# In[35]:

y_test.shape

