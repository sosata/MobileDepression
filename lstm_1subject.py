
# coding: utf-8

# In[29]:

import pandas as pd
import numpy as np
from scipy import stats

subject = '919141'

data_act = pd.read_csv('/data/CS120/'+subject+'/act.csv',sep='\t',header=None)
data_aud = pd.read_csv('/data/CS120/'+subject+'/aud.csv',sep='\t',header=None)
data_bat = pd.read_csv('/data/CS120/'+subject+'/bat.csv',sep='\t',header=None)
data_cal = pd.read_csv('/data/CS120/'+subject+'/cal.csv',sep='\t',header=None)
data_coe = pd.read_csv('/data/CS120/'+subject+'/coe.csv',sep='\t',header=None)
data_fus = pd.read_csv('/data/CS120/'+subject+'/fus.csv',sep='\t',header=None)
data_scr = pd.read_csv('/data/CS120/'+subject+'/scr.csv',sep='\t',header=None)
data_wif = pd.read_csv('/data/CS120/'+subject+'/wif.csv',sep='\t',header=None)
target = pd.read_csv('/data/CS120/'+subject+'/emm.csv',sep='\t',header=None)


# In[30]:

# adapting data to neural net

deltat = 1800
win = 24*3600

for (i,t1) in enumerate(target[0]):
    lat = np.nan
    lng = np.nan
    for t2 in np.arange(t1-win, t1, deltat):
        
        # GPS data
        ind = np.where(data_fus[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            lat = np.nanmean(data_fus[1][ind])
            lng = np.nanmean(data_fus[2][ind])
#         else:
            # nothing, just keep the last one
            
        # communication data
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
            
        # activity data
        ind = np.where(data_act[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            act_onfoot = np.sum(data_act[1][ind]=='ON_FOOT')/float(ind.size)
            act_still = np.sum(data_act[1][ind]=='STILL')/float(ind.size)
            act_invehicle = np.sum(data_act[1][ind]=='IN_VEHICLE')/float(ind.size)
            act_tilting = np.sum(data_act[1][ind]=='TILTING')/float(ind.size)
            act_confidence = np.nanmean(data_act[2][ind])
        else:
            act_onfoot = 0
            act_still = 1
            act_invehicle = 0
            act_tilting = 0
            act_confidence = 100
            
        # screen data
        ind = np.where(data_scr[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            scr_n = ind.size/2.0
        else:
            scr_n = 0
        
        # audio data
        ind = np.where(data_aud[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            aud_amp = np.mean(data_aud[2][ind])
            aud_frq = np.mean(data_aud[3][ind])
        else:
            aud_amp = 0
            aud_frq = 0

        # call data
        ind = np.where(data_cal[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            cal_dur = np.sum(data_cal[1][ind]=='Off-Hook')
        else:
            cal_dur = 0
        
        # battery data
        ind = np.where(data_bat[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            bat_charge = np.mean(data_bat[1][ind])
            bat_state = stats.mode(data_bat[2][ind])[0][0]
        else:
            bat_charge = np.nan
            bat_state = np.nan

        # wifi data
        ind = np.where(data_wif[0].between(t2, t2+deltat, inclusive=True))[0]
        if ind.size:
            wif_n = np.mean(data_wif[3][ind])
        else:
            wif_n = np.nan
        
        # time
        hour = np.mod(t1,86400)/3600.0
        dow = np.mod(t1,86400*7)/86400.0
        
#         stress = target.loc[i,2]
        
        # input vector
        ft = np.array([lat, lng,                       hour, dow,                       sms, phone, incoming, outgoing, missed,                       act_onfoot, act_still, act_invehicle, act_tilting, act_confidence,                       scr_n,                       aud_amp, aud_frq,                       cal_dur,                       bat_charge, bat_state,                       wif_n])
        
        ft = ft.reshape(1,ft.size)
        
        # adding to input matrix
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


# In[31]:

# remove samples that contain nan

ind_del = []
for i in range(x.shape[0]):
    if np.sum(np.sum(np.isnan(x[i,:,:])))>0:
        ind_del.append(i)
print '{}/{} samples removed'.format(len(ind_del),x.shape[0])
x = np.delete(x, ind_del, axis=0)
y = np.delete(y, ind_del, axis=0)


# In[78]:

# build the network

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, TimeDistributed
from keras import regularizers, optimizers

reg = regularizers.WeightRegularizer(l1=0, l2=0)
optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# optim = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model = Sequential()
model.add(LSTM(output_dim=100, input_dim=x.shape[2], return_sequences=False, activation='tanh', W_regularizer=reg,              dropout_W=0.2, dropout_U=0.2))
# model.add(Dense(20, activation='tanh'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(y.shape[1], activation='linear'))
# model.add(TimeDistributed(Dense(output_dim=1)))

model.compile(optimizer=optim, loss='mse')


# In[79]:

# training

split = int(round(x.shape[0]*0.8))

x_train = x[:split,:,:]
y_train = y[:split,:]

x_test = x[split:,:,:]
y_test = y[split:,:]

# centering the input
x_train = x_train - np.ones([x_train.shape[0],x_train.shape[1],1])*np.mean(np.mean(x_train,axis=0),axis=0)
x_test = x_test - np.ones([x_test.shape[0],x_test.shape[1],1])*np.mean(np.mean(x_test,axis=0),axis=0)

# y = y - np.mean(y)

model.fit(x_train, y_train, batch_size=10, verbose=1, nb_epoch=10, validation_data=(x_test,y_test))


# In[80]:

# prediction

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

y_pred_train = model.predict(x_train)
y_pred = model.predict(x_test)

plt.figure(figsize=[5,2])
plt.plot(y_train, color=(0,.5,0))
plt.plot(y_pred_train, color=(0,0,0))
plt.xlim([0, y_train.shape[0]-1])
plt.ylim([0, 9])
plt.text(y_train.shape[0],9,'R2=%.2f' % (1-np.sum(np.power(y_pred_train-y_train,2))/np.sum(np.power(np.mean(y_train)-y_train,2))))
plt.title('train')

plt.figure(figsize=[5,2])
plt.plot(y_test, color=(0,.5,0))
plt.plot(y_pred, color=(0,0,0))
plt.xlim([0, y_test.shape[0]-1])
plt.ylim([0, 9])
plt.text(y_test.shape[0],9,'R2=%.2f' % (1-np.sum(np.power(y_pred-y_test,2))/np.sum(np.power(np.mean(y_test)-y_test,2))))
plt.title('test')

