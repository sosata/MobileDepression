
# coding: utf-8

# In[7]:

# This code prepares sensor and EMA data for the LSTM

from ipyparallel import Client

rc = Client()
dv = rc[:]

@dv.parallel(block = True)
def prepare_data(subjects):

    import pandas as pd
    import numpy as np
    from scipy import stats
    import os

    deltat = 1800
    win = 24*3600

    data_dir = '/data/CS120/'

    x = np.array([])
    y = np.array([])

    for (s,subject) in enumerate(subjects):

        print s,
        if os.path.exists(data_dir+subject+'/act.csv'):
            data_act = pd.read_csv(data_dir+subject+'/act.csv',sep='\t',header=None)
        else:
            print ' skipping - no data'
            continue
        if os.path.exists(data_dir+subject+'/aud.csv'):
            data_aud = pd.read_csv(data_dir+subject+'/aud.csv',sep='\t',header=None)
        else:
            print ' skipping - no data'
            continue
        if os.path.exists(data_dir+subject+'/bat.csv'):
            data_bat = pd.read_csv(data_dir+subject+'/bat.csv',sep='\t',header=None)
        else:
            print ' skipping - no data'
            continue
        if os.path.exists(data_dir+subject+'/cal.csv'):
            data_cal = pd.read_csv(data_dir+subject+'/cal.csv',sep='\t',header=None)
        else:
            print ' skipping - no data'
            continue
        if os.path.exists(data_dir+subject+'/coe.csv'):
            data_coe = pd.read_csv(data_dir+subject+'/coe.csv',sep='\t',header=None)
        else:
            print ' skipping - no data'
            continue
        if os.path.exists(data_dir+subject+'/fus.csv'):
            data_fus = pd.read_csv(data_dir+subject+'/fus.csv',sep='\t',header=None)
        else:
            print ' skipping - no data'
            continue
        if os.path.exists(data_dir+subject+'/scr.csv'):
            data_scr = pd.read_csv(data_dir+subject+'/scr.csv',sep='\t',header=None)
        else:
            print ' skipping - no data'
            continue
        if os.path.exists(data_dir+subject+'/wif.csv'):
            data_wif = pd.read_csv(data_dir+subject+'/wif.csv',sep='\t',header=None)
        else:
            print ' skipping - no data'
            continue
        if os.path.exists(data_dir+subject+'/emm.csv'):
            target = pd.read_csv(data_dir+subject+'/emm.csv',sep='\t',header=None)
        else:
            print ' skipping - no data'
            continue
        print

        for (i,t1) in enumerate(target[0]):
            lat = np.nan
            lng = np.nan

            for t2 in np.arange(t1-win, t1, deltat):

                # GPS data
                ind = np.where(data_fus[0].between(t2, t2+deltat, inclusive=True))[0]
                if ind.size:
                    lat = np.nanmean(data_fus[1][ind])
                    lng = np.nanmean(data_fus[2][ind])

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

                # input vector
                vec = np.array([lat, lng,                               hour, dow,                               sms, phone, incoming, outgoing, missed,                               act_onfoot, act_still, act_invehicle, act_tilting, act_confidence,                               scr_n,                               aud_amp, aud_frq,                               cal_dur,                               bat_charge, bat_state,                               wif_n])

                vec = vec.reshape(1,vec.size)

                # adding to input matrix
                if t2==t1-win:
                    x_sample = vec
                else:
                    x_sample = np.append(x_sample, vec, axis=0)

            target_vec = np.array([s, target.loc[i,1]]) #subject ID is added- shall be used for subject-wise cross-validation
            target_vec = target_vec.reshape(1,target_vec.size)

            if x.any():
                x = np.append(x, x_sample.reshape(1,x_sample.shape[0],x_sample.shape[1]), axis=0)
                y = np.append(y, target_vec, axis=0)
            else:
                x = x_sample.reshape(1,x_sample.shape[0],x_sample.shape[1])
                y = target_vec
    
    return [x,y]


# In[26]:

import os
import pickle

data_dir = '/data/CS120/'
subjects = os.listdir(data_dir)
# subjects = subjects[:12]
data = prepare_data(subjects)
with open('data_lstm.dat', 'w') as file_out:
    pickle.dump(data, file_out)
file_out.close()

