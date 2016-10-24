import numpy as np
import pandas as pd
from smooth_speed import smooth_speed

def filter_speed(data, speed_max):

    # removing duplicate timestamps if they exist
    ind_dup = np.where(np.diff(data[0])==0)[0]+1;
    if ind_dup.size>0:
        data = data.drop(data.index[ind_dup])
        data = data.reset_index(drop=True)
        print '{}% of data removed because of duplicates'.format(ind_dup.size/float(data.shape[0])*100)

    if data.shape[0]>=2:
        spd_temp = np.sqrt(np.power(np.divide(np.diff(data[1]),np.diff(data[0])),2)+\
            np.power(np.divide(np.diff(data[2]),np.diff(data[0])),2))
        spd_temp *= 3600 # convert to km/h
        spd_temp = smooth_speed(spd_temp)
        ind = np.where(spd_temp<=speed_max)[0]
        data = data.loc[ind,:]
        data = data.reset_index(drop=True)
        print '{}% of data removed by speed filtering'.format((data.shape[0]-ind.size)/float(data.shape[0])*100)
    else:
        data = pd.DataFrame()

    return data