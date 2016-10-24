
# coding: utf-8

# In[13]:

import numpy as np
import sklearn.cluster as skc
import pandas as pd
import sys
sys.path.insert(0, '/home/sohrob/Dropbox/Code/MobileDepression/features')
from cluster_kmeans import cluster_kmeans
from filter_speed import filter_speed
from estimate_distance import estimate_distance
from estimate_entropy import estimate_entropy
from estimate_homestay import estimate_homestay
from estimate_variance import estimate_variance

data = pd.read_csv('/home/sohrob/Dropbox/Data/CS120/1022235/fus.csv',sep='\t',header=None)

# centering 
# this may be a wrong thing to do, because it removes the actual skewness of coordinates which are typically not seen on map
# data[1] -= (np.max(data[1])+np.min(data[1]))/2.0
# data[2] -= (np.max(data[2])+np.min(data[2]))/2.0

# converting from degrees to km
data[2] = 111*np.multiply(np.cos(data[1]*np.pi/180),data[2])
data[1] = 111*data[1]

# removing non-stationary data points
data_filtered = data
data_filtered = filter_speed(data_filtered, 1)

# TODO: also filter based on histogram - if a bin contains very few data points

# k-means clustering
labs, centers = cluster_kmeans(data_filtered, 3, 0.5)

# total distance
print 'distance: {} km'.format(estimate_distance(data))

# number of clusters
print 'n cluster: {}'.format(centers.shape[0])

# entropy
ent = estimate_entropy(labs)
print 'entropy: {}'.format(ent)

# normalized entropy
print 'normalized entropy: {}'.format(ent/np.log(centers.shape[0]))

# homestay
print 'homestay: {}'.format(estimate_homestay(np.array(data_filtered[0]), labs))

# location variance
print 'location variance - pre-filter: {}'.format(estimate_variance(data))
print 'location variance - post-filter: {}'.format(estimate_variance(data_filtered))



# In[11]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.figure(figsize=(12,12))
plt.plot(data[2],data[1],'.',color=(.5,.5,.5),markersize=2)
plt.plot(data_filtered[2],data_filtered[1],'.',markersize=2,color=(0,0,1))
plt.plot(centers[:,1],centers[:,0] ,'.',markersize=2, color=(1,0,0))


# In[12]:

plt.figure(figsize=(12,12))
plt.plot(labs,'.',color=(0,0,0),markersize=5)
plt.ylim([-1,centers.shape[0]])


# In[7]:

import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '/home/sohrob/Dropbox/Code/MobileDepression/features')
from act_percentage import act_percentage

data = pd.read_csv('/home/sohrob/Dropbox/Data/CS120/1022235/act.csv',sep='\t',header=None)

# activity

print 'activity percentages: {}'.format(act_percentage(data))


# In[8]:

np.sum(act_percentage(data))

