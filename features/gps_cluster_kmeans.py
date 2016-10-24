# receives lat and long as numpy arrays

import numpy as np
import sklearn.cluster as skc
import warnings
warnings.filterwarnings("ignore")

def cluster_kmeans(data, n0, distance_max):

    if data.shape[0]==1:
        labs = 1
    else:
        distances_max_max = distance_max;
        n = n0;
        while distances_max_max>=distance_max:

            print n, distances_max_max
            km = skc.KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=1000, tol=0.0001, \
                precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=12)
            labs = km.fit_predict(data[[1,2]])
            centers = km.cluster_centers_

            distance_cluster_max = np.array([])
            for (i,cen) in enumerate(centers):
                ind = np.where(labs==i)[0]
                distance_cluster_max = np.append(distance_cluster_max,\
                    np.sqrt(np.max(np.power(cen[0]-data.loc[ind,1],2)+np.power(cen[1]-data.loc[ind,2],2))))
            
            distances_max_max = np.max(distance_cluster_max)

            n += 1

    return labs, centers