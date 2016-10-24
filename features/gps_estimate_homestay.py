from scipy import stats
import numpy as np

def estimate_homestay(time, labs):

	# finding the first 3 clusters

	cluster1 = stats.mode(labs)[0][0]
	ind1 = np.where(labs==cluster1)[0]
	hours1 = np.mod(time[ind1], 86400)/3600.0

	labs2 = labs
	labs2 = np.delete(labs2,ind1)
	cluster2 = stats.mode(labs2)[0][0]
	ind2 = np.where(labs==cluster2)[0]
	hours2 = np.mod(time[ind2], 86400)/3600.0

	labs3 = labs2
	labs3 = np.delete(labs3, ind2)
	cluster3 = stats.mode(labs3)[0][0]
	ind3 = np.where(labs==cluster3)[0]
	hours3 = np.mod(time[ind3], 86400)/3600.0

	# finding which cluster is mostly visited between 12am-6am

	hours1 = hours1[hours1<=6]
	hours2 = hours2[hours2<=6]
	hours3 = hours3[hours3<=6]

	cluster_home = np.argmax(np.array([hours1.size, hours2.size, hours3.size]))

	if cluster_home==0:
	    homestay = np.sum(labs==cluster1)/float(labs.size)
	elif cluster_home==1:
	    homestay = np.sum(labs==cluster2)/float(labs.size)
	elif cluster_home==2:
	    homestay = np.sum(labs==cluster3)/float(labs.size)

	return homestay