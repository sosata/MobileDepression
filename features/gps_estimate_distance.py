import numpy as np

def estimate_distance(data):

	if data.shape[0]<=1:
	    d = 0;
	else:
	    d = np.sum(np.sqrt(np.power(np.diff(data[1]),2)+np.power(np.diff(data[2]),2)))
	return d