import numpy as np

def estimate_variance(data):

	if data.shape[0]<2:
	    variance = np.nan
	else:
	    variance = np.log(np.nanvar(data[1]) + np.nanvar(data[2]));
	    
	return variance