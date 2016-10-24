import numpy as np
cats = np.array(['IN_VEHICLE','ON_BICYCLE','ON_FOOT','RUNNING','STILL','TILTING','UNKNOWN','WALKING'])

def act_percentage(data):

	time = data[0]
	act = data[1]
	
	cat_per = np.zeros([cats.size])
	delta_t = time[time.size-1] - time[0]

	for i in np.arange(0,time.size-1):

		ind = np.where(cats==act[i])[0]
		cat_per[ind] += time[i+1]-time[i]

	cat_per /= float(delta_t)

	return cat_per