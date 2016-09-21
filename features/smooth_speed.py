import numpy as np

def smooth_speed(signal):

	sigma = 5;
	sz = 50;
	gauss = np.exp(-np.power(np.linspace(-sz/2,sz/2,num=sz),2)/(2*sigma^2));
	gauss = gauss/np.sum(gauss);

	signal_out = np.convolve(signal, gauss, mode='same');

	return signal_out