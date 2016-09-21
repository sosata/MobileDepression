import numpy as np

def estimate_entropy(lab):

	ent = 0
	lab_uniq = np.unique(lab)
	for (i,lab_u) in enumerate(lab_uniq):
		p = np.sum(lab==lab_u)/float(lab.size)
		ent = ent - p*np.log(p)

	return ent
