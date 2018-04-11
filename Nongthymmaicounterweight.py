import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import matplotlib as mlab
from matplotlib import cm
import random
import plot_beam_function as plt_beam

count = 50
counter = np.linspace(0.001,0.05,50)
max_tension_top = []
max_compression_top = []
max_tension_bot = []
max_compression_bot = []
max_deflection = []
counter_plot = []
max_eff_t_top = []
max_eff_t_bottom = []
max_eff_c_top = []
max_eff_c_bottom = []

for v in range(len(counter)):

	# Given load
	P = np.zeros(num_dof)
	if UB == True:
		self_wt = 7.80
	if LB == True:
		self_wt = 13.7
	else:
		self_wt = (13.7+7.8)/2.
	for d in range(len(xs)):
		if d != 0 and d != len(xs)-1:
			if d < int(round(len(xs)/2.)):
				if Full == True:
					P[3*d+1] = -bridge_l*0.3/2*(4.1)/len(xs)-self_wt*(bridge_l/(len(xs)-1)*((A2_el[d]+A2_el[d-1])/2.+(A_el[d]+A_el[d-1])/2.)+A_2*ht)
				else:
					P[3*d+1] =-self_wt*(bridge_l/(len(xs)-1)*((A2_el[d]+A2_el[d-1])/2.+(A_el[d]+A_el[d-1])/2.)+A_2*ht)
			else:
				if Full == True:
					P[3*d+1] = -bridge_l*0.3/2*(4.1)/len(xs)-self_wt*(bridge_l/(len(xs)-1)*((A2_el[d]+A2_el[d-1])/2.+(A_el[d]+A_el[d-1])/2.)+A_2*ht)-counter[v]
				else:
					P[3*d+1] =-self_wt*(bridge_l/(len(xs)-1)*((A2_el[d]+A2_el[d-1])/2.+(A_el[d]+A_el[d-1])/2.)+A_2*ht)-counter[v]
	if Single == True:
		d = int(round(len(xs)*2./4.,1))
		P[3*d+1] = -0.61/2.-self_wt*(bridge_l/(len(xs)-1)*((A2_el[d]+A2_el[d-1])/2.+(A_el[d]+A_el[d-1])/2.)+A_2*ht)
