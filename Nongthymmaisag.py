import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import matplotlib as mlab
from matplotlib import cm
import random
import plot_beam_function as plt_beam

count = 8
sag = np.linspace(1.0,8.0,count)
max_tension_top = []
max_compression_top = []
max_tension_bot = []
max_compression_bot = []
max_deflection = []
sag_plot = []
max_eff_t_top = []
max_eff_t_bottom = []
max_eff_c_top = []
max_eff_c_bottom = []

for s in range(len(sag)):
	sag_value = sag[s]/4.
	bot_coord = sag[s]/841*(xs**2)-sag[s]/29*xs

	### Post-Processing ###

	# Bottom chord
	normal_bot_u = normal_s1[0,0:len(xs)-1]
	normal_bot_u = np.append(normal_bot_u,normal_s1[1,len(xs)-2])
	normal_bot_u /= 1000.
	normal_bot_l = normal_s2[0,0:len(xs)-1]
	normal_bot_l = np.append(normal_bot_l,normal_s2[1,len(xs)-2])
	normal_bot_l /= 1000.

	bottom_stress = [np.amax(normal_bot_u), np.amin(normal_bot_u), np.amax(normal_bot_l), np.amin(normal_bot_l)]
	eff_tbot = np.array(bottom_stress)/T_allow_MB
	print(eff_tbot)
	if bottom_stress[1] > 0 and bottom_stress[3] < 0:
		eff_cbot = [0.,bottom_stress[3]/C_allow_MB]
		bottom_stress=[np.amax(normal_bot_u), 0., np.amax(normal_bot_l), np.amin(normal_bot_l)]
	if bottom_stress[3] > 0 and bottom_stress[1] < 0:
		eff_cbot = [bottom_stress[1]/C_allow_MB,0.]
		bottom_stress=[np.amax(normal_bot_u), np.amin(normal_bot_u), np.amax(normal_bot_l), 0.]
	if bottom_stress[1] and bottom_stress[3] > 0:
		eff_cbot = [0.,0.]
		bottom_stress=[np.amax(normal_bot_u), 0., np.amax(normal_bot_l), 0.]
	if bottom_stress[1] < 0 and bottom_stress[3] < 0:
		eff_cbot = [bottom_stress[1]/C_allow_MB,bottom_stress[3]/C_allow_MB]
		bottom_stress=[np.amax(normal_bot_u), np.amin(normal_bot_u), np.amax(normal_bot_l), np.amin(normal_bot_l)]

	# Top chord
	normal_top_u = normal_s1[0,len(xs)-1:(2*len(xs)-2)]
	normal_top_u = np.append(normal_top_u,normal_s1[1,2*len(xs)-3])
	normal_top_u /= 1000.
	normal_top_l = normal_s2[0,len(xs)-1:(2*len(xs)-2)]
	normal_top_l = np.append(normal_top_l,normal_s2[1,2*len(xs)-3])
	normal_top_l /= 1000.

	top_stress = [np.amax(normal_top_u), np.amin(normal_top_u), np.amax(normal_top_l), np.amin(normal_top_l)]
	eff_ttop = np.array(top_stress)/T_allow_MB
	print(eff_ttop)
	if top_stress[1] > 0 and top_stress[3] < 0:
		eff_ctop = [0.,top_stress[3]/C_allow_MB]
		top_stress = [np.amax(normal_top_u), 0., np.amax(normal_top_l), np.amin(normal_top_l)]
	if top_stress[3] > 0 and top_stress[1] < 0:
		eff_ctop = [top_stress[1]/C_allow_MB,0.]
		top_stress = [np.amax(normal_top_u), np.amin(normal_top_u), np.amax(normal_top_l), 0.]
	if top_stress[1] and top_stress[3] > 0:
		eff_ctop = [0.,0.]
		top_stress = [np.amax(normal_top_u), 0., np.amax(normal_top_l), 0.]
	if top_stress[1] < 0 and top_stress[3] < 0:
		eff_ctop = [top_stress[1]/C_allow_MB,top_stress[3]/C_allow_MB]
		top_stress = [np.amax(normal_top_u), np.amin(normal_top_u), np.amax(normal_top_l), np.amin(normal_top_l)]

	if cables == True:
		axial_cables = axial_s[0,3*len(xs)-4:3*len(xs)-2]*-1/1000.
		print("Cable stresses")
		print(axial_cables)
		print("Cable efficiency")
		if Likely == True:
			print(np.max(axial_cables)/T_allow_MB)
		else:
			print(np.max(axial_cables)/T_allow_UB)

	max_tension_top = np.append(max_tension_top,np.amax(top_stress))
	max_compression_top = np.append(max_compression_top,np.amin(top_stress))
	max_tension_bot = np.append(max_tension_bot,np.amax(bottom_stress))
	max_compression_bot = np.append(max_compression_bot,np.amin(bottom_stress))
	max_deflection = np.append(max_deflection,np.amin(u)*-1)
	sag_plot = np.append(sag_plot,sag_value)
	max_eff_t_bottom = np.append(max_eff_t_bottom,np.amax(eff_tbot))
	max_eff_t_top = np.append(max_eff_t_top,np.amax(eff_ttop))
	max_eff_c_bottom = np.append(max_eff_c_bottom,np.amax(eff_cbot))
	max_eff_c_top = np.append(max_eff_c_top,np.amax(eff_ctop))


fig, ax = plt.subplots()
plt.axis([0, 2.25, 0, 45])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.plot(sag_plot,max_tension_top,'co', label = 'Top chord')
ax.plot(sag_plot,max_tension_bot,'mo', label = 'Bottom chord')
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Relationship Between Maximum Tension in Chords and Bottom Chord Sag \n Full Pedestrian Load')
ax.set_xlabel('Sag (m)')
ax.set_ylabel('Stress (MPa)')
plt.show()

fig, ax = plt.subplots()
plt.axis([0, 2.25, 0, 14])
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.plot(sag_plot,max_compression_top*-1,'co', label = 'Top chord')
ax.plot(sag_plot,max_compression_bot*-1,'mo', label = 'Bottom chord')
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Relationship Between Maximum Compression in Chords and Bottom Chord Sag \n Full Pedestrian Load')
ax.set_xlabel('Sag (m)')
ax.set_ylabel('Stress (MPa)')
plt.show()

fig, ax = plt.subplots()
plt.axis([0, 2.25, 0, 0.8])
ax.plot(sag_plot,max_deflection,'bo')
ax.set_title('Relationship Between Maximum Deflection of Bridge and Bottom Chord Sag \n Full Pedestrian Load')
ax.set_xlabel('Sag (m)')
ax.set_ylabel('Deflection (m)')
plt.show()

fig, ax = plt.subplots()
plt.axis([0, 2.25, 0, 1.75])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.plot(sag_plot,max_eff_t_top,'bo', label = 'Top chord')
ax.plot(sag_plot,max_eff_t_bottom,'ro', label ='Bottom chord')
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Relationship Between Tensile Efficiency and Bottom Chord Sag \n Full Pedestrian Load')
ax.set_xlabel('Sag (m)')
ax.set_ylabel('Efficiency')
plt.show()

fig, ax = plt.subplots()
plt.axis([0, 2.25, 0, 1.75])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.plot(sag_plot,max_eff_c_top,'bo', label = 'Top chord')
ax.plot(sag_plot,max_eff_c_bottom,'ro', label ='Bottom chord')
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Relationship Between Compressive Efficiency and Bottom Chord Sag \n Full Pedestrian Load')
ax.set_xlabel('Sag (m)')
ax.set_ylabel('Efficiency')
plt.show()
