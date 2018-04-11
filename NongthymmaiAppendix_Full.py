import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import matplotlib as mlab
from matplotlib import cm
import random
import plot_beam_function as plt_beam


### Variables ###

# Configuration
diagonals = False
equilateral = False
divisions = 75											# must be odd if equilateral is true

# Bounds
UB = False
LB = False
Likely = True

# Load
Full = True
Single = False

### Elements ###

# Cross Sectional Properties #

# Top chord
a_matrix = np.zeros(divisions-1)
b_matrix = np.zeros(divisions-1)
A_el = np.zeros(divisions-1)
I_el = np.zeros(divisions-1)
for p in range(len(a_matrix)):
	a_matrix[p] = -0.032/(divisions-2)*p+0.07				# linearly vary cross-section starting at a = 7 cm
	b_matrix[p] = -0.02/(divisions-2)*p+0.045				# linearly vary cross-section starting at b = 4.5 cm
	A_el[p] = np.pi*a_matrix[p]*b_matrix[p]					# Assign area
	I_el[p] = (np.pi/4)*(a_matrix[p]**3)*b_matrix[p]		# Assign moment of inertia

# Bottom chord
A2_el = np.zeros(divisions-1)
I2_el = np.zeros(divisions-1)
for p in range(len(A2_el)):
	if p < int(round(divisions*0.5,1)):
		A2_el[p] = 2*A_el[p]								# assign 2A to western half
		I2_el[p] = 2*I_el[p]								# assign 2I to western half
	else:
		A2_el[p] = 3*A_el[p]								# assign 3A to western half
		I2_el[p] = 3*I_el[p]								# assign 3I to western half

# Woven members
r = 0.0254/2                                           		# radius of circular vertical members (m)
A_2 = np.pi*(r**2)                                    		# Area of circle = pi*r^2 (m^2)
I_2 = np.pi*(r**4)/4                                 		# I of circle = pi*r^4/4 (m^4)

# Material Properties #
E_1 = 7.6*(10**6)                                     		# Upper bound E in kPa
E_2 = 0.74*(10**6)											# Lower bound E in kPa
E_3 = 6.3*(10**6)											# Middle bound E in kPa
if LB == True:
	E_1 = E_2

# Upper bound strengths
C_strength_UB = -20.5										# Upper bound compressive strength in MPa
T_strength_UB = 61.5										# Upper bound tensile strength in MPa
C_allow_UB = 0.5*C_strength_UB								# Upper bound allowable compressive stress in MPa (SF = 2)
T_allow_UB = 0.5*T_strength_UB								# Upper bound allowable tensile stress in MPa (SF = 2)

# Lower bound strengths
C_strength_LB = -2.0										# Lower bound compressive strength in MPa
T_strength_LB = 4.4											# Lower bound tensile strength in MPa
C_allow_LB = 0.5*C_strength_LB								# Lower bound allowable compressive stress in MPa (SF = 2)
T_allow_LB = 0.5*T_strength_LB								# Lower bound allowable tensile stress in MPa (SF = 2)

# Middle bound strengths
C_strength_MB = -17.0										# Middle bound compressive strength in MPa
T_strength_MB = 40.8										# Middle bound tensile strength in MPa
C_allow_MB = 0.5*C_strength_MB								# Middle bound allowable compressive stress in MPa (SF = 2)
T_allow_MB = 0.5*T_strength_MB								# Middle bound allowable tensile stress in MPa (SF = 2)

# Coordinates #
bridge_l = 29.0
xs = np.linspace(0.0,bridge_l,divisions)
# 0.5 m sag
sag = 0.5
bot_coord = 2./841*(xs**2)-2./29*xs
# 1.0 m sag top
top_coord = 4./841*(xs**2)-4./29*xs+7./4.
# height at ends
ht = 1.75
# initialize counters
i = 0
l = 0
if equilateral == True or diagonals == True:
	coordinates = np.zeros(shape=(2*len(xs)+2*(len(xs)-2)+2,2))
	linked = np.zeros(shape=((2*len(xs)-2),2))
else:
	coordinates = np.zeros(shape=(2*len(xs)+2*(len(xs)-2),2))
	linked = np.zeros(shape=((2*len(xs)-4),2))
for t in range(2):
    for x in range(len(xs)):
		if t == 0:
			coordinates[i] = [xs[x],bot_coord[x]]
		else:
			coordinates[i] = [xs[x],top_coord[x]]
		if i!=0 and i!=(len(xs)-1):
			if i!=(len(xs)) and i!=(2*len(xs)-1):
				if i > len(xs)-1:
					coordinates[i+2*(len(xs))-3] = coordinates[i]
					linked[l] = [i,i+2*(len(xs))-3]
					l += 1
				else:
					coordinates[i+2*len(xs)-1] = coordinates[i]
					linked[l] = [i,i+2*len(xs)-1]
					l += 1
		if equilateral == True or diagonals == True:
			if i == len(xs):
				coordinates[4*len(xs)-4] = coordinates[i]
				linked[2*len(xs)-4] = [i,4*len(xs)-4]
			if i == 2*len(xs)-1:
				coordinates[4*len(xs)-3] = coordinates[i]
				linked[2*len(xs)-3] = [i,4*len(xs)-3]
		i += 1

# Element Properties #
elements = {};
p = 2*len(xs)-2
for i in range(len(xs)-1):
	if Likely == False:
		elements[i]={'A':A2_el[i],'E':E_1, 'I':I2_el[i], 0:i, 1:(i+1)}
	else:
		elements[i]={'A':A2_el[i],'E':E_3, 'I':I2_el[i], 0:i, 1:(i+1)}
for j in range(len(xs)-1,p):
	if Likely == False:
		elements[j]={'A':A_el[j-(len(xs)-1)],'E':E_1, 'I':I_el[j-(len(xs)-1)], 0:(j+1), 1:(j+2)}
	else:
		elements[j]={'A':A_el[j-(len(xs)-1)],'E':E_3, 'I':I_el[j-(len(xs)-1)], 0:(j+1), 1:(j+2)}
if diagonals == True:
	elements[p+len(xs)-2]={'A':A_2,'E':E_2, 'I':I_2, 0:4*len(xs)-4, 1:2*len(xs)}
	for k,w in zip(range(p,p+len(xs)-2),range(p+len(xs)-1,p+2*len(xs)-3)):
		elements[k]={'A':A_2,'E':E_2, 'I':I_2, 0:(k+2), 1:(k+len(xs))}
		if k+2 < (len(xs)-1)/2+(2*len(xs)-1):
			elements[w]={'A':A_2,'E':E_2, 'I':I_2, 0:(k+len(xs)), 1:k+3}
		else:
			if w != p+2*len(xs)-4:
				elements[w]={'A':A_2,'E':E_2, 'I':I_2, 0:(k+2), 1:(k+len(xs)+1)}
			else:
				elements[w]={'A':A_2,'E':E_2, 'I':I_2, 0:(k+2), 1:(k+len(xs)+2)}
if equilateral == True:
	elements[p] ={'A':A_2,'E':E_2, 'I':I_2, 0:4*len(xs)-4, 1:2*len(xs)}
	for d in range(p+1,p+len(xs)-2,2):
		elements[d]={'A':A_2,'E':E_2, 'I':I_2, 0:(d+1), 1:(d+len(xs))}
		elements[d+1]={'A':A_2,'E':E_2, 'I':I_2, 0:d+3, 1:d+len(xs)}
	elements[p+len(xs)-2]={'A':A_2,'E':E_2, 'I':I_2, 0:3*len(xs)-3, 1:4*len(xs)-3}
else:
	for s in range(p,p+len(xs)-2):
		if Likely == True:
			elements[s]={'A':A_2,'E':E_2, 'I':I_2, 0:(s+2), 1:(s+(len(xs)))}
		else:
			elements[s]={'A':A_2,'E':E_1, 'I':I_2, 0:(s+2), 1:(s+(len(xs)))}

# Total number of elements #
nel = len(elements)

# Element DOF #
ele_dof = 2													# 2 degrees of freedom (1 vector translation, 1 rotation)
nen = 2														# number of element nodes

### Element stiffness matrices ##

# Element stiffness matrix in global coordinates
def local_stiffness_frame(elts, crds, e):
    A, E, I = (elts[e]['A'], elts[e]['E'], elts[e]['I'])
    # Compute director vector between nodes
    n = np.array(crds[elts[e][1]])-np.array(crds[elts[e][0]])
	# Compute length of the element and normalize
    L = LA.norm(n)
    n /= L
    # Define the rotation operation
    R = np.array([[0,-1],[1,0]])
    # Compute normal
    s = np.dot(R,n)
    # Compute the coefficients
    Kfw = A*E/L*np.outer(n,n)+12*E*I/(L**3)*np.outer(s,s)
    kmt = 4*E*I/L
    khmt = 2*E*I/L
    kmw = 6*E*I/(L**2)*s
    kft = 6*E*I/(L**2)*s
    # The element stiffness
    space_dim = n.size
    n_nodes = 2
    n_dof = space_dim*n_nodes+n_nodes
    Ke = np.zeros((n_dof, n_dof))
    # Diagonal terms
    Ke[0:2,0:2] = Ke[3:5,3:5] = Kfw
    Ke[2,2] = Ke[5,5] = kmt
    # Non-diagonal terms
    Ke[0:2,2] = Ke[0:2,5] = kft
    Ke[0:2,3:5] = -Kfw
    Ke[2,3:5] = -kmw
    Ke[2,5] = khmt
    Ke[3:5,5] = -kft
    # Make symmetric
    lower_indices = np.tril_indices(n_dof,-1)
    Ke[lower_indices] = 0.
    Ke += np.triu(Ke,1).T
    return Ke

# Element stiffness matrix in local coordinates
def l_stiffness_frame(elts, crds, e):
	A, E, I = (elts[e]['A'], elts[e]['E'], elts[e]['I'])
	# Compute director vector between nodes
	n = np.array(crds[elts[e][1]])-np.array(crds[elts[e][0]])
	# Compute length of the element and normalize
	L = LA.norm(n)
	# Compute the coefficients
	a = E*A/L
	b = 12.*E*I/(L**3)
	c = 6.*E*I/(L**2)
	d = 4.*E*I/L
	e = 2.*E*I/L
	space_dim = 2
	n_nodes = 2
	n_dof = space_dim*n_nodes+n_nodes
	ke = np.zeros((n_dof, n_dof))
	ke[0,:]= [a, 0, 0, -a, 0, 0]
	ke[1,:]= [0, b, c, 0, -b, c]
	ke[2,:]= [0, c, d, 0, -c, e]
	ke[3,:]= [-a, 0, 0, a, 0, 0]
	ke[4,:]= [0, -b, -c, 0, b, -c]
	ke[5,:]= [0, c, e, 0, -c, d]
	return ke


### Global stiffness matrix ###

# Mesh
if equilateral == True or diagonals == True:
	nnodes = len(xs)*2+(2*(len(xs)-2))+2						# number of nodes
else:
	nnodes = len(xs)*2+(2*(len(xs)-2))
space_dim = 2
local_dof = space_dim+(space_dim-1)
num_dof = space_dim*nnodes+nnodes*(space_dim-1)					# total number of degrees of freedom in the system

KG = np.zeros((num_dof,num_dof))

# Loop over all elements
for e in range(nel):
    # Obtain the element stiffness matrix
    KE = local_stiffness_frame(elements, coordinates, e)
    # Assemble the global stiffness matrix
    for p in range(ele_dof):
        global_p = elements[e][p]
        for q in range(ele_dof):
            global_q = elements[e][q]
            KG[global_p*local_dof:(global_p+1)*local_dof,global_q*local_dof:(global_q+1)*local_dof] += KE[p*local_dof:(p+1)*local_dof,q*local_dof:(q+1)*local_dof]


### Boundary Conditions ###

# Nodes of known displacement
bc = np.zeros(12)
t = 3*len(xs)-3
s = 6*len(xs)-3
bc = [0,1,2,t,t+1,t+2,t+3,t+4,t+5,s,s+1,s+2]
#bc = [0,1,t,t+1,t+3,t+4,s,s+1]									# Uncomment for pinned ends


# Define load
P = np.zeros(num_dof)
# self weight
if UB == True:
	self_wt = 7.80
if LB == True:
	self_wt = 13.7
else:
	self_wt = (13.7+7.8)/2.
# apply pedestrian loads and self-weight for different load cases
for d in range(len(xs)):
    if d != 0 and d != len(xs)-1:
		if Full == True:
			P[3*d+1] = -bridge_l*0.3/2*(4.1)/len(xs)-self_wt*(bridge_l/(len(xs)-1)*((A2_el[d]+A2_el[d-1])/2.+(A_el[d]+A_el[d-1])/2.)+A_2*ht)
		else:
			P[3*d+1] =-self_wt*(bridge_l/(len(xs)-1)*((A2_el[d]+A2_el[d-1])/2.+(A_el[d]+A_el[d-1])/2.)+A_2*ht)
if Single == True:
	d = int(round(len(xs)*2./4.,1))
	P[3*d+1] = -0.61/2.-self_wt*(bridge_l/(len(xs)-1)*((A2_el[d]+A2_el[d-1])/2.+(A_el[d]+A_el[d-1])/2.)+A_2*ht)

print(P)

# Updated force matrix
F = P

# Initialize a new matrix with KG values
K = KG.copy()

# Apply hinge conditions to KG
delete = []
for w in range(len(linked)):
    for j in range(2):
		x = [int(3*linked[w,0])+j,int(3*linked[w,1])+j]
		if equilateral == False or linked[w,1]%2 == 0:
			K[x[0],:] += K[x[1],:]
			K[x[1],:] *= 0
			K[x[1],x[1]] = 1
			K[x[1],x[0]] = -1
			F[x[0]] += F[x[1]]
			F[x[1]] = 0
		else:
			delete.append(x[1])
			if j == 1:
				s = int(3*linked[w,1])+2
				delete.append(s)

# Delete unused doubled degrees of freedom
if equilateral == True:
	K = np.delete(K, delete, axis=0)
	K = np.delete(K, delete, axis=1)
	F = np.delete(F, delete, axis=0)
	KG = np.delete(KG, delete, axis=0)
	KG = np.delete(KG, delete, axis=1)

# Updated stiffness matrix
for b in bc:
    K[b,:] *=0
    K[b,b] = 1
    F[b] = 0

### Solutions ###

# Displacements
u = LA.solve(K,F.T)

print("Displacements")
print(u)

# Reactions
R = np.dot(KG,u)

print("Reactions")
print(R)

### Post-Processing ###

# Define Ue, displacement vectors for each element at nodes
# Reinsert deleted doubled degrees of freedom for equilateral case
if equilateral == True:
	for d in delete:
		u = np.insert(u,d,0.)
		if d > len(u):
			u.append(0.)
Ue = np.zeros((2*local_dof,nel))
for e in range(nel):
	n1 = elements[e][0]
	n2 = elements[e][1]
	Ue[0,e] = u[3*n1]
	Ue[1,e] = u[3*n1+1]
	Ue[2,e] = u[3*n1+2]
	Ue[3,e] = u[3*n2]
	Ue[4,e] = u[3*n2+1]
	Ue[5,e] = u[3*n2+2]

# Define Qe to rotate back to local coordinates
def rotation_frame(elts, crds, e):
	v = np.array(crds[elts[e][1]])-np.array(crds[elts[e][0]])
	Qe = np.zeros((local_dof*nen,local_dof*nen))
	L = LA.norm(v)
	v /= L
	if space_dim == 2:
		Qe[0,0] = v[0]
		Qe[0,1] = v[1]
		Qe[1,0] = -v[1]
		Qe[1,1] = v[0]
		Qe[2,2] = 1.0
		Qe[3,3] = v[0]
		Qe[3,4] = v[1]
		Qe[4,3] = -v[1]
		Qe[4,4] = v[0]
		Qe[5,5] = 1.0
	return Qe

# Initialize internal force and stress matrices
force = np.zeros((2*local_dof, nel))
axial_f = np.zeros((nen, nel))
shear_f = np.zeros((nen, nel))
moment_f = np.zeros((nen, nel))
axial_s = np.zeros((nen, nel))
moment_s = np.zeros((nen, nel))
normal_s1 = np.zeros((nen, nel))
normal_s2 = np.zeros((nen, nel))
ke = np.zeros((local_dof, local_dof))
y = 0.07														# definition of a is conservative to account for use of multiple members
# Loop through elements
for e in range(nel):
	ke = l_stiffness_frame(elements, coordinates, e)
	Qe = rotation_frame(elements, coordinates, e)
	ue = np.matmul(Qe,Ue[:,e])
	angle = np.arctan(coordinates[elements[e][0]])
	force[:,e] = np.matmul(ke,ue)
	# Store internal forces and stresses for each element
	axial_f[0,e] = force[0,e]
	axial_s[0,e] = axial_f[0,e]/elements[e]['A']
	axial_f[1,e] = force[3,e]
	axial_s[1,e] = axial_f[1,e]/elements[e]['A']
	shear_f[0,e] = force[1,e]
	shear_f[1,e] = force[4,e]
	moment_f[0,e] = force[2,e]
	moment_s[0,e] = moment_f[0,e]*y/elements[e]['I']
	moment_f[1,e] = force[5,e]
	moment_s[1,e] = moment_f[1,e]*y/elements[e]['I']
	normal_s1[0,e] = -1*axial_s[0,e]+moment_s[0,e]
	normal_s1[1,e] = axial_s[1,e]+moment_s[1,e]*-1
	normal_s2[0,e] = -1*axial_s[0,e]-moment_s[0,e]
	normal_s2[1,e] = axial_s[1,e]-moment_s[1,e]*-1

# Print desired outputs
print("Normal stress 1 max")
print(np.amax(normal_s1)/1000.)
print("Normal stress 1 min")
print(np.amin(normal_s1)/1000.)
print("Normal stress 2 max")
print(np.amax(normal_s2)/1000.)
print("Normal stress 2 min")
print(np.amin(normal_s2)/1000.)
print ("Max deflection")
print(np.amin(u))
print("T Efficiency w/ SF = 2")
eff_t = np.zeros(2)
eff_c = np.zeros(2)
if UB == True:
	eff_t[0]=np.amax(normal_s1[0:2*len(xs)-2])/1000./T_allow_UB
	eff_t[1]=np.amax(normal_s2[0:2*len(xs)-2])/1000./T_allow_UB
	eff_c[0]=np.amin(normal_s1[0:2*len(xs)-2])/1000./C_allow_UB
	eff_c[1]=np.amin(normal_s2[0:2*len(xs)-2])/1000./C_allow_UB
	eff_vt=np.amax(axial_s[0,2*len(xs)-2:3*len(xs)-3]*-1)/1000./T_allow_UB
	eff_vc=np.amin(axial_s[0,2*len(xs)-2:3*len(xs)-3]*-1)/1000./C_allow_UB
if LB == True:
	eff_t[0]=np.amax(normal_s1[0:2*len(xs)-2])/1000./T_allow_LB
	eff_t[1]=np.amax(normal_s2[0:2*len(xs)-2])/1000./T_allow_LB
	eff_c[0]=np.amin(normal_s1[0:2*len(xs)-2])/1000./C_allow_LB
	eff_c[1]=np.amin(normal_s2[0:2*len(xs)-2])/1000./C_allow_LB
	eff_vt=np.amax(axial_s[0,2*len(xs)-2:3*len(xs)-3]*-1)/1000./T_allow_LB
	eff_vc=np.amin(axial_s[0,2*len(xs)-2:3*len(xs)-3]*-1)/1000./C_allow_LB
if Likely == True:
	eff_t[0]=np.amax(normal_s1[0:2*len(xs)-2])/1000./T_allow_MB
	eff_t[1]=np.amax(normal_s2[0:2*len(xs)-2])/1000./T_allow_MB
	eff_c[0]=np.amin(normal_s1[0:2*len(xs)-2])/1000./C_allow_MB
	eff_c[1]=np.amin(normal_s2[0:2*len(xs)-2])/1000./C_allow_MB
	eff_vt=np.amax(axial_s[0,2*len(xs)-2:3*len(xs)-3]*-1)/1000./T_allow_LB
	eff_vc=np.amin(axial_s[0,2*len(xs)-2:3*len(xs)-3]*-1)/1000./C_allow_LB
print(np.amax(np.absolute(eff_t)))
print("C Efficiency w/ SF = 2")
print(np.amax(np.absolute(eff_c)))
print("T Efficiency of verticals w/ SF = 2")
print(eff_vt)
print("C Efficiency of verticals w/ SF = 2")
print(eff_vc)

### Internal Forces ###

# Bottom chord
shear_bot = shear_f[0,0:len(xs)-1]
shear_bot = np.append(shear_bot,shear_f[1,len(xs)-2]*-1)
axial_bot = axial_f[0,0:len(xs)-1]*-1
axial_bot = np.append(axial_bot,axial_f[1,len(xs)-2])
moment_bot = moment_f[0,0:len(xs)-1]
moment_bot = np.append(moment_bot,moment_f[1,len(xs)-2]*-1)

# Top chord
shear_top = shear_f[0,len(xs)-1:(2*len(xs)-2)]
shear_top = np.append(shear_top,shear_f[1,2*len(xs)-3]*-1)
axial_top = axial_f[0,len(xs)-1:(2*len(xs)-2)]*-1
axial_top = np.append(axial_top,axial_f[1,2*len(xs)-3])
moment_top = moment_f[0,len(xs)-1:(2*len(xs)-2)]
moment_top = np.append(moment_top,moment_f[1,2*len(xs)-3]*-1)

# Middle chord
p = 2*len(xs)-2
if equilateral == True:
	axial_mid = axial_f[0,2*len(xs)-2:3*len(xs)-3]*-1
if diagonals == True:
	axial_mid = axial_f[0,2*len(xs)-2:p+2*len(xs)-5]*-1
else:
	axial_mid = axial_f[0,2*len(xs)-2:3*len(xs)-4]*-1

# Moments about centroid of stiffness #
na = np.zeros(len(xs))
moment_na = np.zeros(len(xs))
print(len(moment_bot))
for p in range(len(xs)):
	if p == len(xs)-1:
		xi = coordinates[elements[p+len(xs)-2][1],1]-coordinates[elements[p-1][1],1]+a_matrix[p-1]
		na[p] = (xi*A_el[p-1]+a_matrix[p-1]*A2_el[p-1])/(A_el[p-1]+A2_el[p-1])
		moment_na[p] = moment_bot[p]+moment_top[p]+axial_bot[p]*-1*(na[p]-a_matrix[p-1])-axial_top[p]*-1*(coordinates[elements[p+len(xs)-2][1],1]-na[p])
	else:
		xi = coordinates[elements[p+len(xs)-1][0],1]-coordinates[elements[p][0],1]+a_matrix[p]
		na[p] = (xi*A_el[p]+a_matrix[p]*A2_el[p])/(A_el[p]+A2_el[p])
		moment_na[p] = moment_bot[p]+moment_top[p]+axial_bot[p]*-1*(na[p]-a_matrix[p])-axial_top[p]*-1*(coordinates[elements[p+len(xs)-1][1],1]-na[p])

fig, ax = plt.subplots()
plt.axis([0, 29, -40, 50])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.plot(coordinates[0:len(xs),0],moment_na,'g', label = 'Moment about neutral axis')
if Full == False:
	ax.set_title('Nongthymmai Moment about Centroid of Stiffness \n Single Pedestrian')
else:
	ax.set_title('Nongthymmai Moment about Centroid of Stiffness \n Full Pedestrian')
ax.set_xlabel('Distance along bridge (m)')
ax.set_ylabel('Moment (kNm)')
plt.show()

# Shear and Axial Plot
fig, ax = plt.subplots()
plt.axis([0, 29, -10, 80])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.plot(coordinates[0:len(xs),0],axial_bot,'b', label = 'Axial bottom chord')
ax.plot(coordinates[0:len(xs),0],axial_top,'b--', label = 'Axial top chord')
ax.plot(coordinates[0:len(xs),0],shear_bot,'r', label = 'Shear bottom chord')
ax.plot(coordinates[0:len(xs),0],shear_top,'r--', label = 'Shear top chord')
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
if Full == False:
	ax.set_title('Nongthymmai Internal Forces \n Single Pedestrian')
else:
	ax.set_title('Nongthymmai Internal Forces \n Full Pedestrian')
ax.set_xlabel('Distance along bridge (m)')
ax.set_ylabel('Force (kN)')
plt.show()

# Moment plot
fig, ax = plt.subplots()
plt.axis([0, 29, -3, 3])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.plot(coordinates[0:len(xs),0],moment_top,'g--', label = 'Top chord')
ax.plot(coordinates[0:len(xs),0],moment_bot,'g', label = 'Bottom chord')
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
if Full == False:
	ax.set_title('Nongthymmai Internal Moments in Chords \n Single Pedestrian')
else:
	ax.set_title('Nongthymmai Internal Moments in Chords \n Full Pedestrian')
ax.set_xlabel('Distance along bridge (m)')
ax.set_ylabel('Moment (kNm)')
plt.show()

### Stresses ###

# Bottom chord normal stresses
normal_bot_u = normal_s1[0,0:len(xs)-1]
normal_bot_u = np.append(normal_bot_u,normal_s1[1,len(xs)-2])
normal_bot_u /= 1000.
normal_bot_l = normal_s2[0,0:len(xs)-1]
normal_bot_l = np.append(normal_bot_l,normal_s2[1,len(xs)-2])
normal_bot_l /= 1000.

fig, ax = plt.subplots()
plt.axis([0, 29, -20, 30])
ax.plot(coordinates[0:len(xs),0],normal_bot_u,'c', label = 'Upper max')
ax.plot(coordinates[0:len(xs),0],normal_bot_l,'m', label = 'Lower max')
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
if Full == False:
	ax.set_title('Nongthymmai Normal Stresses for Bottom Chord \n Single Pedestrian')
else:
	ax.set_title('Nongthymmai Normal Stresses for Bottom Chord \n Full Pedestrian')
ax.set_xlabel('Distance along bridge (m)')
ax.set_ylabel('Stress (MPa)')
plt.show()

# Top chord normal stresses
normal_top_u = normal_s1[0,len(xs)-1:(2*len(xs)-2)]
normal_top_u = np.append(normal_top_u,normal_s1[1,2*len(xs)-3])
normal_top_u /= 1000.
print("efficiency of top u end")
print(normal_top_u[0]/T_allow_UB)
normal_top_l = normal_s2[0,len(xs)-1:(2*len(xs)-2)]
normal_top_l = np.append(normal_top_l,normal_s2[1,2*len(xs)-3])
normal_top_l /= 1000.

# Plot
fig, ax = plt.subplots()
plt.axis([0, 29, -20, 40])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.plot(coordinates[0:len(xs),0],normal_top_u,'c--', label = 'Upper max')
ax.plot(coordinates[0:len(xs),0],normal_top_l,'m--', label = 'Lower max')
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
if Full == False:
	ax.set_title('Nongthymmai Normal Stresses for Top Chord \n Single Pedestrian')
else:
	ax.set_title('Nongthymmai Normal Stresses for Top Chord \n Full Pedestrian')
ax.set_xlabel('Distance along bridge (m)')
ax.set_ylabel('Stress (MPa)')
plt.show()
 
# Axial stress chords
axial_bot_s = axial_s[0,0:len(xs)-1]*-1
axial_bot_s = np.append(axial_bot_s,axial_s[1,len(xs)-2])
axial_bot_s /= 1000.
axial_top_s = axial_s[0,len(xs)-1:(2*len(xs)-2)]*-1
axial_top_s = np.append(axial_top_s,axial_s[1,2*len(xs)-3])
axial_top_s /= 1000.

# Plot
fig, ax = plt.subplots()
plt.axis([0, 29, -20, 40])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.plot(coordinates[0:len(xs),0],axial_top_s,'m--', label = 'Top chord')
ax.plot(coordinates[0:len(xs),0],axial_bot_s,'c--', label = 'Bottom chord')
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
if Full == False:
	ax.set_title('Nongthymmai Axial Stresses of Chords \n Single Pedestrian')
else:
	ax.set_title('Nongthymmai Axial Stresses of Chords \n Full Pedestrian')
ax.set_xlabel('Distance along bridge (m)')
ax.set_ylabel('Stress (MPa)')
plt.show()

# Woven elements
p = 2*len(xs)-2
if equilateral == True:
	axial_mid_s = axial_s[0,2*len(xs)-2:3*len(xs)-3]*-1
if diagonals == True:
	axial_mid_s = axial_s[0,2*len(xs)-2:p+2*len(xs)-5]*-1
else:
	axial_mid_s = axial_s[0,2*len(xs)-2:3*len(xs)-4]*-1
axial_mid_s /= 1000.

# Print stresses in the woven elements
print("Max Axial mid")
print(np.amax(axial_mid)/A_2)
print("Max axial mid stress")
print(np.amax(axial_mid_s))
print("Min axial mid stress")
print(np.amin(axial_mid_s))

fig, ax = plt.subplots()
plt.axis([0, 29, -8, 8])

if equilateral == False and diagonals == False:
	ax.plot(coordinates[1:len(xs)-1,0],axial_mid_s,'ko', label = 'Axial verticals')
if equilateral == True:
	ax.plot(coordinates[0:len(xs)-2,0],axial_mid_s,'ko')
if diagonals == True:
	xs_2 = np.linspace(0.0,bridge_l,divisions*2-1)
	coordinates_x = np.zeros(len(xs_2))
	i = 0
	for x in range(len(xs_2)):
		coordinates_x[i] = xs_2[x]
		i += 1
	coordinates_diag = coordinates_x[3:len(xs_2)-3:2]
	ax.plot(coordinates[1:len(xs)-1,0],axial_mid_s[0:len(xs)-2],'ko', label = 'Verticals')
	ax.plot(coordinates_diag,axial_mid_s[len(xs)-2:len(axial_mid_s)+1],'ro', label = 'Diagonals')
if Full == False:
	ax.set_title('Nongthymmai Axial Stresses of Vertical Members \n Single Pedestrian')
else:
	ax.set_title('Nongthymmai Axial Stresses of Vertical Members \n Full Pedestrian')
ax.set_xlabel('Distance along bridge (m)')
ax.set_ylabel('Stress (MPa)')
plt.show()

### Plotting Deformed and Undeformed Shape ###

u = u[:num_dof]

disp =np.zeros((space_dim*nnodes,1))

for i in range(nnodes):
    disp[i*space_dim:(i+1)*space_dim,0]=u[i*(space_dim+1):(space_dim*(i+1)+i)]

coordinates_sol = coordinates + disp.reshape(coordinates.shape)

fig = plt.figure(0)
plt.ion()
ax = fig.gca()
ax.clear()
ax.grid('off')
plt.axis([-1, 30, -5, 5])
color_list = []

for i in range(nel):
    this_color = np.array(([random.random(), random.random(), random.random(), 1]))
    color_list.append(this_color - np.array(([0,0,0,0])))
    x_i = elements[i][0]
    x_j = elements[i][1]
    if (i==0):
		ax.plot([coordinates[x_i][0], coordinates[x_j][0]],\
				[coordinates[x_i][1], coordinates[x_j][1]], color=[1,0,0,1], label='Undeformed Shape')
    else:
		ax.plot([coordinates[x_i][0], coordinates[x_j][0]],\
				[coordinates[x_i][1], coordinates[x_j][1]], color=[1,0,0,1])


plt_beam.plot_beam(ax, fig, nel,u,coordinates,elements)
# Tidy up the figure
ax.grid(True)
ax.legend(loc='upper left')
if Full == False:
	ax.set_title('Nongthymmai Deformed and Undeformed Shape \n Single Pedestrian')
else:
	ax.set_title('Nongthymmai Deformed and Undeformed Shape \n Full Pedestrian')
ax.set_xlabel('Horizontal coordinates (m)')
ax.set_ylabel('Vertical coordinates (m)')
plt.show(block=True)
