#−*−coding: utf−8−*− 
import scipy as sp 
import	numpy	as	np 
from	scipy . interpolate	import	interp1d 
import	matplotlib . pyplot	as	plt
from	matplotlib . pyplot	import	*



#-----------------------------------------------------
#Loading files for Geographic data, arrays creations
#-----------------------------------------------------

# Steady accumulation
x_a0 = np.loadtxt('input_data/a0_geodata.txt', usecols=(0,))
a0_measure =  np.loadtxt('input_data/a0_geodata.txt', usecols=(1,))

# Melting
x_m = np.loadtxt('input_data/m_geodata.txt', usecols=(0,))
m_measure = np.loadtxt('input_data/m_geodata.txt', usecols=(1,))

# Sliding rate
x_s = np.loadtxt('input_data/s_geodata.txt', usecols=(0,))
s_measure = np.loadtxt('input_data/s_geodata.txt', usecols=(1,))

# Surface and Bedrock
x_Salamatin = np.loadtxt('input_data/Geographic_data_from_Salamatin_et_al.txt', usecols=(1,))
Su_measure = np.loadtxt('input_data/Geographic_data_from_Salamatin_et_al.txt', usecols=(2,))
B_measure = np.loadtxt('input_data/Geographic_data_from_Salamatin_et_al.txt', usecols=(3,))

# Tube width
x_Y = np.loadtxt('input_data/Y_geodata.txt', usecols=(0,))
Y_measure = np.loadtxt('input_data/Y_geodata.txt', usecols=(1,))



#--------------------
#Interpolation
#--------------------

x = np.arange(371)

a0	=	interp1d (x_a0 , a0_measure) (x) 
m	=	interp1d (x_m , m_measure) (x)
s	=	interp1d ( x_s , s_measure) (x) 
Su	=	interp1d ( x_Salamatin , Su_measure) (x)
B	=	interp1d ( x_Salamatin , B_measure) (x)
Y	=	interp1d ( x_Y , Y_measure) (x)    



# Calcul du flux total Q

Q = np.zeros(len(x))

for i in range(1,len(Q)):
	Q[i] =  Q[i-1] + (x[i]-x[i-1]) * 1000 * a0[i-1] * Y[i-1] + \
	0.5 * (x[i]-x[i-1]) * 1000 * ( (a0[i]-a0[i-1]) * Y[i-1] + (Y[i]-Y[i-1]) * a0[i-1] ) + \
	(1./3) * (x[i]-x[i-1]) * 1000 * (a0[i]-a0[i-1]) * (Y[i]-Y[i-1])



# Calcul du "basal melting flux" Qm

Qm = [0]*len(x)

for i in range(1,len(Qm)):
  Qm[i] =  Qm[i-1] + 0.5 * (m[i]+m[i-1]) * 0.5 * (Y[i]+Y[i-1]) * (x[i]-x[i-1]) * 1000



#----------------------------------------
#Sauvegarde des données interpolées
#----------------------------------------

np.savetxt('interpolation_results/Q_fld.txt',Q, fmt = "%s")
np.savetxt('interpolation_results/Qm_fld.txt',Qm, fmt = "%s")
np.savetxt('interpolation_results/a0_interpolated.txt', a0, fmt = "%s")
np.savetxt('interpolation_results/m_interpolated.txt', m, fmt = "%s")
np.savetxt('interpolation_results/s_interpolated.txt', s, fmt = "%s")
np.savetxt('interpolation_results/Su_interpolated.txt', Su, fmt = "%s")
np.savetxt('interpolation_results/B_interpolated.txt', B, fmt = "%s")
np.savetxt('interpolation_results/Y_interpolated.txt', Y, fmt = "%s")


