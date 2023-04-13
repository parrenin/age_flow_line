import	numpy	as	np 
import yaml

# Default values for parameters, to prevent spyder errors
max_depth = 3310.
imax = 100
delta = 0.08
x_right = 370
iso_spacing = 20000.
beta = 0.015
thickness = 3767.

yamls = open('parameters.yml').read()
para = yaml.load(yamls, Loader=yaml.FullLoader)
globals().update(para)

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

# Lliboutry parameter
x_p = np.loadtxt('input_data/p_geodata.txt', usecols=(0,))
p_measure = np.loadtxt('input_data/p_geodata.txt', usecols=(1,))


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

x = np.arange(x_right+1)

a0	=	np.interp(x, x_a0, a0_measure) 
m	=	np.interp(x, x_m, m_measure)
s	=	np.interp(x, x_s, s_measure)
p	=	np.interp(x, x_p, p_measure)
Su	=	np.interp(x, x_Salamatin, Su_measure)
B	=	np.interp(x, x_Salamatin , B_measure)
Y	=	np.interp(x, x_Y , Y_measure)    



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
np.savetxt('interpolation_results/p_interpolated.txt', p, fmt = "%s")
np.savetxt('interpolation_results/Su_interpolated.txt', Su, fmt = "%s")
np.savetxt('interpolation_results/B_interpolated.txt', B, fmt = "%s")
np.savetxt('interpolation_results/Y_interpolated.txt', Y, fmt = "%s")


