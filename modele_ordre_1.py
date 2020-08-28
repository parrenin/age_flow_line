#−*−coding: utf−8−*− 
from __future__ import division
import os  
import sys
import scipy as sp 
import  numpy as  np 
from  scipy . interpolate import  interp1d 
import  matplotlib . pyplot as  plt
from  matplotlib . lines  import *  
from math import exp
from math import log
from math import sqrt
from math import isnan



#----------------------------------------------------------
#Loading data from interpolation_results file
#----------------------------------------------------------

Q_fld = np.loadtxt('interpolation_results/Q_fld.txt')
Qm_fld = np.loadtxt('interpolation_results/Qm_fld.txt')
a0_fld = np.loadtxt('interpolation_results/a0_interpolated.txt')
m_fld = np.loadtxt('interpolation_results/m_interpolated.txt')
s_fld = np.loadtxt('interpolation_results/s_interpolated.txt')
Su_fld = np.loadtxt('interpolation_results/Su_interpolated.txt')
B_fld = np.loadtxt('interpolation_results/B_interpolated.txt')
Y_fld = np.loadtxt('interpolation_results/Y_interpolated.txt')



#-------------------------------------------------------------------------------
# Loading data from accu-prior.txt , density-prior.txt ... in "input_data" file
#-------------------------------------------------------------------------------

deut = np.loadtxt('input_data/deuterium.txt')
density_readarray = np.loadtxt('input_data/density-prior.txt')

x_s_geo = np.loadtxt('input_data/s_geodata.txt', usecols=(0,))
s_measure = np.loadtxt('input_data/s_geodata.txt', usecols=(1,))



#---------------------------------------------------------
# Executing model_parameters.py (imax, delta,...) 
#---------------------------------------------------------

exec(open('model_parameters.py').read())




#---------------------------------------------------
# DEPTH - 1D Vostok drill grid for post-processing
#---------------------------------------------------

depth_corrected = np.arange(0., max_depth + 0.1 , 1.)

depth_mid = (depth_corrected[1:] + depth_corrected[:-1])/2

depth_inter = (depth_corrected[1:] - depth_corrected[:-1])



#---------------------------------------------------------------------------
# Relative density interpolation with extrapolation of "depth-density" data
#---------------------------------------------------------------------------

D_depth = density_readarray[:,0]

D_D = density_readarray[:,1]

relative_density = np.interp(depth_mid, D_depth, D_D, right = 1.)

ie_depth = np.cumsum(np.concatenate((np.array([0]), relative_density * depth_inter)))



#----------------------------------------------------------
# DELTA H
#----------------------------------------------------------

DELTA_H = depth_corrected[-1] - ie_depth[-1]



#----------------------------------------------------------
# Mesh generation (pi,theta)
#----------------------------------------------------------

pi = np.linspace( - imax * delta, 0 ,  imax + 1)

theta = np.linspace( 0 , - imax * delta,  imax + 1)



#----------------------------------------------------------
# Total flux Q(m^3/yr)
#----------------------------------------------------------
 
Q = Q_fld[-1] * np.exp(pi) # Q_ref = Q_fld[-1] 

Q = np.insert(Q,0,0) # prolongement du vecteur pour les calculs à l'extrémité gauche du maillage



#----------------------------------------------------------
# OMEGA
#----------------------------------------------------------

OMEGA = np.exp(theta)



#----------------------------------------------------------
# interpolation of flow line data files for x, Qm, ...
#----------------------------------------------------------

x_fld = np.arange(x_right+1)

x = np.interp(Q, Q_fld , x_fld)
Qm = np.interp(Q, Q_fld , Qm_fld)
Y = np.interp(x, x_fld,Y_fld)
S = np.interp(x, x_fld, Su_fld)
B = np.interp(x, x_fld, B_fld)

B[0] = B[1] # Altitude du socle constante au niveau du dôme
S[0] = S[1] # Altitude de la surface constante au niveau du dôme



#--------------------------------------------------
# Accumulation a(m/yr)
#--------------------------------------------------

a = np.interp(Q, Q_fld, a0_fld)



#--------------------------------------------------
# Calcul de la surface ice-equivalent S_ie
#--------------------------------------------------

S_ie = S - DELTA_H



#--------------------------------------------------
# Melting
#--------------------------------------------------

m = np.interp(x, x_fld, m_fld)



#------------------------------------------------------
# Calculs de theta_min et theta_max
#------------------------------------------------------

theta_max = np.zeros(imax+1)

theta_min = np.where(Qm[1:] > 0, \
np.maximum(np.log(Qm[1:].clip(min=10**-100)/Q[1:]), theta[-1] * np.ones((imax+1,))) ,\
theta[-1] * np.ones((imax+1,))  ) 



#------------------------------------------------------
# Routine de calcul de omega(zeta)
#------------------------------------------------------

# Lliboutry model for the horizontal flux shape function

p = 3.0 * np.ones(imax+1) 

s = np.interp(x[1:], x_s_geo,s_measure)

zeta = np.linspace(1,0,1001).reshape(1001,1)

omega = zeta * s + (1-s) * ( 1 - (p+2)/(p+1) * (1-zeta) + 1/(p+1) * np.power(1-zeta,p+2)  ) 



#-------------------------------------------------------
# GRID
#-------------------------------------------------------

grid = np.ones( ( imax + 1 , imax + 2 ) )

grid[:,0] = grid[:,1] = np.where( theta >= theta_min[0], 1, 0 )


print('Before defining grid boolean')
for j in range(2, imax+2 ):
    grid[2:,j] = np.where(np.logical_and(theta[2:] >= theta_min[j-1], grid[1:-1,j-1] ==1), 1, 0)
print('After defining grid boolean')



#-------------------------------------------------------
# Matrice omega : mat_omega
#-------------------------------------------------------

mat_omega = np.where( grid[:,1:] == 1 ,\
( np.dot(OMEGA.reshape(imax+1,1),Q[1:].reshape(1,imax+1)) - Qm[1:] )\
/ (Q[1:] - Qm[1:]), np.nan )



#-------------------------------------------------------
# Matrice z_ie
#-------------------------------------------------------

z_ie = np.zeros( (imax+1, imax+2) )

print('Before defining z_ie')
for j in range(0,imax+1):
    inter = np.interp( -mat_omega[:,j], -omega[:,j].flatten(), zeta.flatten())
    z_ie[:,j+1] = np.where(grid[:,j+1] == 1, B[j+1] + inter * ( S_ie[j+1] - B[j+1] ), np.nan)
z_ie[:,0] = z_ie[:,1]
print('After defining z_ie')



#-------------------------------------------------------
# Matrice theta_min
#-------------------------------------------------------

mat_theta_min =  np.tile ( theta_min , (imax+1,1) )



#-------------------------------------------------------
# Matrice theta
#-------------------------------------------------------

mat_theta = np.where( grid[:,1:] == 1 , theta.reshape(imax+1,1) , np.nan )



#-------------------------------------------------------
# Matrice OMEGA : mat_OMEGA
#-------------------------------------------------------

mat_OMEGA = np.where( grid[:,1:] == 1 , OMEGA.reshape(imax+1,1) , np.nan )



#-------------------------------------------------------
# Matrice pi : mat_pi
#-------------------------------------------------------

mat_pi = np.where( grid[:,1:] == 1 , pi , np.nan )



#-------------------------------------------------------
# Matrice x : mat_x
#-------------------------------------------------------

mat_x = np.where( grid == 1, x, np.nan )



#-------------------------------------------------------
# Matrice depth_ie : mat_depth_ie
#-------------------------------------------------------

mat_depth_ie = np.where( grid == 1 , S_ie - z_ie , np.nan )

mat_depth_ie[0,:] = 0


#-------------------------------------------------------
# Matrice du flux q : mat_q
#-------------------------------------------------------

mat_q = np.where( grid[:,1:] == 1 , Q[1:] * mat_OMEGA , np.nan )

mat_q[:,0] = mat_q[0,0] # ligne de flux verticale au dôme


#-------------------------------------------------------
# Matrice a0 : mat_a0
#-------------------------------------------------------

mat_a0 = np.zeros((imax+1, imax+1))

mat_a0[0,:] = a[1:]

mat_a0[1:,0] = np.where(grid[1:,1] == 1, mat_a0[0,0], np.nan)


print('Before defining mat_a0')
for j in range(1,imax+1):
    mat_a0[1:,j] = np.where(grid[1:,j] ==1, mat_a0[:-1,j-1], np.nan)
print('After defining mat_a0')



 
#-------------------------------------------------------
# Matrice x0 : mat_x0
#-------------------------------------------------------

mat_x0 = np.zeros((imax+1, imax+1+1))

mat_x0[:,0] = np.where( grid[:,0] == 1 , 0, np.nan )

mat_x0 [0,1:] = x[1:]

mat_x0[:,1] = np.where( grid[:,1] == 1 , mat_x0[0][1] * mat_OMEGA[:,0], np.nan )

print('Before defining mat_x0')
for j in range(2,imax+1+1):
    mat_x0[1:,j] = np.where(grid[1:,j] ==1, mat_x0[:-1,j-1], np.nan)
print('After defining mat_x0')


#-------------------------------------------------------
# Matrice STEADY-AGE: 
#-------------------------------------------------------

#-----------------------------------------------------------------------------------------------
# Ordre 1
#-----------------------------------------------------------------------------------------------

print('Before calculation of steady age matrix.')

mat_steady_age = np.zeros((imax+1, imax+2))

for i in range(1,imax+1):
  if grid[i][1] == 1 :
    mat_steady_age[i][1] = mat_steady_age[i-1][1] + delta / a[1] \
    * (z_ie[i-1][1] - z_ie[i][1]) / ( OMEGA[i-1] - OMEGA[i] )
  else:
    mat_steady_age[i][1] = np.nan

mat_steady_age[:,0] = mat_steady_age[:,1]


for j in range(2,imax+2):
    c = (a[j] - a[j-1]) / delta 
    d = a[j] - c * pi[j-1]
    e = ((z_ie[1:,j] - z_ie[:-1,j]) / (OMEGA[1:] - OMEGA[:-1]) - (z_ie[:-1,j-1] - z_ie[1:,j-1]) / (OMEGA[:-1] - OMEGA[1:]) ) / delta
    f = (z_ie[1:,j] - z_ie[:-1,j]) / ( OMEGA[1:] - OMEGA[:-1] ) - e * pi[j-1]
    if c == 0:
        mat_steady_age[1:,j] = np.where(grid[1:, j] == 1, mat_steady_age[:-1,j-1] + (1 / d) * ( e * (pi[j-1]*pi[j-1] - pi[j-2]*pi[j-2]) / 2 + f * delta ), np.nan)
    else:
        mat_steady_age[1:,j] = np.where(grid[1:, j] == 1, mat_steady_age[:-1,j-1] + (e * pi[j-1] + f) * log (abs ( c * pi[j-1] + d ) ) / c - \
                    (e * pi[j-2] + f) * log ( abs ( c * pi[j-2] + d ) ) / c - \
                    ( e / c ) * ( ( pi[j-1] + d/c ) * log (abs(c * pi[j-1] + d ) )-\
                    ( pi[j-2] + d/c ) * log (abs (c * pi[j-2] + d )) - delta  ), np.nan) 

print('After calculation of steady age matrix.')


# Rq: NaN dans coeff e dû à l'utilisation d'un schéma aux différences finies avant
# z_ie[ imax-1, imax] = NaN, pas de répercutions néfastes.

# Try to remove loops 



#-------------------------------------------------------
# Matrice de la fonction d'amincissement tau_ie
#-------------------------------------------------------

tau_ie = np.where( grid[1:,1:] == 1, ( z_ie[:-1,1:] - z_ie[1:,1:] ) \
/ ( mat_steady_age[1:,1:] - mat_steady_age[:-1,1:] ) \
/ mat_a0[:-1,:] , np.nan)



#----------------------------------------------------------
#  Post-processing: transfert des résulats du modèle sur 
#  la grille 1D du forage Vostok.
#----------------------------------------------------------



#----------------------------------------------------------
#  Calcul de theta Vostok IceCore : theta_vic
#----------------------------------------------------------

if mat_depth_ie[imax,imax+1] < ie_depth[len(ie_depth)-1]:
	sys.exit("\n Attention problème d'interpolation post-processing:\n \
        le modèle donne des résultats jusqu'à une profondeur maximale trop \n \
        faible par rapport à la profondeur maximale du forage Vostok \n \
        Pour transposer correctement les résultats \n \
	      du maillage sur toute la hauteur du forage Vostok il faut augmenter \n \
        le nombre de noeuds du maillage.")

theta_vic = np.log(interp1d( mat_depth_ie[:,imax+1][~np.isnan(mat_depth_ie[:,imax+1])] , \
mat_OMEGA[:,imax][~np.isnan(mat_OMEGA[:,imax])]) (ie_depth))



#----------------------------------------------------------
#  Calcul steady a0 vostok icecore
#----------------------------------------------------------

steady_a0 = interp1d(-theta[:][~np.isnan(mat_a0[:,imax])], \
mat_a0[:,imax][~np.isnan(mat_a0[:,imax])]   )(-theta_vic)



#----------------------------------------------------------
#  Calcul de R(t)
#----------------------------------------------------------

R_t = np.exp( beta * (deut - deut[0]) )
#FIXME: we should import R from AICC2012 and make it averaged to 1.


#----------------------------------------------------------
#  a0_vic
#----------------------------------------------------------

a0_vic = steady_a0 * R_t



#----------------------------------------------------------
#  Calcul steady_age vostok icecore (yr b 1997)
#----------------------------------------------------------

steady_age = interp1d(-theta[:][~np.isnan(mat_steady_age[:,imax+1])],\
    mat_steady_age[:,imax+1][~np.isnan(mat_steady_age[:,imax+1])] )(-theta_vic)


# Cubic spline without derivative constraint

steady_age_sp = interp1d(-theta[:][~np.isnan(mat_steady_age[:,imax+1])],\
    mat_steady_age[:,imax+1][~np.isnan(mat_steady_age[:,imax+1])], kind='cubic' ) (-theta_vic)


# Cubic spline with derivative constraint at surface
# On rajoute un point " proche de theta = 0 " afin d'imposer la dérivée
# Cela peu créer dans certains cas une matrice singulière (problème robustesse)


new_theta = np.insert(-theta[:][~np.isnan(mat_steady_age[:,imax+1])],1, 1/1000000 )
chi_0 = np.insert(mat_steady_age[:,imax+1][~np.isnan(mat_steady_age[:,imax+1])],1,\
0. + 1/(steady_a0[0]) * (ie_depth[1] - ie_depth[0]) / (theta_vic[0] -theta_vic[1]) * 1/1000000 )

steady_age_sp_2 = interp1d(new_theta, chi_0, kind = 'cubic')(-theta_vic)



#----------------------------------------------------------
#  Calcul Age vostok icecore (yr b 1997)
#----------------------------------------------------------

Age = np.cumsum((steady_age[1:] - steady_age[:-1]) / (R_t[:-1]))

Age = np.insert(Age,0,steady_age[0])



#----------------------------------------------------------
#  Calcul tau_middle vostok icecore (yr b 1997)
#----------------------------------------------------------

tau_middle = 1 / steady_a0[:-1] / ( steady_age[1:] - steady_age[:-1] )



#----------------------------------------------------------
#  Calcul tau_ie_middle vostok icecore (yr b 1997)
#----------------------------------------------------------

tau_ie_middle = (ie_depth[1:] - ie_depth[:-1]) / steady_a0[:-1] / (steady_age[1:] - steady_age[:-
1])



# Tau_ie avec l'interpolation "cubique-spline naturel" de l'âge stationnaire

tau_ie_middle_sp = (ie_depth[1:] - ie_depth[:-1]) / steady_a0[:-1] / \
(steady_age_sp[1:] - steady_age_sp[:-1])


# Tau_ie avec l'interpolation "cubique-spline- dérivée imposée" de l'âge stationnaire

tau_ie_middle_sp_2 = (ie_depth[1:] - ie_depth[:-1]) / steady_a0[:-1] / \
(steady_age_sp_2[1:] - steady_age_sp_2[:-1])



#----------------------------------------------------------
#  Calcul matrice depth: mat_depth
#----------------------------------------------------------

mat_depth = np.interp(mat_depth_ie, np.append(ie_depth, ie_depth[-1]+10000.), np.append(depth_corrected, depth_corrected[-1]+10000.))

# "x_new" out of range --> linear extrapolation 



#----------------------------------------------------------
#  Calcul matrice z : mat_z
#----------------------------------------------------------

mat_z = S - mat_depth



#----------------------------------------------------------
#  Calcul matrice Age : mat_Age
#----------------------------------------------------------

mat_Age = np.interp(mat_steady_age, np.append(steady_age, 100*steady_age[1]), np.append(Age, 100*Age[-1]))


#----------------------------------------------------------
#  Calcul des isochrones pour des ages données
#----------------------------------------------------------

Age_iso = np.arange( iso_spacing , np.amax(Age), iso_spacing) # Lignes isochrones avec un pas constant

steady_age_iso = interp1d(Age, steady_age)(Age_iso)

print('Before mat_theta_iso')
mat_theta_iso = np.zeros((len(Age_iso), imax+2))
for j in range(1,imax+2):
    mat_theta_iso[:,j] = np.interp(steady_age_iso[:], mat_steady_age[:,j], mat_theta[:,j-1])
mat_theta_iso[:,0] = mat_theta_iso[:,1]


print('Before mat_z_iso')
mat_z_iso = np.empty_like(mat_theta_iso)
for j in range(1,imax+2):
    mat_z_iso[:,j] = np.interp( - np.exp(mat_theta_iso[:,j]), - OMEGA, mat_z[:,j] )
mat_z_iso[:,0] = mat_z_iso[:,1]


mat_x_iso = np.tile(x,(len(Age_iso),1))


# Try to remove loops....

