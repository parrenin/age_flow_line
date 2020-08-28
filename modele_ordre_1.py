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



#----------------------------------------------------------
# Classes
#----------------------------------------------------------

def interp1d_extrap(x,y):
    def f(xp):
        g=interp1d(x,y, bounds_error=False)
        return np.where(xp<x[0],y[0],np.where(xp>x[-1],y[-1],g(xp)))    
    return f


def interp1d_linear_extrap(x,y, kind = 'linear'):
    def f(xp):
        g = interp1d (x , y , kind, bounds_error=False)
        slope = (y[-1] - y[-2]) / ( x[-1] - x[-2] )
        ordo_origine = y[-2] - slope * x[-2]
        xp2 = xp
        xp2[np.isnan(xp2)] = 0
        return np.where( xp2 > x[-1], slope * xp + ordo_origine, g(xp) )    
    return f



#---------------------------------------------------
# DEPTH - 1D Vostok drill grid for post-processing
#---------------------------------------------------

depth_corrected = np.arange(0., 3310. + 0.1 , 1.)

depth_mid = (depth_corrected[1:] + depth_corrected[:-1])/2

depth_inter = (depth_corrected[1:] - depth_corrected[:-1])



#---------------------------------------------------------------------------
# Relative density interpolation with extrapolation of "depth-density" data
#---------------------------------------------------------------------------

D_depth = density_readarray[:,0]

D_D = density_readarray[:,1]

relative_density = interp1d_extrap(D_depth, D_D) (depth_mid)

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

x_fld = np.arange(371)

x = interp1d ( Q_fld , x_fld)(Q)
Qm = interp1d ( Q_fld , Qm_fld)(Q)
Y = interp1d(x_fld,Y_fld)(x)
S = interp1d(x_fld, Su_fld)(x)
B = interp1d(x_fld, B_fld)(x)

B[0] = B[1] # Altitude du socle constante au niveau du dôme
S[0] = S[1] # Altitude de la surface constante au niveau du dôme



#--------------------------------------------------
# Accumulation a(m/yr)
#--------------------------------------------------

a = interp1d(Q_fld, a0_fld)(Q)



#--------------------------------------------------
# Calcul de la surface ice-equivalent S_ie
#--------------------------------------------------

S_ie = S - DELTA_H



#--------------------------------------------------
# Melting
#--------------------------------------------------

m = interp1d(x_fld, m_fld)(x)



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

s = interp1d(x_s_geo,s_measure)(x[1:])

zeta = np.linspace(1,0,1001).reshape(1001,1)

omega = zeta * s + (1-s) * ( 1 - (p+2)/(p+1) * (1-zeta) + 1/(p+1) * np.power(1-zeta,p+2)  ) 



#-------------------------------------------------------
# GRID
#-------------------------------------------------------

grid = np.ones( ( imax + 1 , imax + 2 ) )

grid[:,0] = grid[:,1] = np.where( theta >= theta_min[0], 1, 0 )


for j in range(2, imax+2 ):
  for i in range(1, imax+1):
    if ( theta[i] >= theta_min[j-1] and grid[i-1][j-1] == 1 ):
      grid[i][j] = 1
    else:
      grid[i][j] = 0



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

for j in range(0,imax+1):
  for i in range(0,imax+1):
    if grid[i][j+1] == 1 :
      inter = interp1d( - omega[:,j], zeta, axis = 0 )( - mat_omega[i][j])
      z_ie[i,j+1] = B[j+1] + inter * ( S_ie[j+1] - B[j+1] )
    else:
      z_ie[i,j+1] = np.nan

z_ie[:,0] = z_ie[:,1]



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


for j in range(1,imax+1):
  for i in range(1,imax+1):
    if grid[i][j+1] == 1 :
      mat_a0[i][j] = mat_a0[i-1,j-1]
    else:
      mat_a0[i,j] = np.nan


 
#-------------------------------------------------------
# Matrice x0 : mat_x0
#-------------------------------------------------------

mat_x0 = np.zeros((imax+1, imax+1+1))

mat_x0[:,0] = np.where( grid[:,0] == 1 , 0, np.nan )

mat_x0 [0,1:] = x[1:]

mat_x0[:,1] = np.where( grid[:,1] == 1 , mat_x0[0][1] * mat_OMEGA[:,0], np.nan )

for j in range(2,imax+1+1):
  for i in range(1,imax+1):
    if grid[i][j] == 1 :
      mat_x0[i][j] = mat_x0[i-1][j-1]
    else:
      mat_x0[i,j] = np.nan



#-------------------------------------------------------
# Matrice STEADY-AGE: 
#-------------------------------------------------------

#-----------------------------------------------------------------------------------------------
# Ordre 1
#-----------------------------------------------------------------------------------------------

mat_steady_age = np.zeros((imax+1, imax+2))

for i in range(1,imax+1):
  if grid[i][1] == 1 :
    mat_steady_age[i][1] = mat_steady_age[i-1][1] + delta / a[1] \
    * (z_ie[i-1][1] - z_ie[i][1]) / ( OMEGA[i-1] - OMEGA[i] )
  else:
    mat_steady_age[i][1] = np.nan

mat_steady_age[:,0] = mat_steady_age[:,1]


for j in range(2,imax+2):
  for i in range(1,imax+1):
    if grid[i][j] == 1 :

      c = (a[j] - a[j-1]) / delta 

      d = a[j] - c * pi[j-1]

      e = ((z_ie[i,j] - z_ie[i-1,j]) / (OMEGA[i] - OMEGA[i-1]) - \
      (z_ie[i-1,j-1] - z_ie[i,j-1]) / (OMEGA[i-1] - OMEGA[i]) ) / delta	

      f = (z_ie[i,j] - z_ie[i-1,j]) / ( OMEGA[i] - OMEGA[i-1] ) - e * pi[j-1]

      if c == 0:
        mat_steady_age[i,j] = mat_steady_age[i-1][j-1] + \
        (1 / d) * ( e * (pi[j-1]*pi[j-1] - pi[j-2]*pi[j-2]) / 2 + \
        f * delta )

      else:
        mat_steady_age[i][j] = mat_steady_age[i-1][j-1] + \
        (e * pi[j-1] + f) * log (abs ( c * pi[j-1] + d ) ) / c - \
        (e * pi[j-2] + f) * log ( abs ( c * pi[j-2] + d ) ) / c - \
        ( e / c ) * ( ( pi[j-1] + d/c ) * log (abs(c * pi[j-1] + d ) )-\
        ( pi[j-2] + d/c ) * log (abs (c * pi[j-2] + d )) - delta  ) 

    else:
      mat_steady_age[i][j] = np.nan


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

mat_depth = np.where( grid == 1, interp1d_linear_extrap(ie_depth,depth_corrected)(mat_depth_ie), np.nan )

# "x_new" out of range --> linear extrapolation 



#----------------------------------------------------------
#  Calcul matrice z : mat_z
#----------------------------------------------------------

mat_z = S - mat_depth



#----------------------------------------------------------
#  Calcul matrice Age : mat_Age
#----------------------------------------------------------

mat_Age = np.where( grid == 1, interp1d_linear_extrap(steady_age, Age)(mat_steady_age), np.nan  )



#----------------------------------------------------------
#  Calcul des isochrones pour des ages données
#----------------------------------------------------------

Age_iso = np.arange( 20000 , np.amax(Age), 20000) # Lignes isochrones avec un pas de 20000 ans

steady_age_iso = interp1d(Age, steady_age)(Age_iso)

mat_theta_iso = np.zeros((len(Age_iso), imax+2))

for i in range(len(Age_iso)):
    for j in range(1,imax+2):
        mat_theta_iso[i,j] = interp1d(mat_steady_age[:,j], mat_theta[:,j-1])(steady_age_iso[i])
mat_theta_iso[:,0] = mat_theta_iso[:,1]


mat_z_iso = np.empty_like(mat_theta_iso)

for i in range(len(Age_iso)):
    for j in range(1,imax+2):
        mat_z_iso[i,j] = interp1d(- OMEGA, mat_z[:,j])( - exp(mat_theta_iso[i,j])  )
mat_z_iso[:,0] = mat_z_iso[:,1]


mat_x_iso = np.tile(x,(len(Age_iso),1))


# Try to remove loops....

