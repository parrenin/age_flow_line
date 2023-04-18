# TODO: make sure the interpolated values for a, Y and Q are consistent.
# FIXME: Convert all comments to English.

import sys
import numpy as np
from scipy.interpolate import interp1d
from math import log
import yaml
import matplotlib.pyplot as plt
import time

# Registration of start time
START_TIME = time.perf_counter()

# Setting experiment directory
datadir = sys.argv[1]
if datadir[-1] != '/':
    datadir = datadir+'/'
print('Parameters directory is: ', datadir)

# -----------------------------------------------------------------------------
# Loading data from accu-prior.txt , density-prior.txt ... in "input_data" file
# -----------------------------------------------------------------------------

deut = np.loadtxt(datadir+'deuterium.txt')
density_readarray = np.loadtxt(datadir+'density-prior.txt')

# x_s_geo = np.loadtxt('input_data/s_geodata.txt', usecols=(0,))
# s_measure = np.loadtxt('input_data/s_geodata.txt', usecols=(1,))

# ---------------------------------------------------------
# Reading parameters.yml file (imax, delta,...)
# ---------------------------------------------------------

# Default values for parameters, to prevent spyder errors
max_depth = 3310.
imax = 100
delta = 0.08
x_right = 370
iso_spacing = 20000.
iso_max = 1000000.
beta = 0.015
thickness = 3767.
create_figs = True
fig_age_max = 1000000
fig_age_spacing = 10000
fig_age_spacing_labels = 100000

yamls = open(datadir+'parameters.yml').read()
para = yaml.load(yamls, Loader=yaml.FullLoader)
globals().update(para)

# -----------------------------------------------------
# Loading files for Geographic data, arrays creations
# -----------------------------------------------------

# Steady accumulation
x_a0, a0_measure = np.loadtxt(datadir+'accumulation.txt', unpack=True)

# Melting
x_m, m_measure = np.loadtxt(datadir+'melting.txt', unpack=True)

# Sliding rate
x_s, s_measure = np.loadtxt(datadir+'sliding.txt', unpack=True)

# Lliboutry parameter
x_p, p_measure = np.loadtxt(datadir+'p_Lliboutry.txt', unpack=True)


# Surface and Bedrock
x_B, B_measure = np.loadtxt(datadir+'bedrock.txt', unpack=True)
x_Su, Su_measure = np.loadtxt(datadir+'surface.txt', unpack=True)

# Tube width
x_Y, Y_measure = np.loadtxt(datadir+'tube_width.txt', unpack=True)

# --------------------
# Interpolation
# --------------------

# FIXME: Use a parameter for the horizontal step
# FIXME: Or even better, don't use any parameter and use natural samping
x_fld = np.arange(x_right+1)

a0_fld = np.interp(x_fld, x_a0, a0_measure)
m_fld = np.interp(x_fld, x_m, m_measure)
s_fld = np.interp(x_fld, x_s, s_measure)
p_fld = np.interp(x_fld, x_p, p_measure)
Su_fld = np.interp(x_fld, x_Su, Su_measure)
B_fld = np.interp(x_fld, x_B, B_measure)
Y_fld = np.interp(x_fld, x_Y, Y_measure)

# Computation of total flux Q

Q_fld = np.zeros(len(x_fld))

for i in range(1, len(Q_fld)):
    Q_fld[i] = Q_fld[i-1] + (x_fld[i]-x_fld[i-1]) * 1000 * a0_fld[i-1] * \
        Y_fld[i-1] + 0.5 * (x_fld[i]-x_fld[i-1]) * 1000 * \
        ((a0_fld[i]-a0_fld[i-1]) * Y_fld[i-1] + (Y_fld[i]-Y_fld[i-1])
            * a0_fld[i-1]) + (1./3) * (x_fld[i]-x_fld[i-1]) * 1000 * \
        (a0_fld[i]-a0_fld[i-1]) * (Y_fld[i]-Y_fld[i-1])

# Computation of basal melting flux Qm

Qm_fld = [0]*len(x_fld)

for i in range(1, len(Qm_fld)):
    Qm_fld[i] = Qm_fld[i-1] + 0.5 * (m_fld[i]+m_fld[i-1]) * 0.5 *\
        (Y_fld[i]+Y_fld[i-1]) * (x_fld[i]-x_fld[i-1]) * 1000

# ---------------------------------------------------
# DEPTH - 1D Vostok drill grid for post-processing
# ---------------------------------------------------

# FIXME: Is this fine vertical grid really necessary?

depth_corrected = np.arange(0., max_depth + 0.1, 1.)

depth_mid = (depth_corrected[1:] + depth_corrected[:-1])/2

depth_inter = (depth_corrected[1:] - depth_corrected[:-1])

# ---------------------------------------------------------------------------
# Relative density interpolation with extrapolation of "depth-density" data
# ---------------------------------------------------------------------------

# FIXME: What should we do with firn density? Maybe just a firn correction?

D_depth = density_readarray[:, 0]

D_D = density_readarray[:, 1]

relative_density = np.interp(depth_mid, D_depth, D_D, right=1.)

ie_depth = np.cumsum(np.concatenate((np.array([0]),
                                     relative_density * depth_inter)))

# ----------------------------------------------------------
# DELTA H
# ----------------------------------------------------------

DELTA_H = depth_corrected[-1] - ie_depth[-1]

# ----------------------------------------------------------
# Mesh generation (pi,theta)
# ----------------------------------------------------------

pi = np.linspace(-imax * delta, 0,  imax + 1)

theta = np.linspace(0, - imax * delta,  imax + 1)

# ----------------------------------------------------------
# Total flux Q(m^3/yr)
# ----------------------------------------------------------

Q = Q_fld[-1] * np.exp(pi)  # Q_ref = Q_fld[-1]

# We set the zero value for the dome
Q = np.insert(Q, 0, 0)

# ----------------------------------------------------------
# OMEGA
# ----------------------------------------------------------

OMEGA = np.exp(theta)

# ----------------------------------------------------------
# interpolation of flow line data files for x, Qm, ...
# ----------------------------------------------------------

x_fld = np.arange(x_right+1)

# FIXME : should we interpolate in x or in Q?
x = np.interp(Q, Q_fld, x_fld)
Qm = np.interp(Q, Q_fld, Qm_fld)
Y = np.interp(x, x_fld, Y_fld)
S = np.interp(x, x_fld, Su_fld)
B = np.interp(x, x_fld, B_fld)
s = np.interp(x[1:], x_fld, s_fld)
p = np.interp(x[1:], x_fld, p_fld)

B[0] = B[1]  # Altitude du socle constante au niveau du dôme
S[0] = S[1]  # Altitude de la surface constante au niveau du dôme

# --------------------------------------------------
# Accumulation a(m/yr)
# --------------------------------------------------

# FIXME: again here, should we interpolate in x or Q?
a = np.interp(Q, Q_fld, a0_fld)

# --------------------------------------------------
# Calcul de la surface ice-equivalent S_ie
# --------------------------------------------------

S_ie = S - DELTA_H

# --------------------------------------------------
# Melting
# --------------------------------------------------

# FIXME: why do we interpolate m while we already interpolated Qm?
m = np.interp(x, x_fld, m_fld)

# ------------------------------------------------------
# Computation of theta_min and theta_max
# ------------------------------------------------------

theta_max = np.zeros(imax+1)

theta_min = np.where(Qm[1:] > 0,
                     np.maximum(np.log(Qm[1:].clip(min=10**-100)/Q[1:]),
                                theta[-1] * np.ones((imax+1,))),
                     theta[-1] * np.ones((imax+1,)))

# ------------------------------------------------------
# Computation of omega=fct(zeta)
# ------------------------------------------------------

# Lliboutry model for the horizontal flux shape function

# FIXME: Is there no better we do invert omega?
# Maybe a spline would allow to decrease the number of nodes.
zeta = np.linspace(1, 0, 1001).reshape(1001, 1)

omega = zeta * s + (1-s) * (1 - (p+2)/(p+1) * (1-zeta) +
                            1/(p+1) * np.power(1-zeta, p+2))

# -------------------------------------------------------
# GRID
# -------------------------------------------------------

grid = np.ones((imax + 1, imax + 2))

grid[:, 0] = grid[:, 1] = np.where(theta >= theta_min[0], 1, 0)


print('Before defining grid boolean')
for j in range(2, imax+2):
    grid[2:, j] = np.where(np.logical_and(theta[2:] >= theta_min[j-1],
                                          grid[1:-1, j-1] == 1), 1, 0)
print('After defining grid boolean')

# -------------------------------------------------------
# Matrice omega : mat_omega
# -------------------------------------------------------

mat_omega = np.where(grid[:, 1:] == 1,
                     (np.dot(OMEGA.reshape(imax+1, 1),
                             Q[1:].reshape(1, imax+1))-Qm[1:])/(Q[1:]-Qm[1:]),
                     np.nan)

# -------------------------------------------------------
# Matrix z_ie
# -------------------------------------------------------

z_ie = np.zeros((imax+1, imax+2))

print('Before defining z_ie')
for j in range(0, imax+1):
    inter = np.interp(-mat_omega[:, j], -omega[:, j].flatten(), zeta.flatten())
    z_ie[:, j+1] = np.where(grid[:, j+1] == 1, B[j+1]+inter*(S_ie[j+1]-B[j+1]),
                            np.nan)
z_ie[:, 0] = z_ie[:, 1]
print('After defining z_ie')

# -------------------------------------------------------
# Matrix theta_min
# -------------------------------------------------------

mat_theta_min = np.tile(theta_min, (imax+1, 1))

# -------------------------------------------------------
# Matrix theta
# -------------------------------------------------------

mat_theta = np.where(grid[:, 1:] == 1, theta.reshape(imax+1, 1), np.nan)

# -------------------------------------------------------
# Matrix OMEGA: mat_OMEGA
# -------------------------------------------------------

mat_OMEGA = np.where(grid[:, 1:] == 1, OMEGA.reshape(imax+1, 1), np.nan)

# -------------------------------------------------------
# Matrix pi: mat_pi
# -------------------------------------------------------

mat_pi = np.where(grid[:, 1:] == 1, pi, np.nan)

# -------------------------------------------------------
# Matrix x: mat_x
# -------------------------------------------------------

mat_x = np.where(grid == 1, x, np.nan)

# -------------------------------------------------------
# Matrix depth_ie: mat_depth_ie
# -------------------------------------------------------

mat_depth_ie = np.where(grid == 1, S_ie - z_ie, np.nan)

mat_depth_ie[0, :] = 0

# -------------------------------------------------------
# Matrix of stream function q: mat_q
# -------------------------------------------------------

mat_q = np.where(grid[:, 1:] == 1, Q[1:] * mat_OMEGA, np.nan)

mat_q[:, 0] = mat_q[0, 0]  # ligne de flux verticale au dôme


# -------------------------------------------------------
# Matrix a0: mat_a0
# -------------------------------------------------------

mat_a0 = np.zeros((imax+1, imax+1))

mat_a0[0, :] = a[1:]

mat_a0[1:, 0] = np.where(grid[1:, 1] == 1, mat_a0[0, 0], np.nan)

print('Before defining mat_a0')
for j in range(1, imax+1):
    mat_a0[1:, j] = np.where(grid[1:, j] == 1, mat_a0[:-1, j-1], np.nan)
print('After defining mat_a0')

# -------------------------------------------------------
# Matrix x0: mat_x0
# -------------------------------------------------------

mat_x0 = np.zeros((imax+1, imax+1+1))

mat_x0[:, 0] = np.where(grid[:, 0] == 1, 0, np.nan)

mat_x0[0, 1:] = x[1:]

mat_x0[:, 1] = np.where(grid[:, 1] == 1, mat_x0[0][1]*mat_OMEGA[:, 0], np.nan)

print('Before defining mat_x0')
for j in range(2, imax+1+1):
    mat_x0[1:, j] = np.where(grid[1:, j] == 1, mat_x0[:-1, j-1], np.nan)
print('After defining mat_x0')

# -------------------------------------------------------
# Matrix STEADY-AGE:
# -------------------------------------------------------

# -----------------------------------------------------------------------------
# First oder computation of age
# -----------------------------------------------------------------------------

print('Before calculation of steady age matrix.')

mat_steady_age = np.zeros((imax+1, imax+2))

for i in range(1, imax+1):
    if grid[i][1] == 1:
        mat_steady_age[i][1] = mat_steady_age[i-1][1] + delta / a[1] \
            * (z_ie[i-1][1] - z_ie[i][1]) / (OMEGA[i-1] - OMEGA[i])
    else:
        mat_steady_age[i][1] = np.nan

mat_steady_age[:, 0] = mat_steady_age[:, 1]

# FIXME: use polynomes instead of rational fractions
for j in range(2, imax+2):
    c = (a[j] - a[j-1]) / delta
    d = a[j] - c * pi[j-1]
    e = ((z_ie[1:, j] - z_ie[:-1, j]) / (OMEGA[1:] - OMEGA[:-1]) -
         (z_ie[:-1, j-1] - z_ie[1:, j-1]) / (OMEGA[:-1] - OMEGA[1:])) / delta
    f = (z_ie[1:, j] - z_ie[:-1, j]) / (OMEGA[1:] - OMEGA[:-1]) - e * pi[j-1]
    if c == 0:
        mat_steady_age[1:, j] = np.where(grid[1:, j] == 1,
                                         mat_steady_age[:-1, j-1] + (1 / d) *
                                         (e * (pi[j-1]*pi[j-1] -
                                          pi[j-2]*pi[j-2])
                                          / 2 + f * delta), np.nan)
    else:
        mat_steady_age[1:, j] = np.where(grid[1:, j] == 1,
                                         mat_steady_age[:-1,j-1] + (e*pi[j-1]
                                          + f)*log(abs(c*pi[j-1]+d))/c -
                                         (e * pi[j-2] + f) * log(abs(c*pi[j-2]
                                         + d)) / c -(e/c)*((pi[j-1] + d/c) *
                                         log(abs(c * pi[j-1] + d))-(pi[j-2] +
                                         d/c)*log(abs(c*pi[j-2]+d))-delta),
                                         np.nan)

print('After calculation of steady age matrix.')

# Note: NaN dans coeff e dû à l'utilisation d'un schéma aux différences finies
# avant z_ie[ imax-1, imax] = NaN, pas de répercutions néfastes.

# Try to remove loops

# -------------------------------------------------------
# Matrix of thinning function: tau_ie
# -------------------------------------------------------

tau_ie = np.where(grid[1:, 1:] == 1, (z_ie[:-1, 1:] - z_ie[1:, 1:])
                  / (mat_steady_age[1:, 1:] - mat_steady_age[:-1, 1:])
                  / mat_a0[:-1, :], np.nan)

# ----------------------------------------------------------
# Post-processing: transfert of the modeling results
# on the 1D grid of the drilling site
# ----------------------------------------------------------

# ----------------------------------------------------------
#  Computation fo theta for the ice core: theta_vic
# ----------------------------------------------------------

# FIXME: Rename to theta_ic (V was for Vostok)

if mat_depth_ie[imax, imax+1] < ie_depth[len(ie_depth)-1]:
    sys.exit("\n Attention problème d'interpolation post-processing:\n \
        le modèle donne des résultats jusqu'à une profondeur maximale trop \n \
        faible par rapport à la profondeur maximale du forage Vostok \n \
        Pour transposer correctement les résultats \n \
        du maillage sur toute la hauteur du forage Vostok il faut augmenter\n \
        le nombre de noeuds du maillage.")

theta_vic = np.log(np.interp(ie_depth, mat_depth_ie[:, imax+1]
                             [~np.isnan(mat_depth_ie[:, imax+1])],
                             mat_OMEGA[:, imax]
                             [~np.isnan(mat_OMEGA[:, imax])]))

# ----------------------------------------------------------
#  Computation steady a0 ice core
# ----------------------------------------------------------

steady_a0 = np.interp(-theta_vic, -theta[:][~np.isnan(mat_a0[:, imax])],
                      mat_a0[:, imax][~np.isnan(mat_a0[:, imax])])

# ----------------------------------------------------------
#  Computation of R(t)
# ----------------------------------------------------------

R_t = np.exp(beta * (deut - deut[0]))

# FIXME: we should import R from AICC2012 and make it averaged to 1.

# ----------------------------------------------------------
#  a0_vic
# ----------------------------------------------------------

a0_vic = steady_a0 * R_t

# ----------------------------------------------------------
#  Computation of steady_age vostok icecore
# ----------------------------------------------------------

# FIXME: we should set the surface age in the YAML file.

steady_age = np.interp(-theta_vic, -theta[:][~np.isnan(mat_steady_age[
    :, imax+1])], mat_steady_age[:, imax+1][~np.isnan(mat_steady_age[
        :, imax+1])])


# Cubic spline without derivative constraint

steady_age_sp = interp1d(-theta[:][~np.isnan(mat_steady_age[:, imax+1])],
                         mat_steady_age[:, imax+1][~np.isnan(mat_steady_age[
                             :, imax+1])], kind='cubic')(-theta_vic)

# Cubic spline with derivative constraint at surface
# On rajoute un point " proche de theta = 0 " afin d'imposer la dérivée
# Cela peu créer dans certains cas une matrice singulière (problème robustesse)
# FIXME: Is there not a less dirty solution?

new_theta = np.insert(-theta[:][~np.isnan(mat_steady_age[:, imax+1])], 1,
                      1/1000000)
chi_0 = np.insert(mat_steady_age[:, imax+1][~np.isnan(mat_steady_age[
    :, imax+1])], 1, 0.+1/(steady_a0[0])*(ie_depth[1] - ie_depth[0]) /
    (theta_vic[0] - theta_vic[1]) * 1/1000000)

steady_age_sp_2 = interp1d(new_theta, chi_0, kind='cubic')(-theta_vic)

# ----------------------------------------------------------
#  Computation of Age for the ice core
# ----------------------------------------------------------

Age = np.cumsum((steady_age[1:] - steady_age[:-1]) / (R_t[:-1]))

Age = np.insert(Age, 0, steady_age[0])

# ----------------------------------------------------------
#  Computation of tau_middle for the ice core
# ----------------------------------------------------------

tau_middle = 1./steady_a0[:-1] / (steady_age[1:] - steady_age[:-1])

# ----------------------------------------------------------
#  Computation of tau_ie_middle for the ice core
# ----------------------------------------------------------

tau_ie_middle = (ie_depth[1:] - ie_depth[:-1]) / steady_a0[:-1] / \
                (steady_age[1:] - steady_age[:-1])

# Tau_ie avec l'interpolation "cubique-spline naturel" de l'âge stationnaire

tau_ie_middle_sp = (ie_depth[1:] - ie_depth[:-1]) / steady_a0[:-1] / \
                   (steady_age_sp[1:] - steady_age_sp[:-1])

# Tau_ie avec l'interpolation "cubique-spline- dérivée imposée"
# de l'âge stationnaire

tau_ie_middle_sp_2 = (ie_depth[1:] - ie_depth[:-1]) / steady_a0[:-1] / \
                     (steady_age_sp_2[1:] - steady_age_sp_2[:-1])

# ----------------------------------------------------------
#  Calcul matrice depth: mat_depth
# ----------------------------------------------------------

mat_depth = np.interp(mat_depth_ie, np.append(ie_depth, ie_depth[-1]+10000.),
                      np.append(depth_corrected, depth_corrected[-1]+10000.))

# "x_new" out of range --> linear extrapolation

# ----------------------------------------------------------
#  Calcul matrice z : mat_z
# ----------------------------------------------------------

mat_z = S - mat_depth

# ----------------------------------------------------------
#  Calcul matrice Age : mat_Age
# ----------------------------------------------------------

mat_Age = np.interp(mat_steady_age, np.append(steady_age, 100*steady_age[-1]),
                    np.append(Age, 100*Age[-1]))

# ----------------------------------------------------------
#  Calcul des isochrones pour des ages données
# ----------------------------------------------------------

# Lignes isochrones avec un pas constant
Age_iso = np.arange(iso_spacing, iso_max+1., iso_spacing)

steady_age_iso = np.interp(Age_iso, np.append(Age, 100*Age[-1]),
                           np.append(steady_age, 100*steady_age[-1]))

print('Before mat_theta_iso')
mat_theta_iso = np.zeros((len(Age_iso), imax+2))
for j in range(1, imax+2):
    mat_theta_iso[:, j] = np.interp(steady_age_iso[:], mat_steady_age[:, j],
                                    mat_theta[:, j-1])
mat_theta_iso[:, 0] = mat_theta_iso[:, 1]

print('Before mat_z_iso')
mat_z_iso = np.empty_like(mat_theta_iso)
for j in range(1, imax+2):
    mat_z_iso[:, j] = np.interp(-np.exp(mat_theta_iso[:, j]), -OMEGA,
                                mat_z[:, j])
mat_z_iso[:, 0] = mat_z_iso[:, 1]

mat_x_iso = np.tile(x, (len(Age_iso), 1))


if create_figs:

    # ----------------------------------------
    # Figure 1: Tube width Y
    # ----------------------------------------

    plt.figure(1)
    plt.plot(pi, Y[1:], '-')
    plt.legend([r'Tube width: Y (m)'], loc='best')
    plt.xlabel(r'$\pi$', fontsize=18)
    plt.ylabel(r'$Y \ (m)$', fontsize=18)
    plt.grid()
    plt.savefig(datadir+'tube_width_theta.pdf')

    # ---------------------------------------------------
    # Figure 2: Accumulation
    # ---------------------------------------------------

    plt.figure(2)
    plt.plot(pi, a[1:], '-')
    plt.legend([r'accumulation: a (m/year)'], loc='best')
    plt.xlabel(r'$\pi$', fontsize=18)
    plt.ylabel(r'$a \ (m/year)$', fontsize=18)
    plt.savefig(datadir+'accumulation_theta.pdf')
    plt.grid()

    # ----------------------------------------
    # Figure 3: Melting rate
    # ----------------------------------------

    plt.figure(3)
    plt.plot(pi, m[1:], '-')
    plt.legend([r'melting: m (m/year)'], loc='best')
    plt.xlabel(r'$\pi$', fontsize=18)
    plt.ylabel(r'$m \ (m/year)$', fontsize=18)
    plt.savefig(datadir+'melting_theta.pdf')
    plt.grid()

    # ----------------------------------------------------------
    # Visualisation maillage (pi,theta)
    # ----------------------------------------------------------

    plt.figure(4)
    plt.vlines(pi, theta_min, theta_max)
    for i in range(0, imax+1):
        plt.hlines(mat_theta[i, :], mat_pi[0, :], np.zeros((imax+1,)))
    plt.xlabel(r'$\pi$', fontsize=18)
    plt.ylabel(r'$\theta$', fontsize=18)
    plt.savefig(datadir+'mesh_pi_theta.pdf')

    # ----------------------------------------------------------
    # Visualisation du maillage (x, z)
    # ----------------------------------------------------------

    z_ie_min = np.zeros((imax+2))
    for i in range(imax+2):
        z_ie_min[i] = np.asarray(np.amin(z_ie[:, i][~np.isnan(z_ie[:, i])]))

    plt.figure(5, figsize=(15, 5))
    for i in range(0, imax+1):
        plt.plot(x[:], z_ie[i, :],  ls='-', marker='.', color='black')
    plt.vlines(x, z_ie_min, S_ie)
    plt.xlabel(r'$X$', fontsize=18)
    plt.ylabel(r'$Z$', fontsize=18)
    plt.savefig(datadir+'mesh_x_z.pdf')

    # -------------------------------------------------------------------------
    # Visualisation des paramètres de l'écoulement sur le maillage (pi,theta)
    # -------------------------------------------------------------------------

# FIXME: Is this not a bit too complicated?

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)

    # Twin the x-axis twice to make independent y-axes.
    axes = [ax.twinx(), ax.twinx()]

    # Make some space on the right side for the extra y-axis.
    fig.subplots_adjust(right=0.8)

    # Move the last y-axis spine over to the right by 20% of the width
    axes[0].spines['right'].set_position(('axes', 1.01))
    axes[0].spines['right'].set_color('Green')
    axes[1].spines['right'].set_position(('axes', 1.09))
    axes[1].spines['right'].set_color('Red')

    # To make the border of the right-most axis visible, we need to turn the
    # frame on.
    # This hides the other plots, however, so we need to turn its fill off.
    axes[0].set_frame_on(True)
    axes[0].patch.set_visible(False)
    axes[1].set_frame_on(True)
    axes[1].patch.set_visible(False)

    # And finally we get to plot ...
    colors = ('Green', 'Red')
    datas = [a[1:], m[1:]]
    ynames = ['a (m/yr)', 'm (m/yr)']
    ax.set_xlabel(r'$\pi$', fontsize=18)
    ax.set_ylabel('Y (relative unit)')
    ax.plot(pi, Y[1:], color='k')

    for ax, color, data, yname in zip(axes, colors, datas, ynames):
        ax.plot(pi, data, color=color)
        ax.set_ylabel(yname, color=color)
        ax.tick_params(axis='y', colors=color)
    plt.savefig(datadir+'flow_parameters_pi_theta.pdf')

    # -------------------------------------------------------------------------
    # Visualisation des paramètres de l'écoulement sur le maillage (x,z)
    # -------------------------------------------------------------------------

# FIXME: idem here, this is a bit complicated.

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    fig.subplots_adjust(right=0.8)

    axes = [ax.twinx(), ax.twinx()]

    # Move the last y-axis spine over to the right by 20% of the width
    axes[0].spines['right'].set_position(('axes', 1.01))
    axes[0].spines['right'].set_color('Green')
    axes[1].spines['right'].set_position(('axes', 1.09))
    axes[1].spines['right'].set_color('Red')

    # To make the border of the right-most axis visible, we need to turn the
    # frame on.
    # This hides the other plots, however, so we need to turn its fill off.
    axes[0].set_frame_on(True)
    axes[0].patch.set_visible(False)
    axes[1].set_frame_on(True)
    axes[1].patch.set_visible(False)

    colors = ('Green', 'Red')
    datas = [a, m]
    ax.set_xlabel('x (km)', fontsize=18)
    ax.set_ylabel('Y (relative unit)')
    ax.plot(x, Y, color='k')

    for ax, color, data, yname in zip(axes, colors, datas, ynames):
        ax.plot(x, data, color=color)
        ax.set_ylabel(yname, color=color)
        ax.tick_params(axis='y', colors=color)
    plt.savefig(datadir+'flow_parameters_x_z.pdf')

    # ----------------------------------------------------------
    # Display of age and isochrones in (x,z)
    # ----------------------------------------------------------

    fig = plt.figure(8, figsize=(12, 6))
    ax = fig.add_subplot(111)

    ax.plot(x, S, label='Surface', color='0')
    ax.plot(x, B, label='Bedrock', color='0')

    for i in range(len(Age_iso)):
        ax.plot(mat_x_iso[i, :], mat_z_iso[i, :], label=str(Age_iso[i])+' yr',
                color='w')
    levels = np.arange(0, fig_age_max, fig_age_spacing)
    levels_cb = np.arange(0, fig_age_max, fig_age_spacing_labels)
    cp = plt.contourf(mat_x, mat_z, mat_Age/1000., levels=levels,
                      cmap='jet')
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.set_label('Modeled steady age (kyr)')
    ax.set_xlabel(r'$x$', fontsize=19)
    ax.set_ylabel(r'$z$', fontsize=19)
    box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.grid()
#    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(datadir+'age_x_z.pdf')

    # ---------------------------------------------------------------------
    # Lignes isochrones en (pi,theta)
    # ---------------------------------------------------------------------

    fig = plt.figure(9, figsize=(12, 6))
    ax = fig.add_subplot(111)
    for i in range(len(Age_iso)):
        ax.plot(pi, mat_theta_iso[i, 1:], marker='.',
                label=str(Age_iso[i])+' year', color=plt.cm.cool(30*i))
    ax.set_xlabel(r'$\pi$', fontsize=19)
    ax.set_ylabel(r'$\theta$', fontsize=19)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.grid()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(datadir+'isochrones_pi_theta.pdf')

    # ---------------------------------------------------------------------
    # Age-Profondeur dans le forage
    # ---------------------------------------------------------------------

# FIXME: we could have several drillings along the flow lines

    plt.figure(10)
    plt.plot(Age, depth_corrected, '-')
    plt.gca().invert_yaxis()
    plt.xlabel(r'$age\ (yr\ b \ 1997)$', fontsize=15)
    plt.ylabel(r'$depth \ (m)$', fontsize=15)
    plt.savefig(datadir+'age_depth.pdf')

    # ---------------------------------------------------------------------
    # R(t) - Age
    # ---------------------------------------------------------------------

    plt.figure(11)
    plt.plot(Age, R_t, '-')
    plt.xlabel(r'$time \ (yr\ b\ 1997 )$', fontsize=15)
    plt.ylabel(r'$R(t)$', fontsize=15)
    plt.savefig(datadir+'R_temporal_factor.pdf')

    # ---------------------------------------------------------------------
    # Fonction d'amincissement
    # ---------------------------------------------------------------------

    plt.figure(12, figsize=(12, 8))

    plt.plot(tau_ie_middle, depth_corrected[:-1], ls='-',
             label=r'$ordre\ 1 - lin\'eaire$')
    plt.plot(tau_ie_middle_sp, depth_corrected[:-1], ls='-',
             label=r'$ordre\ 1 - spline\ cubique$')
    plt.plot(tau_ie_middle_sp_2, depth_corrected[:-1], ls='-',
             label=r'$ordre\ 1 - spline\ cubique - d\'eriv\'ee\ impos\'ee$')
    plt.ylim([max(depth_corrected), -200.])
    plt.xlabel(r'$\tau_{ie} \ (yr \ b1997) $', fontsize=18)
    plt.ylabel(r'$depth \ (m)$', fontsize=18)
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(datadir+'thinning_profile.pdf')

    # ----------------------------------------------------------
    # Visualisation des lignes de courant
    # ----------------------------------------------------------

# FIXME: For more than 100 lines, it becomes impossible to see.

    plt.figure(13, figsize=(15, 7))
    plt.plot(x, S, label='Surface', color='0')
    for i in range(2, imax+1):  # add a step to plot less trajecories
        plt.plot(x[i:,], np.diagonal(mat_z, i), color='blue')
        plt.plot(x[2:-i+1,], np.diagonal(mat_z, -i+2)[2:], color='blue')
    plt.plot(x[2:,], np.diagonal(mat_z, 1)[1:], color='blue')
    plt.vlines(x[0], B[0], S[0], color='blue')  # ice divide
    plt.vlines(x[1], B[1], S[1], color='blue')  # vertical flow
    plt.plot(0, 0, label="Trajectoires", color='blue')
    plt.plot(x, B, label='Socle', color='0')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim([-5, x[imax+1] + 5])
    plt.xlabel(r'$X$', fontsize=19)
    plt.ylabel(r'$Z$', fontsize=19)
    plt.grid()
    plt.savefig(datadir+'stream_lines.pdf')

    # ----------------------------------------------------------
    # Ice Origin
    # ----------------------------------------------------------

    plt.figure(14)
    plt.plot(mat_x0[:, imax+1], mat_depth[:, imax+1], ls='-', marker='.')
    plt.gca().invert_yaxis()
    plt.xlabel(r'$ICE\ ORIGIN \ (km)$', fontsize=15)
    plt.ylabel(r'$DEPTH \ (m)$', fontsize=15)
    plt.savefig(datadir+'ice_origin.pdf')

    plt.show()

# Program execution time
MESSAGE = 'Program execution time: '+str(time.perf_counter()-START_TIME)+' s.'
print(MESSAGE)
