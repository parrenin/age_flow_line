# TODO: make sure the interpolated values for a, Y and Q are consistent.
# FIXME: there is no output in the program. Maybe output at least for the core.

import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
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

# deut = np.loadtxt(datadir+'deuterium.txt')
age_R, R = np.loadtxt(datadir+'temporal_factor.txt', unpack=True)
density_readarray = np.loadtxt(datadir+'density-prior.txt')

# x_s_geo = np.loadtxt('input_data/s_geodata.txt', usecols=(0,))
# s_measure = np.loadtxt('input_data/s_geodata.txt', usecols=(1,))

# ---------------------------------------------------------
# Reading parameters.yml file (imax, delta, ...)
# ---------------------------------------------------------

# Default values for parameters, to prevent spyder errors
max_depth = 3310.
step_depth = 1.
imax = 100
delta = 0.08
x_right = 370
traj_step = 10
fig_age_max = 1000000
fig_age_spacing = 10000
fig_age_spacing_labels = 100000
beta = 0.015
thickness = 3767.
create_figs = True
fig_format = 'pdf'

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

depth_corrected = np.arange(0., max_depth + 0.0001, step_depth)

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
s = np.interp(x, x_fld, s_fld)
p = np.interp(x, x_fld, p_fld)

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

# We just use theta_max for the mesh plot
theta_max = np.zeros(imax+1)

theta_min = np.where(Qm[1:] > 0,
                     np.maximum(np.log(Qm[1:].clip(min=10**-100)/Q[1:]),
                                theta[-1] * np.ones((imax+1,))),
                     theta[-1] * np.ones((imax+1,)))


# -------------------------------------------------------
# GRID
# -------------------------------------------------------

grid = np.ones((imax + 1, imax + 2), dtype=bool)

grid[:, 0] = grid[:, 1] = theta >= theta_min[0]

print('Before defining grid boolean')
# We need an iteration here, to treat a line after the previous one
# imax+2 here so that we stop at imax+1
for j in range(2, imax+2):
    grid[2:, j] = np.logical_and(theta[2:] >= theta_min[j-1], grid[1:-1, j-1])
print('After defining grid boolean')

# -------------------------------------------------------
# Matrix theta
# -------------------------------------------------------

mat_theta = theta.reshape(imax+1, 1)*np.ones((1, imax+1))
mat_theta = np.where(grid[:, 1:], mat_theta, np.nan)

# ------------------------------------------------------
# Computation of omega=fct(zeta)
# ------------------------------------------------------

# Lliboutry model for the horizontal flux shape function

# FIXME: Is there no better way to invert omega?
# Maybe a spline would allow to decrease the number of nodes.
zeta = np.linspace(1, 0, 1001).reshape(1001, 1)

omega = zeta * s + (1-s) * (1 - (p+2)/(p+1) * (1-zeta) +
                            1/(p+1) * np.power(1-zeta, p+2))

# -------------------------------------------------------
# Matrice omega : mat_omega
# -------------------------------------------------------

mat_omega = np.zeros((imax+1, imax+2))

mat_omega[:, 1:] = np.where(grid[:, 1:],
                            (np.dot(OMEGA.reshape(imax+1, 1),
                             Q[1:].reshape(1, imax+1))-Qm[1:])/(Q[1:]-Qm[1:]),
                            np.nan)
mat_omega[:, 0] = mat_omega[:, 1]

# -------------------------------------------------------
# Matrix mat_z_ie
# -------------------------------------------------------

mat_z_ie = np.zeros((imax+1, imax+2))

print('Before defining z_ie')
for j in range(1, imax+2):
    inter = np.interp(-mat_omega[:, j], -omega[:, j].flatten(),
                      zeta.flatten())
    mat_z_ie[:, j] = np.where(grid[:, j], B[j]+inter*(S_ie[j]-B[j]),
                              np.nan)
mat_z_ie[:, 0] = mat_z_ie[:, 1]
print('After defining z_ie')

# -------------------------------------------------------
# Matrix OMEGA: mat_OMEGA
# -------------------------------------------------------

mat_OMEGA = np.where(grid[:, 1:], OMEGA.reshape(imax+1, 1), np.nan)

# -------------------------------------------------------
# Matrix pi: mat_pi
# -------------------------------------------------------

mat_pi = np.where(grid[:, 1:], pi, np.nan)

# -------------------------------------------------------
# Matrix x: mat_x
# -------------------------------------------------------

mat_x = np.where(grid, x, np.nan)

# -------------------------------------------------------
# Matrix depth_ie: mat_depth_ie
# -------------------------------------------------------

mat_depth_ie = np.where(grid, S_ie - mat_z_ie, np.nan)

mat_depth_ie[0, :] = 0

# -------------------------------------------------------
# Matrix of stream function q: mat_q
# -------------------------------------------------------

mat_q = np.where(grid[:, 1:], Q[1:] * mat_OMEGA, np.nan)

# -------------------------------------------------------
# Matrix a0: mat_a0
# -------------------------------------------------------

print('Before defining mat_a0')

# FIXME: mat_0 does not have the same size as mat_x0
mat_a0 = np.where(grid[:, 1:], toeplitz(a[1]*np.ones(imax+1), a[1:]),
                  np.nan)

print('After defining mat_a0')

# -------------------------------------------------------
# Matrix x0: mat_x0
# -------------------------------------------------------

print('Before defining mat_x0')

mat_x0 = np.zeros((imax+1, imax+2))
mat_x0[:, 0] = np.where(grid[:, 0], 0, np.nan)

# x0 is not defined when trajectories reach the dome area, so we set to x[1].
# Maybe it would be better to put nans?
mat_x0[:, 1:] = np.where(grid[:, 1:], toeplitz(x[1]*np.ones(imax+1), x[1:]),
                         np.nan)

print('After defining mat_x0')

# -------------------------------------------------------
# Matrix STEADY-AGE:
# -------------------------------------------------------

print('Before calculation of steady age matrix.')

mat_steady_age = np.zeros((imax+1, imax+2))

# FIXME: Maybe make a matrix with dz/dOmega

for i in range(1, imax+1):
    if grid[i][1]:
        mat_steady_age[i][1] = mat_steady_age[i-1][1] + delta / a[1] \
            * (mat_z_ie[i-1][1] - mat_z_ie[i][1]) / (OMEGA[i-1] - OMEGA[i])
    else:
        mat_steady_age[i][1] = np.nan

# FIXME: This is actually wrong, since a[0]!=a[1], etc.
mat_steady_age[:, 0] = mat_steady_age[:, 1]

for j in range(2, imax+2):
    c = (a[j] - a[j-1]) / delta
    d = a[j] - c * pi[j-1]
    e = ((mat_z_ie[1:, j] - mat_z_ie[:-1, j]) / (OMEGA[1:] - OMEGA[:-1]) -
         (mat_z_ie[:-1, j-1] - mat_z_ie[1:, j-1]) / (OMEGA[:-1] - OMEGA[1:]))\
        / delta
    f = (mat_z_ie[1:, j] - mat_z_ie[:-1, j]) / (OMEGA[1:] - OMEGA[:-1]) -\
        e * pi[j-1]
    if c == 0:
        mat_steady_age[1:, j] = np.where(grid[1:, j],
                                         mat_steady_age[:-1, j-1] + (1 / d) *
                                         (e * (pi[j-1]*pi[j-1] -
                                          pi[j-2]*pi[j-2])
                                          / 2 + f * delta), np.nan)
    else:
        mat_steady_age[1:, j] = np.where(grid[1:, j],
                                         mat_steady_age[:-1, j-1] + (e*pi[j-1]
                                         + f)*log(abs(c*pi[j-1] + d))/c -
                                         (e * pi[j-2] + f) * log(abs(c*pi[j-2]
                                                                     + d))
                                         / c - (e/c)*((pi[j-1] + d/c) *
                                         log(abs(c * pi[j-1] + d))-(pi[j-2] +
                                                                    d/c)
                                         * log(abs(c * pi[j-2] + d)) - delta),
                                         np.nan)

# Theta_min and z_ie_min, which are the grid min for each vertical profile
# They are used for plotting the meshes
theta_min = np.nanmin(mat_theta, axis=0)
z_ie_min = np.nanmin(mat_z_ie, axis=0)

print('After calculation of steady age matrix.')

# Note: NaN dans coeff e dû à l'utilisation d'un schéma aux différences finies
# avant z_ie[ imax-1, imax] = NaN, pas de répercutions néfastes.

# Try to remove loops

# -------------------------------------------------------
# Matrix of thinning function: tau_ie
# -------------------------------------------------------

# FIXME: Use a more accurate scheme for thinning.
mat_tau_ie = np.where(grid[1:, 1:], (mat_z_ie[:-1, 1:] - mat_z_ie[1:, 1:])
                      / (mat_steady_age[1:, 1:] - mat_steady_age[:-1, 1:])
                      / (mat_a0[:-1, :] + mat_a0[1:, :]) * 2, np.nan)

# ----------------------------------------------------------
# Post-processing: transfert of the modeling results
# on the 1D grid of the drilling site
# ----------------------------------------------------------

# FIXME: We could have several drillings along the flow line.
# And each drilling would have its age_surf

# ----------------------------------------------------------
#  Computation of theta for the ice core: theta_ic
# ----------------------------------------------------------

if mat_depth_ie[imax, imax+1] < ie_depth[len(ie_depth)-1]:
    sys.exit("\n Attention problème d'interpolation post-processing:\n \
        le modèle donne des résultats jusqu'à une profondeur maximale trop \n \
        faible par rapport à la profondeur maximale du forage Vostok \n \
        Pour transposer correctement les résultats \n \
        du maillage sur toute la hauteur du forage Vostok il faut augmenter\n \
        le nombre de noeuds du maillage.")

theta_ic = np.log(np.interp(ie_depth, mat_depth_ie[:, imax+1]
                            [~np.isnan(mat_depth_ie[:, imax+1])],
                            mat_OMEGA[:, imax]
                            [~np.isnan(mat_OMEGA[:, imax])]))

# ----------------------------------------------------------
#  Computation steady a0 ice core
# ----------------------------------------------------------

steady_a0 = np.interp(-theta_ic, -theta[:][~np.isnan(mat_a0[:, imax])],
                      mat_a0[:, imax][~np.isnan(mat_a0[:, imax])])

# ----------------------------------------------------------
#  Computation of steady_age vostok icecore
# ----------------------------------------------------------

steady_age = np.interp(-theta_ic, -theta[:][~np.isnan(mat_steady_age[
    :, imax+1])], mat_steady_age[:, imax+1][~np.isnan(mat_steady_age[
        :, imax+1])])


# Cubic spline without derivative constraint

steady_age_sp = interp1d(-theta[:][~np.isnan(mat_steady_age[:, imax+1])],
                         mat_steady_age[:, imax+1][~np.isnan(mat_steady_age[
                             :, imax+1])], kind='cubic')(-theta_ic)

# Cubic spline with derivative constraint at surface
# On rajoute un point " proche de theta = 0 " afin d'imposer la dérivée
# Cela peu créer dans certains cas une matrice singulière (problème robustesse)
# FIXME: Is there not a less dirty solution?

new_theta = np.insert(-theta[:][~np.isnan(mat_steady_age[:, imax+1])], 1,
                      1/1000000)
chi_0 = np.insert(mat_steady_age[:, imax+1][~np.isnan(mat_steady_age[
    :, imax+1])], 1, 0.+1/(steady_a0[0])*(ie_depth[1] - ie_depth[0]) /
    (theta_ic[0] - theta_ic[1]) * 1/1000000)

steady_age_sp_2 = interp1d(new_theta, chi_0, kind='cubic')(-theta_ic)

# -------------------------------
# Computation of steady_age_R
# -------------------------------

steady_age_R = np.concatenate((np.array([age_R[0]]),
                               (age_R[1:] - age_R[:-1]) * R[:-1]))
steady_age_R = np.cumsum(steady_age_R)

# ----------------------------------------------------------
#  Computation of Age for the ice core
# ----------------------------------------------------------

# FIXME: Why "Age" has a capital "A" and not "steady_age"?

Age = np.interp(steady_age, steady_age_R, age_R)

# ----------------------------------------------------------
#  a0_ic
# ----------------------------------------------------------

# FIXME: We should not do a linear interpolation here.
# And therefore, a0_ic should be one element less.

a0_ic = steady_a0 * np.interp(steady_age, steady_age_R, R)

# ----------------------------------------------------------
#  Computation of tau_middle for the ice core
# ----------------------------------------------------------

tau_middle = 1./steady_a0[:-1] / (steady_age[1:] - steady_age[:-1])

# ----------------------------------------------------------
#  Computation of tau_ie_middle for the ice core
# ----------------------------------------------------------

# FIXME: check what is the most natural approach for thining

tau_ie_middle = (ie_depth[1:] - ie_depth[:-1]) / steady_a0[:-1] / \
                (steady_age[1:] - steady_age[:-1])

# Tau_ie with "natural cubic spline"

tau_ie_middle_sp = (ie_depth[1:] - ie_depth[:-1]) / steady_a0[:-1] / \
                   (steady_age_sp[1:] - steady_age_sp[:-1])

# Tau_ie with "cubic-spline - imposed derivative"

tau_ie_middle_sp_2 = (ie_depth[1:] - ie_depth[:-1]) / steady_a0[:-1] / \
                     (steady_age_sp_2[1:] - steady_age_sp_2[:-1])

# ----------------------------------------------------------
#  Computation of depth matrix: mat_depth
# ----------------------------------------------------------

mat_depth = np.interp(mat_depth_ie, np.append(ie_depth, ie_depth[-1]+10000.),
                      np.append(depth_corrected, depth_corrected[-1]+10000.))

# ----------------------------------------------------------
#  Computation of z matrix: mat_z
# ----------------------------------------------------------

mat_z = S - mat_depth

# ----------------------------------------------------------
#  Computation age matrix: mat_age
# ----------------------------------------------------------

# Rmq if age_R[0]>age_surf, there is a top layer of age age_R[0]
# FIXME: we should set the surface age in the YAML file for the age plot.
mat_age = np.interp(mat_steady_age, np.append(steady_age_R,
                                              100*steady_age_R[-1]),
                    np.append(age_R, 100*age_R[-1]))

# -----------
# FIGURES
# -----------

# FIXME: Some graphs are plotted with S_ie and other with S, be consistent.

if create_figs:

    print('Before creating figures.')

    # ----------------------------------------------------------
    # Display of (pi,theta) mesh
    # ----------------------------------------------------------

    fig, ax = plt.subplots()
    plt.vlines(pi, theta_min, theta_max, color='k', linewidths=0.1)
    for i in range(0, imax+1):
        plt.plot(pi, mat_theta[i, :], color='k', linewidth=0.1)
    plt.xlabel(r'$\pi$', fontsize=18)
    plt.ylabel(r'$\theta$', fontsize=18)
    plt.savefig(datadir+'mesh_pi_theta.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of (x, z) mesh
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(x, S_ie, label='Surface', color='0')
    plt.plot(x, B, label='Bedrock', color='0')
    # The vertical grid step can increase near the bed.
    # This is due do iso-omega layers being thicker near the bed.
    for i in range(0, imax+1):
        plt.plot(x[:], mat_z_ie[i, :],  ls='-', color='k', linewidth=0.1)
    plt.vlines(x, z_ie_min, S_ie, color='k', linewidths=0.1)
    plt.xlabel(r'$x$ (km)', fontsize=18)
    plt.ylabel(r'$z$ (m)', fontsize=18)
    plt.savefig(datadir+'mesh_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # -------------------------------------------------------------------------
    # Boundary conditions of the flow in (pi,theta)
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    fig.subplots_adjust(right=0.8)
    ax.set_xlabel('x (km)', fontsize=18)
    ax.set_ylabel('Y (relative unit)')
    ax.plot(pi, Y[1:], color='k')
    ax.spines.right.set_visible(False)

    color = 'g'
    ax1 = ax.twinx()
    ax1.spines['right'].set_position(('axes', 1.))
    ax1.spines['right'].set_color(color)
    ax1.plot(pi, a[1:], color=color)
    ax1.set_ylabel('a (m/yr)', color=color)
    ax1.tick_params(axis='y', colors=color)

    color = 'r'
    ax2 = ax.twinx()
    ax2.spines['right'].set_position(('axes', 1.09))
    ax2.spines['right'].set_color(color)
    ax2.plot(pi, m[1:], color=color)
    ax2.set_ylabel('m (m/yr)', color=color)
    ax2.tick_params(axis='y', colors=color)

    # -------------------------------------------------------------------------
    # Boundary conditions of the flow in (x,z)
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    fig.subplots_adjust(right=0.8)
    ax.set_xlabel('x (km)', fontsize=18)
    ax.set_ylabel('Y (relative unit)')
    ax.plot(x, Y, color='k')
    ax.spines.right.set_visible(False)

    ax1 = ax.twinx()
    ax1.spines['right'].set_position(('axes', 1.))
    ax1.spines['right'].set_color('g')
    ax1.plot(x, a, color='g')
    ax1.set_ylabel('a (m/yr)', color='g')
    ax1.tick_params(axis='y', colors='g')

    ax2 = ax.twinx()
    ax2.spines['right'].set_position(('axes', 1.09))
    ax2.spines['right'].set_color('r')
    ax2.plot(x, m, color='r')
    ax2.set_ylabel('m (m/yr)', color='r')
    ax2.tick_params(axis='y', colors='r')

    # ----------------------------------------------------------
    # Display of iso-omega lines in (x, z)
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(x[1:], S[1:], label='Surface', color='0')
    plt.plot(x[1:], B[1:], label='Bedrock', color='0')
    levels = np.arange(0, 1.01, 0.01)
    levels_cb = np.arange(0, 11, 1)/10.
    # There is no node on the bedrock, so the color does not go down there.
    cp = plt.contourf(mat_x[:, 1:], mat_z[:, 1:], mat_omega[:, 1:],
                      levels=levels,
                      cmap='jet')
    cp2 = plt.contour(mat_x[:, 1:], mat_z[:, 1:], mat_omega[:, 1:],
                      levels=levels_cb,
                      colors='k')
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    cb.set_label(r'$\omega$')
    ax.set_xlabel(r'$x$ (km)', fontsize=19)
    ax.set_ylabel(r'$z$ (m)', fontsize=19)
    plt.savefig(datadir+'iso-omega_lines.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of age and isochrones in (x, z)
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(x[1:], S[1:], label='Surface', color='0')
    plt.plot(x[1:], B[1:], label='Bedrock', color='0')

    # FIXME: Could we plot refrozen ice here?

    levels = np.arange(0, fig_age_max, fig_age_spacing)
    levels_cb = np.arange(0, fig_age_max, fig_age_spacing_labels)
    cp = plt.contourf(mat_x[:, 1:], mat_z[:, 1:], mat_age[:, 1:]/1000.,
                      levels=levels,
                      cmap='jet')
    cp2 = plt.contour(mat_x[:, 1:], mat_z[:, 1:], mat_age[:, 1:]/1000.,
                      levels=levels_cb,
                      colors='k')
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    cb.set_label('Modeled age (kyr)')
    ax.set_xlabel(r'$x$ (km)', fontsize=19)
    ax.set_ylabel(r'$z$ (m)', fontsize=19)
    ax.grid()
    plt.savefig(datadir+'age_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ---------------------------------------------------------------------
    # Display of age and isochrones in (pi,theta)
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(pi, np.zeros_like(S[1:]), label='Surface', color='0')
    plt.plot(pi, theta_min, label='Bedrock', color='0')

    levels = np.arange(0, fig_age_max, fig_age_spacing)
    levels_cb = np.arange(0, fig_age_max, fig_age_spacing_labels)
    cp = plt.contourf(mat_pi, mat_theta, mat_age[:, 1:]/1000., levels=levels,
                      cmap='jet')
    cp2 = plt.contour(mat_pi, mat_theta, mat_age[:, 1:]/1000.,
                      levels=levels_cb, colors='k')
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    cb.set_label('Modeled age (kyr)')
    ax.set_xlabel(r'$\pi$', fontsize=19)
    ax.set_ylabel(r'$\theta$', fontsize=19)
    ax.grid()
    plt.savefig(datadir+'age_pi_theta.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of thinning function
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(x[1:], S[1:], label='Surface', color='0')
    plt.plot(x[1:], B[1:], label='Bedrock', color='0')

    levels = np.arange(0, 1.21, 0.01)
    levels_cb = np.arange(0, 13, 1)/10.
    cp = plt.contourf(mat_x[1:, 1:], (mat_z[1:, 1:] + mat_z[:-1, 1:])/2,
                      mat_tau_ie,
                      levels=levels,
                      cmap='jet')
    cp2 = plt.contour(mat_x[1:, 1:], (mat_z[1:, 1:] + mat_z[:-1, 1:])/2,
                      mat_tau_ie,
                      levels=levels_cb,
                      colors='k')
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    cb.set_label('Thinning function (no unit))')
    ax.set_xlabel(r'$x$ (km)', fontsize=19)
    ax.set_ylabel(r'$z$ (m)', fontsize=19)
    ax.grid()
    plt.savefig(datadir+'thinning_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of stream lines
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(15, 7))
    # Rmq: We use plt.contour instead of plotting the individual lines, since
    # it is simpler and slightly faster.
    # Rmq2: We don't exactly go down to the bedrock here but this is normal.
    # Trajectories that come from the surface and traj that come from the dome.
    levels = np.concatenate((Q[-1:0:-traj_step],
                             mat_q[traj_step::traj_step, 0]))
    levels = np.flip(levels[~np.isnan(levels)])
    color = 'k'
    lw = 0.2
    plt.contour(mat_x[:, 1:], mat_z[:, 1:], mat_q, colors=color,
                levels=levels, linewidths=lw)
    # Color contour plot.
    from matplotlib import ticker
    cp = plt.contourf(mat_x[:, 1:], mat_z[:, 1:], mat_q, levels=levels,
                      locator=ticker.LogLocator())
#    cb = plt.colorbar(cp)
    # plt.vlines(x[0], B[0], S[0], color='blue')  # ice divide
    # plt.vlines(x[1], B[1], S[1], color='blue')  # 1st horizontal node
    plt.plot(x[1:], S[1:], label='Surface', color='0')
    # Fake plot for the legend
    plt.plot(x[1], B[1], label="Trajectories", color=color, linewidth=lw)
    plt.plot(x[1:], B[1:], label='Bedrock', color='0')
    plt.legend(loc='lower left')
    plt.xlim([-5, x[imax+1] + 5])
    plt.xlabel(r'$x$ (km)', fontsize=19)
    plt.ylabel(r'$z$ (m)', fontsize=19)
    plt.grid()
    plt.savefig(datadir+'stream_lines.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ---------------------------------------------------------------------
    # R(t) - Age
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots()
    plt.stairs(R[:-1], age_R/1000, baseline=None)
    plt.xlabel('time (kyr)', fontsize=15)
    plt.ylabel(r'$R(t)$', fontsize=15)
    plt.savefig(datadir+'R_temporal_factor.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Graphs vs depth for the ice core
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_ylabel('depth (m)')
    ax.invert_yaxis()

    ax.plot(mat_x0[:, imax+1], mat_depth[:, imax+1], color='r')
    ax.set_xlabel(r'$x$ origin (km)', color='r')
    ax.spines['bottom'].set_color('r')
    ax.tick_params(axis='x', colors='r')

    ax2 = ax.twiny()
    ax2.spines.bottom.set_visible(False)
    ax2.plot(Age/1000, depth_corrected, color='b')
    ax2.set_xlabel('age (kyr)', color='b')
    ax2.spines['top'].set_color('b')
    ax2.tick_params(axis='x', colors='b')

    ax3 = ax.twiny()
    ax3.spines['top'].set_position(('axes', 1.1))
    ax3.spines.bottom.set_visible(False)
    ax3.plot(tau_ie_middle_sp_2, depth_corrected[:-1], color='g')
    ax3.set_xlabel('thinning function (no unit)', color='g')
    ax3.spines['top'].set_color('g')
    ax3.tick_params(axis='x', colors='g')

    plt.savefig(datadir+'ice_core_vs_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------
    # Graphs vs age for the ice core
    # ----------------------------------------------

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlabel('age (kyr)')
    ax.set_ylabel('layer thickness (m/yr)')
    ax.stairs(a0_ic[:-1], Age/1000, baseline=None, label='accumulation')
    ax.stairs(tau_ie_middle_sp_2 * a0_ic[:-1], Age/1000, baseline=None,
              label='layer thickness')
    ax.legend()

    plt.savefig(datadir+'ice_core_vs_age.'+fig_format,
                format=fig_format, bbox_inches='tight')

    plt.show()

# Program execution time
MESSAGE = 'Program execution time: '+str(time.perf_counter()-START_TIME)+' s.'
print(MESSAGE)
