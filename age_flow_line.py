import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
import yaml
import matplotlib.pyplot as plt
import time


def interp_stair_aver(x_out, x_in, y_in):
    """Return a staircase interpolation of a (x_in,y_in) series
    at x_out abscissas with averaging."""
    x_mod = x_in+0
    y_mod = y_in+0
    if x_out[0] < x_in[0]:
        x_mod = np.concatenate((np.array([x_out[0]]), x_mod))
        y_mod = np.concatenate((np.array([y_in[0]]), y_mod))
    if x_out[-1] > x_in[-1]:
        x_mod = np.concatenate((x_mod, np.array([x_out[-1]])))
        y_mod = np.concatenate((y_mod, np.array([y_in[-1]])))
    y_int = np.cumsum(np.concatenate((np.array([0]),
                                      y_mod[:-1]*(x_mod[1:]-x_mod[:-1]))))
# Maybe this is suboptimal since we compute twice g(xp[i]):
    y_out = (np.interp(x_out[1:], x_mod, y_int) -
             np.interp(x_out[:-1], x_mod, y_int)) / (x_out[1:]-x_out[:-1])
    return y_out


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
D_depth, D_D = np.loadtxt(datadir+'relative_density.txt', unpack=True)

# ---------------------------------------------------------
# Reading parameters.yml file (imax, delta, ...)
# ---------------------------------------------------------

# Default values for parameters, to prevent spyder errors
max_depth = 3310.
step_depth = 1.
imax = 100
delta = 0.08
age_surf = -50
x_right = 370
traj_step = 10
fig_age_max = 1000000
fig_age_spacing = 10000
fig_age_spacing_labels = 100000
beta = 0.015
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

x_fld = np.concatenate((x_a0, x_m, x_Y))
x_fld = x_fld[x_fld <= x_right]
x_fld = np.unique(np.sort(x_fld))

a0_fld = np.interp(x_fld, x_a0, a0_measure)
Y_fld = np.interp(x_fld, x_Y, Y_measure)
m_fld = np.interp(x_fld, x_m, m_measure)

# Computation of total flux Q
# Formula checked 2023/04/27 by F. Parrenin
dQdx = (x_fld[1:]-x_fld[:-1]) * 1000 * \
    (a0_fld[:-1] * Y_fld[:-1] +
     0.5 * ((a0_fld[1:]-a0_fld[:-1]) * Y_fld[:-1] + (Y_fld[1:]-Y_fld[:-1])
            * a0_fld[:-1]) +
     1./3 * (a0_fld[1:]-a0_fld[:-1]) * (Y_fld[1:]-Y_fld[:-1]))
dQdx = np.insert(dQdx, 0, 0)
Q_fld = np.cumsum(dQdx)

# Computation of basal melting flux Qm
dQmdx = (x_fld[1:]-x_fld[:-1]) * 1000 * \
    (m_fld[:-1] * Y_fld[:-1] +
     0.5 * ((m_fld[1:]-m_fld[:-1]) * Y_fld[:-1] + (Y_fld[1:]-Y_fld[:-1])
            * m_fld[:-1]) +
     1./3 * (m_fld[1:]-m_fld[:-1]) * (Y_fld[1:]-Y_fld[:-1]))
dQmdx = np.insert(dQmdx, 0, 0)
Qm_fld = np.cumsum(dQmdx)

# ----------------------------------------------------------
# Mesh generation (pi,theta)
# ----------------------------------------------------------

pi = np.linspace(-imax * delta, 0,  imax + 1)

# Rmk: We could make a column vector here and for OMEGA
theta = np.linspace(0, - imax * delta,  imax + 1)

# ----------------------------------------------------------
# Total flux Q(m^3/yr)
# ----------------------------------------------------------

Q = Q_fld[-1] * np.exp(pi)  # Q_ref = Q_fld[-1]

# ----------------------------------------------------------
# OMEGA
# ----------------------------------------------------------

OMEGA = np.exp(theta)

# ----------------------------------------------------------
# interpolation of flow line data files for x, Qm, ...
# ----------------------------------------------------------

# We need to interpolate x in Q, but then we can interpolate in x.
# We could also interpolate everything in Q.
x = np.interp(Q, Q_fld, x_fld)
a = np.interp(x, x_fld, a0_fld)
Qm = np.interp(x, x_fld, Qm_fld)
Y = np.interp(x, x_Y, Y_measure)
S = np.interp(x, x_Su, Su_measure)
B = np.interp(x, x_B, B_measure)
s = np.interp(x, x_s, s_measure)
p = np.interp(x, x_p, p_measure)

# -----------------------------------------------------
# depth vs ie-depth conversion with density data
# -----------------------------------------------------

D_depth_ie = np.cumsum(np.concatenate((np.array([0]),
                                       D_D[:-1] * (D_depth[1:]-D_depth[:-1]))))

# ----------------------------------------------------------
# DELTA H
# ----------------------------------------------------------

DELTA_H = D_depth[-1] - D_depth_ie[-1]

# --------------------------------------------------
# Calcul de la surface ice-equivalent S_ie
# --------------------------------------------------

S_ie = S - DELTA_H

# --------------------------------------------------
# Melting
# --------------------------------------------------

# m is just used for the boundary conditions plot.
m = np.interp(x, x_fld, m_fld)

# ------------------------------------------------------
# Computation of theta_min and theta_max
# ------------------------------------------------------

# We just use theta_max for the mesh plot
theta_max = np.zeros(imax+1)

theta_min = np.where(Qm > 0,
                     np.maximum(np.log(Qm.clip(min=10**-100)/Q),
                                -imax*delta * np.ones((imax+1,))),
                     -imax*delta * np.ones((imax+1,)))


# -------------------------------------------------------
# GRID
# -------------------------------------------------------

grid = np.ones((imax + 1, imax + 1), dtype=bool)

grid[:, 0] = theta >= theta_min[0]

print('Before defining grid boolean',
      round(time.perf_counter()-START_TIME, 4), 's.')
# We need an iteration here, to treat a column after the previous one
# Maybe it would be possible to have a matrix formula with logical_and, but is
# it worth it?
for j in range(1, imax+1):
    grid[1:, j] = np.logical_and(theta[1:] >= theta_min[j-1], grid[0:-1, j-1])
print('After defining grid boolean ',
      round(time.perf_counter()-START_TIME, 4), 's.')

# -------------------------------------------------------
# Matrix theta
# -------------------------------------------------------

mat_theta = theta.reshape(imax+1, 1)*np.ones((1, imax+1))
mat_theta = np.where(grid, mat_theta, np.nan)
# theta_min_mesh is the grid min for each vertical profile, to plot the mesh.
theta_min_mesh = np.nanmin(mat_theta, axis=0)

# -------------------------------------------------------
# Matrice omega : mat_omega
# -------------------------------------------------------

mat_omega = np.zeros((imax+1, imax+1))

mat_omega = np.where(grid,
                     (np.dot(OMEGA.reshape(imax+1, 1),
                             Q.reshape(1, imax+1))-Qm)/(Q-Qm),
                     np.nan)

# ------------------------------------------------------
# Computation of omega=fct(zeta)
# ------------------------------------------------------

print('Before defining z_ie',
      round(time.perf_counter()-START_TIME, 4), 's.')

# Lliboutry model for the horizontal flux shape function

# Rmq: We could try more accurate solutions like pynverse or
# scipy.optimize.root_scalar, but it will probably be slower.
zeta = np.linspace(1, 0, 1001).reshape(1001, 1)

omega = zeta * s + (1-s) * (1 - (p+2)/(p+1) * (1-zeta) +
                            1/(p+1) * np.power(1-zeta, p+2))

# -------------------------------------------------------
# Matrix mat_z_ie
# -------------------------------------------------------

mat_z_ie = np.zeros((imax+1, imax+1))
# Rmq: I don't see a way to prevent the loop here, because of interp.
for j in range(0, imax+1):
    inter = np.interp(-mat_omega[:, j], -omega[:, j].flatten(),
                      zeta.flatten())
    mat_z_ie[:, j] = np.where(grid[:, j], B[j]+inter*(S_ie[j]-B[j]),
                              np.nan)

# z_ie_min is the grid min for each vertical profile, used to plot the mesh
z_ie_min_mesh = np.nanmin(mat_z_ie, axis=0)

print('After defining z_ie',
      round(time.perf_counter()-START_TIME, 4), 's.')

# -------------------------------------------------------
# Matrix OMEGA: mat_OMEGA
# -------------------------------------------------------

mat_OMEGA = np.where(grid, OMEGA.reshape(imax+1, 1), np.nan)

# -------------------------------------------------------
# Matrix pi: mat_pi
# -------------------------------------------------------

mat_pi = np.where(grid, pi, np.nan)

# -------------------------------------------------------
# Matrix x: mat_x
# -------------------------------------------------------

mat_x = np.where(grid, x, np.nan)

# -------------------------------------------------------
# Matrix depth_ie: mat_depth_ie
# -------------------------------------------------------

mat_depth_ie = np.where(grid, S_ie - mat_z_ie, np.nan)

mat_depth_ie[0, :] = 0

# ----------------------------------------------------------
#  Computation of depth matrix: mat_depth
# ----------------------------------------------------------

mat_depth = np.interp(mat_depth_ie, np.append(D_depth_ie,
                                              D_depth_ie[-1]+10000.),
                      np.append(D_depth, D_depth[-1]+10000.))

# ----------------------------------------------------------
#  Computation of z matrix: mat_z
# ----------------------------------------------------------

mat_z = S - mat_depth

# -------------------------------------------------------
# Matrix of stream function q: mat_q
# -------------------------------------------------------

mat_q = np.where(grid, Q * mat_OMEGA, np.nan)

# -------------------------------------------------------
# Matrix a0: mat_a0
# -------------------------------------------------------

print('Before defining mat_a0',
      round(time.perf_counter()-START_TIME, 4), 's.')

mat_a0 = np.where(grid, toeplitz(a[0]*np.ones(imax+1), a),
                  np.nan)

print('After defining mat_a0',
      round(time.perf_counter()-START_TIME, 4), 's.')

# -------------------------------------------------------
# Matrix x0: mat_x0
# -------------------------------------------------------

print('Before defining mat_x0',
      round(time.perf_counter()-START_TIME, 4), 's.')

mat_x0 = np.zeros((imax+1, imax+1))

# x0 is not defined when trajectories reach the dome area, so we set to x[0].
# Maybe it would be better to put nans?
mat_x0 = np.where(grid, toeplitz(x[0]*np.ones(imax+1), x),
                  np.nan)

print('After defining mat_x0',
      round(time.perf_counter()-START_TIME, 4), 's.')

# -------------------------------------------------------
# Matrix STEADY-AGE:
# -------------------------------------------------------

print('Before calculation of steady age matrix.',
      round(time.perf_counter()-START_TIME, 4), 's.')

mat_steady_age = np.zeros((imax+1, imax+1))

# Dome boundary condition
mat_steady_age[1:, 0] = delta / a[0] * np.cumsum((mat_z_ie[:-1, 0] -
                                                  mat_z_ie[1:, 0]) /
                                                 (OMEGA[:-1] - OMEGA[1:]))

dzdOMEGA = (mat_z_ie[1:, :] - mat_z_ie[:-1, :]) /\
           (OMEGA[1:] - OMEGA[:-1]).reshape(imax, 1)
# Calculation line by line, F. Parrenin, 2023/04/28
# Rmq : It is possible to calculate column by column or diagonal by diagonal
for i in range(1, imax+1):
    mat_steady_age[i, 1:] = mat_steady_age[i-1, :-1] + delta * (
        dzdOMEGA[i-1, :-1]/a[:-1] +
        0.5 * (dzdOMEGA[i-1, 1:] - dzdOMEGA[i-1, :-1])/a[:-1] +
        0.5 * dzdOMEGA[i-1, :-1] * (1/a[1:] - 1/a[:-1]) +
        1./3 * (dzdOMEGA[i-1, 1:] - dzdOMEGA[i-1, :-1]) * (1/a[1:] - 1/a[:-1]))
mat_steady_age = np.where(grid, mat_steady_age, np.nan)

print('After calculation of steady age matrix.',
      round(time.perf_counter()-START_TIME, 4), 's.')

# -------------------------------------------------------
# Matrix of thinning function: tau_ie
# -------------------------------------------------------

# FIXME: Use a more accurate scheme for thinning.
mat_tau_ie = np.where(grid[1:, :], (mat_z_ie[:-1, :] - mat_z_ie[1:, :])
                      / (mat_steady_age[1:, :] - mat_steady_age[:-1, :])
                      / (mat_a0[:-1, :] + mat_a0[1:, :]) * 2, np.nan)

# -------------------------------
# Computation of steady_age_R
# -------------------------------

steady_age_R = np.concatenate((np.array([age_R[0]]),
                               (age_R[1:] - age_R[:-1]) * R[:-1]))
steady_age_R = np.cumsum(steady_age_R)

# ----------------------------------------------------------
#  Computation age matrix: mat_age
# ----------------------------------------------------------

# Rmq if age_R[0]>age_surf, there is a top layer of age age_R[0]
mat_age = np.interp(mat_steady_age+age_surf,
                    np.append(steady_age_R, 100*steady_age_R[-1]),
                    np.append(age_R, 100*age_R[-1]))

# ----------------------------------------------------------
# Post-processing: transfering of the modeling results
# on the 1D grid of the drilling site
# ----------------------------------------------------------

# FIXME: We could have several drillings along the flow line.
# And each drilling could have its own age_surf
# FIXME: The position of the ice core could be defined in the YAML file

print('Before calculating for the ice core',
      round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------
# Depth_ic and ie_depth_ic along the drilling
# ---------------------------------------------------

depth_ic = np.arange(0., max_depth + 0.0001, step_depth)
ie_depth_ic = np.interp(depth_ic, D_depth, D_depth_ie)

# ----------------------------------------------------------
#  Computation of theta for the ice core: theta_ic
# ----------------------------------------------------------

if mat_depth_ie[imax, imax] < ie_depth_ic[len(ie_depth_ic)-1]:
    sys.exit("The mesh does not extend down to the bottom of the core.")

theta_ic = np.log(np.interp(ie_depth_ic, mat_depth_ie[:, imax]
                            [~np.isnan(mat_depth_ie[:, imax])],
                            mat_OMEGA[:, imax]
                            [~np.isnan(mat_OMEGA[:, imax])]))

# ----------------------------------------------------------
#  Computation steady a0 ice core
# ----------------------------------------------------------

steady_a0_ic = np.interp(-theta_ic, -theta[:][~np.isnan(mat_a0[:, imax])],
                         mat_a0[:, imax][~np.isnan(mat_a0[:, imax])])

# ----------------------------------------------------------
#  Computation of steady_age vostok icecore
# ----------------------------------------------------------

# Cubic spline with derivative constraint at surface
# We had a point close to the surface to impose the derivative of the age

new_theta_ic = np.insert(-theta[:][~np.isnan(mat_steady_age[:, imax])], 1,
                         1/1000000)
chi_0 = np.insert(mat_steady_age[:, imax][~np.isnan(mat_steady_age[
    :, imax])], 1, 0.+1/(steady_a0_ic[0])*(ie_depth_ic[1] - ie_depth_ic[0]) /
    (theta_ic[0] - theta_ic[1]) * 1/1000000)

steady_age_ic = interp1d(new_theta_ic, chi_0, kind='cubic')(-theta_ic)

# ----------------------------------------------------------
#  Computation of age for the ice core
# ----------------------------------------------------------

age_ic = np.interp(steady_age_ic+age_surf, steady_age_R, age_R)
print('Bottom age for the ice core:', age_ic[-1])

# ----------------------------------------------------------
#  a0_ic
# ----------------------------------------------------------

# Here, steady_a0_ic is at the node, while a0_ic is for an interval
a0_ic = (steady_a0_ic[1:]+steady_a0_ic[:-1])/2 *\
    interp_stair_aver(steady_age_ic, steady_age_R, R)

# ----------------------------------------------------------
#  Computation of tau_middle for the ice core
# ----------------------------------------------------------

aa = (steady_a0_ic[1:]+steady_a0_ic[:-1]) / 2
tau_middle = 1./aa / (steady_age_ic[1:] - steady_age_ic[:-1])

# ----------------------------------------------------------
#  Computation of tau_ie_middle for the ice core
# ----------------------------------------------------------

tau_ie_middle = (ie_depth_ic[1:] - ie_depth_ic[:-1]) / aa / \
                (steady_age_ic[1:] - steady_age_ic[:-1])

# FIXME: there is no output in the program. Maybe output at least for the core.

# -----------
# FIGURES
# -----------

if create_figs:

    print('Before creating figures.',
          round(time.perf_counter()-START_TIME, 4), 's.')

    # ----------------------------------------------------------
    # Display of (pi,theta) mesh
    # ----------------------------------------------------------

    fig, ax = plt.subplots()
    plt.vlines(pi, theta_min_mesh, theta_max, color='k', linewidths=0.1)
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
    plt.plot(x, S, label='Surface', color='0')
    plt.plot(x, B, label='Bedrock', color='0')
    # The vertical grid step can increase near the bed.
    # This is due do iso-omega layers being thicker near the bed.
    for i in range(0, imax+1):
        plt.plot(x, mat_z[i, :],  ls='-', color='k', linewidth=0.1)
    plt.vlines(x, z_ie_min_mesh, S, color='k', linewidths=0.1)
    plt.xlabel(r'$x$ (km)', fontsize=18)
    plt.ylabel(r'$z$ (m)', fontsize=18)
    plt.savefig(datadir+'mesh_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # -------------------------------------------------------------------------
    # Boundary conditions of the flow in pi
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    fig.subplots_adjust(right=0.8)
    ax.set_xlabel('x (km)', fontsize=18)
    ax.set_ylabel('Y (relative unit)')
    ax.plot(pi, Y, color='k')
    ax.spines.right.set_visible(False)

    color = 'g'
    ax1 = ax.twinx()
    ax1.spines['right'].set_position(('axes', 1.))
    ax1.spines['right'].set_color(color)
    ax1.plot(pi, a, color=color)
    ax1.set_ylabel('a (m/yr)', color=color)
    ax1.tick_params(axis='y', colors=color)

    color = 'r'
    ax2 = ax.twinx()
    ax2.spines['right'].set_position(('axes', 1.09))
    ax2.spines['right'].set_color(color)
    ax2.plot(pi, m, color=color)
    ax2.set_ylabel('m (m/yr)', color=color)
    ax2.tick_params(axis='y', colors=color)

    # -------------------------------------------------------------------------
    # Boundary conditions of the flow in x
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
    plt.plot(x, S, label='Surface', color='0')
    plt.plot(x, B, label='Bedrock', color='0')
    levels = np.arange(0, 1.01, 0.01)
    levels_cb = np.arange(0, 11, 1)/10.
    # There is no node on the bedrock, so the color does not go down there.
    cp = plt.contourf(mat_x, mat_z, mat_omega,
                      levels=levels,
                      cmap='jet')
    cp2 = plt.contour(mat_x, mat_z, mat_omega,
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

    plt.plot(x, S, label='Surface', color='0')
    plt.plot(x, B, label='Bedrock', color='0')

    # FIXME: Could we plot refrozen ice here?

    levels = np.arange(0, fig_age_max, fig_age_spacing)
    levels_cb = np.arange(0, fig_age_max, fig_age_spacing_labels)
    cp = plt.contourf(mat_x, mat_z, mat_age/1000.,
                      levels=levels,
                      cmap='jet')
    cp2 = plt.contour(mat_x, mat_z, mat_age/1000.,
                      levels=levels_cb,
                      colors='k')
    # Corner trajectory
    level0 = np.array([Q[0]])
    plt.contour(mat_x, mat_z, mat_q, colors='k', linestyles='dashed',
                levels=level0, linewidths=1)
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

    plt.plot(pi, theta_max, label='Surface', color='0')
    plt.plot(pi, theta_min_mesh, label='Bedrock', color=None)

    levels = np.arange(0, fig_age_max, fig_age_spacing)
    levels_cb = np.arange(0, fig_age_max, fig_age_spacing_labels)
    cp = plt.contourf(mat_pi, mat_theta, mat_age/1000., levels=levels,
                      cmap='jet')
    cp2 = plt.contour(mat_pi, mat_theta, mat_age/1000.,
                      levels=levels_cb, colors='k')
    # Corner trajectory
    level0 = np.array([Q[0]])
    plt.contour(mat_pi, mat_theta, mat_q, colors='k', linestyles='dashed',
                levels=level0, linewidths=1)
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

    plt.plot(x, S, label='Surface', color='0')
    plt.plot(x, B, label='Bedrock', color='0')
    zz = np.insert((mat_z[1:, :] + mat_z[:-1, :])/2, 0, S, axis=0)
    tt = np.insert(mat_tau_ie, 0, np.ones(imax+1), axis=0)
    levels = np.arange(0, 1.21, 0.01)
    levels_cb = np.arange(0, 13, 1)/10.
    cp = plt.contourf(mat_x, zz, tt,
                      levels=levels,
                      cmap='jet')
    cp2 = plt.contour(mat_x, zz, tt,
                      levels=levels_cb,
                      colors='k')
    # Corner trajectory
    level0 = np.array([Q[0]])
    plt.contour(mat_x, mat_z, mat_q, colors='k', linestyles='dashed',
                levels=level0, linewidths=1)
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
                             mat_q[0::traj_step, 0]))
    levels = np.flip(levels[~np.isnan(levels)])
    color = 'k'
    lw = 0.2
    plt.contour(mat_x, mat_z, mat_q, colors=color,
                levels=levels, linewidths=lw)
    # Corner trajectory
    level0 = np.array([Q[0]])
    plt.contour(mat_x, mat_z, mat_q, colors='k', linestyles='dashed',
                levels=level0, linewidths=1)
    # Color contour plot.
    from matplotlib import ticker
    cp = plt.contourf(mat_x, mat_z, mat_q, levels=levels,
                      locator=ticker.LogLocator())
    plt.plot(x, S, label='Surface', color='0')
    # Fake plots for the legend
    plt.plot(x, B, label="Trajectories", color=color, linewidth=lw)
    plt.plot(x, B, label="Corner trajectory", color='k', linewidth=1,
             linestyle='dashed')
    plt.plot(x, B, label='Bedrock', color='0')
    plt.legend(loc='lower left')
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

    ax.plot(mat_x0[:, imax], mat_depth[:, imax], color='r')
    ax.set_xlabel(r'$x$ origin (km)', color='r')
    ax.spines['bottom'].set_color('r')
    ax.tick_params(axis='x', colors='r')

    ax2 = ax.twiny()
    ax2.spines.bottom.set_visible(False)
    ax2.plot(age_ic/1000, depth_ic, color='b')
    ax2.set_xlabel('age (kyr)', color='b')
    ax2.spines['top'].set_color('b')
    ax2.tick_params(axis='x', colors='b')

    ax3 = ax.twiny()
    ax3.spines['top'].set_position(('axes', 1.1))
    ax3.spines.bottom.set_visible(False)
    ax3.plot(tau_ie_middle, depth_ic[:-1], color='g')
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
    ax.stairs(a0_ic, age_ic/1000, baseline=None, label='accumulation')
    ax.stairs(tau_ie_middle * a0_ic, age_ic/1000, baseline=None,
              label='layer thickness')
    ax.legend()

    plt.savefig(datadir+'ice_core_vs_age.'+fig_format,
                format=fig_format, bbox_inches='tight')

    plt.show()

# Program execution time
MESSAGE = 'Program execution time: '+str(time.perf_counter()-START_TIME)+' s.'
print(MESSAGE)
