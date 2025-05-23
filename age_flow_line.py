import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
import yaml
import matplotlib.pyplot as plt
import time
import math
import os
import resource

# FIXME: Use firn density profile from Salamatin's equation
# FIXME: Try again natural sampling with an intelligent interpolation fct.
# FIXME: Use an intelligent interpolation with linear temporal factor
# TODO: Implement R as a fuction of depth


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

# ---------------------------------------------------------
# Reading parameters.yml file (imax, delta, ...)
# ---------------------------------------------------------

# Default values for parameters, to prevent spyder errors
ic = {}
imax = 100
delta = 0.08
age_surf = -50
x_right = 370.
x_step = 1.
thickness_ie = False
accu_relative = 1.
accu_present = True
R_exp = 1.
temp_fact_linear = False
traj_step = 10
fig_age_max = 1000000
fig_age_spacing = 10000
fig_age_spacing_labels = 100000
fig_age_iso = [100, 200, 300, 400]
beta = 0.015
create_figs = True
fig_format = 'pdf'
comp_flowline = None
comp_isochrones = None
obs_bedrock = None
output_ic = True
output_fl = True

yamls = open(datadir+'parameters.yml').read()
para = yaml.load(yamls, Loader=yaml.FullLoader)
globals().update(para)

# -----------------------------------------------------------------------------
# Loading data from temporal_factor.txt and relative_density.txt
# -----------------------------------------------------------------------------

# deut = np.loadtxt(datadir+'deuterium.txt')
age_R, R = np.loadtxt(datadir+'temporal_factor.txt', unpack=True)
age_R = np.append(age_R, age_R[-1]+1)
age_R = np.append(age_R, 10000000)
R = np.append(R, 1)
R = np.append(R, 1)
R = R**R_exp
if temp_fact_linear:
    R = (R[1:] + R[:-1])/2
    R = np.append(R, R[-1])

D_depth, D_D = np.loadtxt(datadir+'relative_density.txt', unpack=True)
D_depth = np.append(D_depth, D_depth[-1]+0.01)
D_depth = np.append(D_depth, 10000)
D_D = np.append(D_D, 1)
D_D = np.append(D_D, 1)

# -----------------------------------------------------
# Loading files for Geographic data, arrays creations
# -----------------------------------------------------

# Steady accumulation
x_a, a_measure = np.loadtxt(datadir+'accumulation.txt', unpack=True)
a_measure = a_measure * accu_relative
if accu_present:
    a_measure = a_measure / R[0]

# Melting
x_m, m_measure = np.loadtxt(datadir+'melting.txt', unpack=True)

# Sliding rate
x_s, s_measure = np.loadtxt(datadir+'sliding.txt', unpack=True)

# Lliboutry parameter
x_p, p_measure = np.loadtxt(datadir+'p_Lliboutry.txt', unpack=True)

# Surface
x_Su, Su_measure = np.loadtxt(datadir+'surface.txt', unpack=True)

# Thickness
x_H, H_measure = np.loadtxt(datadir+'thickness.txt', unpack=True)

# Tube width
x_Y, Y_measure = np.loadtxt(datadir+'tube_width.txt', unpack=True)

# --------------------
# Interpolation
# --------------------

x_fld = np.arange(0, x_right+x_step, x_step)

a_fld = np.interp(x_fld, x_a, a_measure)
Y_fld = np.interp(x_fld, x_Y, Y_measure)
m_fld = np.interp(x_fld, x_m, m_measure)

# Computation of total flux Q
# Formula checked 2023/04/27 by F. Parrenin
dQdx = (x_fld[1:]-x_fld[:-1]) * 1000 * \
    (a_fld[:-1] * Y_fld[:-1] +
     0.5 * ((a_fld[1:]-a_fld[:-1]) * Y_fld[:-1] + (Y_fld[1:]-Y_fld[:-1])
            * a_fld[:-1]) +
     1./3 * (a_fld[1:]-a_fld[:-1]) * (Y_fld[1:]-Y_fld[:-1]))
# Simple scheme, for checking
# dQdx = (a0_fld[1:]+a0_fld[:-1])*(Y_fld[1:]+Y_fld[:-1])/4 * \
#    (x_fld[1:]-x_fld[:-1])*1000
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
Qm = np.interp(Q, Q_fld, Qm_fld)
a = np.interp(x, x_a, a_measure)
Y = np.interp(x, x_Y, Y_measure)
S = np.interp(x, x_Su, Su_measure)
H = np.interp(x, x_H, H_measure)
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
print('DELTA_H:', DELTA_H)

# --------------------------------------------------
# Computation of H, S_ie and B
# --------------------------------------------------

if thickness_ie:
    H_ie = H
    H = H + DELTA_H
B = S - H
S_ie = S - DELTA_H
H_ie = H - DELTA_H

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
    mat_z_ie[:, j] = np.where(grid[:, j], B[j]+inter*H_ie[j],
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
depth_max_mesh = np.nanmax(mat_depth, axis=0)

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

# a0 is not defined when trajectories reach the dome area, so we set to a[0].
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
# The grid for the age can be slightly different if the nb of nodes increases
grid_age = ~np.isnan(mat_steady_age)

print('After calculation of steady age matrix.',
      round(time.perf_counter()-START_TIME, 4), 's.')

# -------------------------------------------------------
# Matrix of thinning function: mat_tau
# -------------------------------------------------------

mat_tau = np.where(grid[1:, :], (mat_z_ie[:-1, :] - mat_z_ie[1:, :])
                   / (mat_steady_age[1:, :] - mat_steady_age[:-1, :])
                   / (mat_a0[:-1, :] + mat_a0[1:, :]) * 2, np.nan)

# --------------------------------------------------------------------
# Matrix of thinning function with analytical formula: mat_tau_anal
# --------------------------------------------------------------------

tau_reduc_pitheta = np.zeros((imax+1, imax+1))

# Calculation line by line, F. Parrenin, 2023/05/03
# Based on the analytical formula from Parrenin (HDR, 2013)
for i in range(1, imax+1):
    tau_reduc_pitheta[i, 1:] = tau_reduc_pitheta[i-1, :-1] + \
        dzdOMEGA[i-1, 1:] / a[1:] - dzdOMEGA[i-1, :-1] / a[:-1]
mat_tau_anal = np.ones((imax+1, imax+1))
mat_tau_anal[1:-1, :] = 1/(mat_tau_anal[1:-1, :] - tau_reduc_pitheta[1:-1, :]
                           * a / (dzdOMEGA[:-1, :] + dzdOMEGA[1:, :]) * 2)
mat_tau_anal *= a / mat_a0 * OMEGA.reshape(imax+1, 1)
mat_tau_anal[-1, :] = np.nan

# ---------------------------------
# Surface velocity
# ---------------------------------

ux_surf = Q/Y/dzdOMEGA[0, :]

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
mat_age = np.interp(mat_steady_age+age_surf, steady_age_R, age_R)

# ----------------------------------------------------------
# Post-processing: transfering of the modeling results
# on the 1D grid of the drilling sites
# ----------------------------------------------------------

print('Before calculating for the ice cores',
      round(time.perf_counter()-START_TIME, 4), 's.')

for name in ic:
    
    # ---------------------------------------------------
    # pi for the ice core
    # ---------------------------------------------------

    ic[name]['Q'] = np.interp(ic[name]['x'], x_fld, Q_fld)
    ic[name]['pi'] = math.log(ic[name]['Q']/Q_fld[-1])

    # ---------------------------------------------------
    # Depth_ic and ie_depth_ic along the drilling
    # ---------------------------------------------------

    ic[name]['depth'] = np.arange(0., ic[name]['max_depth'] + 0.0001,
                                  ic[name]['step_depth'])
    ie_depth_ic = np.interp(ic[name]['depth'], D_depth, D_depth_ie)
    ic[name]['depth_ie'] = ie_depth_ic
    
    # -------------------------------------------------------
    # Calculation of the surrounding nodes for the ice core
    # -------------------------------------------------------

    if ic[name]['x'] > x_right:
        sys.exit("The ice core is downstream of the domain.")
    elif ic[name]['x'] < x[0]:
        sys.exit("The ice core is upstream of the domain.")
    elif ic[name]['x'] == x_right:
        ggrid = grid_age[:, imax]
        ddepth_ie = mat_depth_ie[:, imax][ggrid]
        OOMEGA = mat_OMEGA[:, imax][ggrid]
        aa0 = mat_a0[:, imax][ggrid]
        ssteady_age = mat_steady_age[:, imax][ggrid]
        xx0 = mat_x0[:, imax][ggrid]
        ttau = mat_tau_anal[:, imax][ggrid]
        ttheta = theta[ggrid]
        ic[name]['S'] = S[imax]
        ic[name]['PI'] = 0.
    else:
        ii = np.argmax(x[x <= ic[name]['x']])
        inter = (pi[ii+1]-ic[name]['pi']) / delta
        ggrid = np.logical_and(grid_age[:, ii], grid_age[::, ii+1])
        ddepth_ie = inter * mat_depth_ie[:, ii][ggrid] +\
            (1-inter) * mat_depth_ie[:, ii+1][ggrid]
        OOMEGA = inter * mat_OMEGA[:, ii][ggrid] +\
            (1-inter) * mat_OMEGA[:, ii+1][ggrid]
        aa0 = inter * mat_a0[:, ii][ggrid] +\
            (1-inter) * mat_a0[:, ii+1][ggrid]
        ssteady_age = inter * mat_steady_age[:, ii][ggrid] +\
            (1-inter) * mat_steady_age[:, ii+1][ggrid]
        xx0 = inter * mat_x0[:, ii][ggrid] + (1-inter) * mat_x0[:, ii+1][ggrid]
        ttau = inter * mat_tau_anal[:, ii][ggrid] + \
            (1-inter) * mat_tau_anal[:, ii+1][ggrid]
        ttheta = theta[ggrid]
        ic[name]['S'] = inter * S[ii] + (1-inter) * S[ii+1]
        ic[name]['PI'] = pi[ii] + inter*(pi[ii+1]-pi[ii])

    # ----------------------------------------------------------
    #  Computation of theta for the ice core: theta_ic
    # ----------------------------------------------------------

    if ddepth_ie[-1] < ie_depth_ic[-1]:
        sys.exit("The mesh does not extend down to the bottom of the", name, "ice core.")

    ic[name]['theta'] = np.log(np.interp(ie_depth_ic, ddepth_ie, OOMEGA))
    ic[name]['Omega'] = np.interp(ie_depth_ic, ddepth_ie, OOMEGA)

    # ----------------------------------------------------------
    #  Computation steady a0 and x0 for the ice core
    # ----------------------------------------------------------

    # Be careful, xp must be in increasing order for np.interp
    steady_a0_ic = np.interp(-ic[name]['theta'], -ttheta, aa0)
#    steady_a0_ic = np.interp(np.exp(ic[name]['theta']), Q_fld/Q_fld[-1], a_fld)
    ic[name]['x0'] = np.interp(-ic[name]['theta'], -ttheta, xx0)

    # ----------------------------------------------------------
    #  Computation of steady_age icecore
    # ----------------------------------------------------------

    # Quadratic spline with derivative constraint at surface
    # We had a point close to the surface to impose the derivative of the age

    new_ddepth_ie = np.insert(ddepth_ie, 1, 1/1000000)
    new_ssteady_age = np.insert(ssteady_age, 1,
                      1/1000000/steady_a0_ic[0])
    steady_age_ic = interp1d(new_ddepth_ie, new_ssteady_age, assume_sorted=True,
                              kind='quadratic')(ic[name]['depth_ie'])

    # steady_age_ic = np.interp(-ic[name]['theta'], -ttheta, ssteady_age)

    # ----------------------------------------------------------
    #  Computation of age for the ice core
    # ----------------------------------------------------------

    ic[name]['age'] = np.interp(steady_age_ic+age_surf, steady_age_R, age_R)
    print('Bottom age for the', name, 'ice core:', ic[name]['age'][-1])

    # ----------------------------------------------------------
    #  a0_ic
    # ----------------------------------------------------------

    # Here, steady_a0_ic is at the node, while a0_ic is for an interval
    ic[name]['a0'] = (steady_a0_ic[1:]+steady_a0_ic[:-1])/2 *\
        interp_stair_aver(steady_age_ic, steady_age_R, R)

    # ----------------------------------------------------------
    #  Computation of tau_ic for the ice core
    # ----------------------------------------------------------

    ic[name]['tau'] = np.interp(ic[name]['depth_ie'], ddepth_ie, ttau)
    ic[name]['tau'] = (ic[name]['tau'][1:]+ic[name]['tau'][:-1])/2.
    steady_age_ic_recalc = np.cumsum((ie_depth_ic[1:]-ie_depth_ic[:-1])/ \
                                      (steady_a0_ic[1:]+steady_a0_ic[:-1])*2/ \
                                          ic[name]['tau'])
    steady_age_ic_recalc = np.insert(steady_age_ic_recalc, 0, 0.)
    ic[name]['age_int'] = np.interp(steady_age_ic_recalc+age_surf,
                                    steady_age_R, age_R)
    print('max age deviation for', name, 'between the two numerical schemes:',
          np.max(np.abs(steady_age_ic - steady_age_ic_recalc)))
    aa = (steady_a0_ic[1:]+steady_a0_ic[:-1]) / 2
    ic[name]['tau_diff'] = (ie_depth_ic[1:] - ie_depth_ic[:-1]) / aa / \
        (steady_age_ic[1:] - steady_age_ic[:-1])

    # ----------------------------------------------------------
    # Output for the ice cores
    # ----------------------------------------------------------

    if output_ic:
        output = np.vstack((ic[name]['depth'], ic[name]['age'],
                            np.append(ic[name]['tau'], np.nan),
                            np.append(ic[name]['a0'], np.nan),
                            ic[name]['x0'],
                            steady_a0_ic))
        np.savetxt(datadir+name+'_ice_core_output.txt', np.transpose(output),
                   header="depth age thinning accu x_origin accu_steady")

    # -----------------------------------------------------------
    # Reading of comparison files
    # -----------------------------------------------------------
    if ic[name]['comp'] is not None:
        ic[name]['cp_depth'], ic[name]['cp_age'], ic[name]['cp_x'],\
            ic[name]['cp_tau'] = np.loadtxt(datadir+ic[name]['comp'],
                                            unpack=True)

# ---------------------------------------
# Reading of flow line comparison file
# ---------------------------------------

if comp_flowline is not None:
    cp_fl_x, cp_fl_ux_surf = np.loadtxt(datadir+comp_flowline, unpack=True)

if comp_isochrones is not None:
    readarray = np.loadtxt(datadir+comp_isochrones, unpack=True)
    cp_iso_x = readarray[0, :]
    cp_iso_depth = readarray[1:, :]
    cp_iso_nb = cp_iso_depth.shape[0]

if obs_bedrock is not None:
    readarray = np.loadtxt(datadir+obs_bedrock, unpack=True)
    obs_bed_x = readarray[0, :]
    obs_bed_depth = readarray[1, :]
    obs_bed_depth = np.interp(x, obs_bed_x, obs_bed_depth)
    obs_bed_z = S - obs_bed_depth
    obs_bed_x = np.interp(x, obs_bed_x, obs_bed_x)

# ----------------------------------------------------------
# Output quantities along the flow line
# ----------------------------------------------------------

if output_fl:
    output = np.vstack((x, Q, a, ux_surf))
    np.savetxt(datadir+'flow_line_output.txt', np.transpose(output),
               header='x(km) total_flux accu(m/yr) surf_velocity(m/yr)')

# -----------
# FIGURES
# -----------

# Note: We don't plot refrozen ice since the mesh does not always extend to it.

if create_figs:

    print('Before creating figures.',
          round(time.perf_counter()-START_TIME, 4), 's.')

    for name in ic:

        ic[name]['XX'] = ic[name]['x'] * np.ones(2)
        ic[name]['ZZ'] = np.array([ic[name]['S'],
                                   ic[name]['S']-ic[name]['max_depth']])
        ic[name]['DD'] = np.array([0, ic[name]['max_depth']])
        ic[name]['PP'] = ic[name]['PI']*np.ones(2)
        ic[name]['TT'] = np.array([0, ic[name]['theta'][-1]])
    color_core = 'r'
    lw_core = 2
    ls_core = 'dashed'

    # ----------------------------------------------------------
    # Display of (pi,theta) mesh
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.vlines(pi, theta_min_mesh, theta_max, color='grey', linewidths=0.1)
    for i in range(0, imax+1):
        plt.plot(pi, mat_theta[i, :], color='grey', linewidth=0.1)
    plt.xlabel(r'$\pi$', fontsize=18)
    plt.ylabel(r'$\theta$', fontsize=18)
    for name in ic:
        plt.plot(ic[name]['PP'], ic[name]['TT'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['PP'][0], 0.03), ha='center',
                     va='bottom', color=color_core)
    plt.savefig(datadir+'mesh_pi_theta.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of (x, z) mesh
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(x, S, label='Surface', color='0')
    # The vertical grid step can increase near the bed.
    # This is due do iso-omega layers being thicker near the bed.
    for i in range(0, imax+1):
        plt.plot(x, mat_z[i, :],  ls='-', color='grey', linewidth=0.1)
    plt.vlines(x, z_ie_min_mesh, S, color='grey', linewidths=0.1)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_z, color='k', linewidth=2.)
        plt.plot(x, B, label='Bedrock', color='violet', linewidth=2.)
    else:        
        plt.plot(x, B, label='Bedrock', color='k', linewidth=2.)
    plt.xlabel(r'$x$ (km)', fontsize=18)
    plt.ylabel(r'$z$ (m)', fontsize=18)
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['x'], ic[name]['S']+20), ha='center',
                     va='bottom', color=color_core)
    plt.savefig(datadir+'mesh_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of (x, depth) mesh
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(x, np.zeros_like(x), label='Surface', color='0')
    # The vertical grid step can increase near the bed.
    # This is due do iso-omega layers being thicker near the bed.
    for i in range(0, imax+1):
        plt.plot(x, mat_depth[i, :],  ls='-', color='grey', linewidth=0.1)
    plt.vlines(x, depth_max_mesh, S, color='grey', linewidths=0.1)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_depth, color='k', linewidth=2.)
        plt.plot(x, S-B, label='Bedrock', color='violet', linewidth=2.)
    else:        
        plt.plot(x, S-B, label='Bedrock', color='k', linewidth=2.)
    plt.xlabel(r'$x$ (km)', fontsize=18)
    plt.ylabel(r'$depth$ (m)', fontsize=18)
    ax.invert_yaxis()
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['x'], -20), ha='center',
                     va='bottom', color=color_core)
    plt.savefig(datadir+'mesh_x_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')
    
    # -------------------------------------------------------------------------
    # Boundary conditions of the flow in x
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    fig.subplots_adjust(right=0.8)
    ax.set_xlabel('x (km)', fontsize=18)
    axic = ax.secondary_xaxis("top")
    axic.set_xticks(ticks=[ic[name]['x'] for name in ic],
                    labels=[name for name in ic])
    axic.tick_params(colors='r')

    ax.set_ylabel('Y (relative unit)')
    ax.plot(x_fld, Y_fld, color='k')
    ax.spines.right.set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    ax1 = ax.twinx()
    ax1.spines['right'].set_position(('axes', 1.))
    ax1.spines['right'].set_color('g')
    ax1.plot(x_fld, a_fld, color='g')
    ax1.set_ylabel('a (m/yr)', color='g')
    ax1.tick_params(axis='y', colors='g')

    ax2 = ax.twinx()
    ax2.spines['right'].set_position(('axes', 1.09))
    ax2.spines['right'].set_color('b')
    ax2.plot(x_fld, m_fld, color='b')
    ax2.set_ylabel('m (m/yr)', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.savefig(datadir+'boundary_conditions_x.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Calculated quantities along the flow line
    # ----------------------------------------------------------

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    fig.subplots_adjust(right=0.8)
    ax.set_xlabel('x (km)', fontsize=18)
    axic = ax.secondary_xaxis("top")
    axic.set_xticks(ticks=[ic[name]['x'] for name in ic],
                    labels=[name for name in ic])
    axic.tick_params(colors='r')

    ax.set_ylabel('Q (relative unit)')
    ax.plot(x, Q, color='k')
    ax.spines.right.set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    ax1 = ax.twinx()
    ax1.spines['right'].set_position(('axes', 1.))
    ax1.spines['right'].set_color('g')
    ax1.plot(x, a*R[0], color='g')
    ax1.set_ylabel('a (m/yr)', color='g')
    ax1.tick_params(axis='y', colors='g')

    ax2 = ax.twinx()
    ax2.spines['right'].set_position(('axes', 1.09))
    ax2.spines['right'].set_color('b')
    ax2.plot(x, ux_surf*R[0], color='b')
    if comp_flowline is not None:
        ax2.plot(cp_fl_x, cp_fl_ux_surf, color='b', linestyle='dashed')
    ax2.set_ylabel('surface velocity (m/yr)', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.savefig(datadir+'calculated_quantities_x.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of iso-omega lines in (x, z)
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(x, S, label='Surface', color='0')
    levels = np.arange(0, 1.01, 0.01)
    levels_cb = np.arange(0, 11, 1)/10.
    # There is no node on the bedrock, so the color does not go down there.
    cp = plt.contourf(mat_x, mat_z, mat_omega,
                      levels=levels,
                      cmap='plasma')
    cp2 = plt.contour(mat_x, mat_z, mat_omega,
                      levels=levels_cb,
                      colors='k', linewidths=0.5)
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    cb.set_label(r'$\omega$')
    ax.set_xlabel(r'$x$ (km)', fontsize=19)
    ax.set_ylabel(r'$z$ (m)', fontsize=19)
    bottom, top = plt.ylim()
    plt.ylim((bottom, top+50))
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['x'], ic[name]['S']+50), ha='center',
                     va='bottom', color=color_core)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_z, color='k', linewidth=2.)
        plt.plot(x, B, label='Bedrock', color='violet', linewidth=2.)
    else:
        plt.plot(x, B, label='Bedrock', color='k', linewidth=2.)
    plt.savefig(datadir+'iso-omega_lines_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of iso-omega lines in (x, depth)
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(x, np.zeros_like(x), label='Surface', color='0')
    levels = np.arange(0, 1.01, 0.01)
    levels_cb = np.arange(0, 11, 1)/10.
    # There is no node on the bedrock, so the color does not go down there.
    cp = plt.contourf(mat_x, mat_depth, mat_omega,
                      levels=levels,
                      cmap='plasma')
    cp2 = plt.contour(mat_x, mat_depth, mat_omega,
                      levels=levels_cb,
                      colors='k', linewidths=0.5)
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    cb.set_label(r'$\omega$')
    ax.set_xlabel(r'$x$ (km)', fontsize=19)
    ax.set_ylabel(r'$depth$ (m)', fontsize=19)
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['x'], -50), ha='center',
                     va='bottom', color=color_core)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_depth, color='k', linewidth=2.)
        plt.plot(x, S-B, label='Bedrock', color='violet', linewidth=2.)
    else:
        plt.plot(x, S-B, label='Bedrock', color='k', linewidth=2.)
    ax.invert_yaxis()
    bottom, top = plt.ylim()
    plt.ylim((bottom, -50))
    plt.savefig(datadir+'iso-omega_lines_x_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of age and isochrones in (x, z)
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(x, S, label='Surface', color='0')

    if comp_isochrones is not None:
        for i in range(cp_iso_nb):
            plt.plot(cp_iso_x, np.interp(cp_iso_x, x, S) - cp_iso_depth[i, :],
                     color='k', linestyle='dotted', linewidth=0.5)
    levels = np.arange(0, fig_age_max, fig_age_spacing)
    levels_cb = np.arange(0, fig_age_max, fig_age_spacing_labels)
    levels_iso = np.array(fig_age_iso)
    cp = plt.contourf(mat_x, mat_z, mat_age/1000.,
                      levels=levels,
                      cmap='plasma_r')
    cp2 = plt.contour(mat_x, mat_z, mat_age/1000.,
                      levels=levels_iso,
                      colors='k', linewidths=0.5)
    # Corner trajectory
    level0 = np.array([Q[0]])
    plt.contour(mat_x, mat_z, mat_q, colors='k', linestyles='dashed',
                levels=level0, linewidths=1)
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    ax.clabel(cp2)
    cb.set_label('Modeled age (kyr)')
    ax.set_xlabel(r'$x$ (km)', fontsize=19)
    ax.set_ylabel(r'$z$ (m)', fontsize=19)
    bottom, top = plt.ylim()
    plt.ylim((bottom, top+50))
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['x'], ic[name]['S']+50), ha='center',
                     va='bottom', color=color_core)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_z, color='k', linewidth=2.)
        plt.plot(x, B, label='Bedrock', color='violet', linewidth=2.)
    else:
        plt.plot(x, B, label='Bedrock', color='k', linewidth=2.)
    plt.savefig(datadir+'age_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of age and isochrones in (x, depth)
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(x, np.zeros_like(S), label='Surface', color='0')

    if comp_isochrones is not None:
        for i in range(cp_iso_nb):
            plt.plot(cp_iso_x, cp_iso_depth[i, :],
                     color='k', linestyle='dotted', linewidth=0.5)
    levels = np.arange(0, fig_age_max, fig_age_spacing)
    levels_cb = np.arange(0, fig_age_max, fig_age_spacing_labels)
    levels_iso = np.array(fig_age_iso)
    cp = plt.contourf(mat_x, mat_depth, mat_age/1000.,
                      levels=levels,
                      cmap='plasma_r')
    cp2 = plt.contour(mat_x, mat_depth, mat_age/1000.,
                      levels=levels_iso,
                      colors='k', linewidths=0.5)
    # Corner trajectory
    level0 = np.array([Q[0]])
    plt.contour(mat_x, mat_depth, mat_q, colors='k', linestyles='dashed',
                levels=level0, linewidths=1)
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    ax.clabel(cp2)
    cb.set_label('Modeled age (kyr)')
    ax.set_xlabel(r'$x$ (km)', fontsize=19)
    ax.set_ylabel(r'depth (m)', fontsize=19)
    ax.invert_yaxis()
    bottom, top = plt.ylim()
    plt.ylim((bottom, -50))
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['x'], -50), ha='center', va='bottom',
                     color=color_core)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_depth, color='k', linewidth=2.)
        plt.plot(x, S-B, label='Bedrock', color='violet', linewidth=2.)
    else:
        plt.plot(x, S-B, label='Bedrock', color='k', linewidth=2.)
    plt.savefig(datadir+'age_x_depth.'+fig_format,
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
                      cmap='plasma_r')
    cp2 = plt.contour(mat_pi, mat_theta, mat_age/1000.,
                      levels=levels_iso, colors='k', linewidths=0.5)
    # Corner trajectory
    level0 = np.array([Q[0]])
    plt.contour(mat_pi, mat_theta, mat_q, colors='k', linestyles='dashed',
                levels=level0, linewidths=1)
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    ax.clabel(cp2)
    cb.set_label('Modeled age (kyr)')
    ax.set_xlabel(r'$\pi$', fontsize=19)
    ax.set_ylabel(r'$\theta$', fontsize=19)
    for name in ic:
        plt.plot(ic[name]['PP'], ic[name]['TT'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['PP'][0], 0.03), ha='center',
                     va='bottom', color=color_core)
    plt.savefig(datadir+'age_pi_theta.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of thinning function
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(x, S, label='Surface', color='0')
    zz = np.insert((mat_z[1:, :] + mat_z[:-1, :])/2, 0, S, axis=0)
    tt = np.insert(mat_tau, 0, np.ones(imax+1), axis=0)
    levels = np.arange(0, 1.01, 0.01)
    levels_cb = np.arange(0, 11, 1)/10.
    cp = plt.contourf(mat_x, zz, tt,
                      levels=levels,
                      cmap='plasma')
    cp2 = plt.contour(mat_x, zz, tt,
                      levels=levels_cb,
                      colors='k', linewidths=0.5)
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
    bottom, top = plt.ylim()
    plt.ylim((bottom, top+50))
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['x'], ic[name]['S']+50), ha='center',
                     va='bottom', color=color_core)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_z, color='k', linewidth=2.)
        plt.plot(x, B, label='Bedrock', color='violet', linewidth=2.)
    else:
        plt.plot(x, B, label='Bedrock', color='k', linewidth=2.)
    plt.savefig(datadir+'thinning_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of thinning function - analytical formula - (x,z)
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(x, S, label='Surface', color='0')
    levels = np.arange(0, 1.01, 0.01)
    levels_cb = np.arange(0, 11, 1)/10.
    cp = plt.contourf(mat_x, mat_z, mat_tau_anal,
                      levels=levels,
                      cmap='plasma')
    cp2 = plt.contour(mat_x, mat_z, mat_tau_anal,
                      levels=levels_cb,
                      colors='k', linewidths=0.5)
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
    bottom, top = plt.ylim()
    plt.ylim((bottom, top+50))
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['x'], ic[name]['S']+50), ha='center',
                     va='bottom', color=color_core)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_z, color='k', linewidth=2.)
        plt.plot(x, B, label='Bedrock', color='violet', linewidth=2.)
    else:
        plt.plot(x, B, label='Bedrock', color='k', linewidth=2.)
    plt.savefig(datadir+'thinning_analytical_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of thinning function - analytical formula - (x,depth)
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(x, np.zeros_like(x), label='Surface', color='0')
    levels = np.arange(0, 1.01, 0.01)
    levels_cb = np.arange(0, 11, 1)/10.
    cp = plt.contourf(mat_x, mat_depth, mat_tau_anal,
                      levels=levels,
                      cmap='plasma')
    cp2 = plt.contour(mat_x, mat_depth, mat_tau_anal,
                      levels=levels_cb,
                      colors='k', linewidths=0.5)
    # Corner trajectory
    level0 = np.array([Q[0]])
    plt.contour(mat_x, mat_depth, mat_q, colors='k', linestyles='dashed',
                levels=level0, linewidths=1)
    cb = plt.colorbar(cp)
    cb.set_ticks(levels_cb)
    cb.set_ticklabels(levels_cb)
    cb.add_lines(cp2)
    cb.set_label('Thinning function (no unit))')
    ax.set_xlabel(r'$x$ (km)', fontsize=19)
    ax.set_ylabel(r'$depth$ (m)', fontsize=19)
    ax.invert_yaxis()
    bottom, top = plt.ylim()
    plt.ylim((bottom, -50))
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core)
        plt.annotate(name, (ic[name]['x'], -50), ha='center',
                     va='bottom', color=color_core)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_depth, color='k', linewidth=2.)
        plt.plot(x, S-B, label='Bedrock', color='violet', linewidth=2.)
    else:
        plt.plot(x, S-B, label='Bedrock', color='k', linewidth=2.)
    plt.savefig(datadir+'thinning_analytical_x_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')


    # ----------------------------------------------------------
    # Display of stream lines - (x,z)
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
#    from matplotlib import ticker
#    cp = plt.contourf(mat_x, mat_z, mat_q, levels=levels,
#                      locator=ticker.LogLocator())
    plt.plot(x, S, label='Surface', color='0')
    # Fake plots for the legend
    plt.plot(x, B, label="Trajectories", color=color, linewidth=lw)
    plt.plot(x, B, label="Corner trajectory", color='k', linewidth=1,
             linestyle='dashed')
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core, label=name)
        plt.annotate(name, (ic[name]['x'], ic[name]['S']+50), ha='center',
                     va='bottom', color=color_core)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_z, color='k', linewidth=2.)
        plt.plot(x, B, label='Bedrock', color='violet', linewidth=2.)
    else:
        plt.plot(x, B, label='Bedrock', color='k', linewidth=2.)
#    plt.legend(loc='lower left')
    plt.xlabel(r'$x$ (km)', fontsize=19)
    plt.ylabel(r'$z$ (m)', fontsize=19)
    bottom, top = plt.ylim()
    plt.ylim((bottom, top+50))
    plt.savefig(datadir+'stream_lines_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Display of stream lines - (x,depth)
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
    plt.contour(mat_x, mat_depth, mat_q, colors=color,
                levels=levels, linewidths=lw)
    # Corner trajectory
    level0 = np.array([Q[0]])
    plt.contour(mat_x, mat_depth, mat_q, colors='k', linestyles='dashed',
                levels=level0, linewidths=1)
    # Color contour plot.
#    from matplotlib import ticker
#    cp = plt.contourf(mat_x, mat_z, mat_q, levels=levels,
#                      locator=ticker.LogLocator())
    plt.plot(x, np.zeros_like(x), label='Surface', color='0')
    # Fake plots for the legend
    plt.plot(x, S-B, label="Trajectories", color=color, linewidth=lw)
    plt.plot(x, S-B, label="Corner trajectory", color='k', linewidth=1,
             linestyle='dashed')
    for name in ic:
        plt.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
                 color=color_core, linestyle=ls_core, label=name)
        plt.annotate(name, (ic[name]['x'], -50), ha='center',
                     va='bottom', color=color_core)
    if obs_bedrock is not None:
        plt.plot(obs_bed_x, obs_bed_depth, color='k', linewidth=2.)
        plt.plot(x, S-B, label='Bedrock', color='violet', linewidth=2.)
    else:
        plt.plot(x, S-B, label='Bedrock', color='k', linewidth=2.)
#    plt.legend(loc='lower left')
    plt.xlabel(r'$x$ (km)', fontsize=19)
    plt.ylabel(r'$depth$ (m)', fontsize=19)
    ax.invert_yaxis()
    bottom, top = plt.ylim()
    plt.ylim((bottom, -50))
    plt.savefig(datadir+'stream_lines_x_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ---------------------------------------------------------------------
    # R(t) - Age
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots()
    plt.stairs(R[:-2], age_R[:-1]/1000, baseline=None)
    plt.xlabel('time (kyr)', fontsize=15)
    plt.ylabel(r'$R(t)$', fontsize=15)
    plt.savefig(datadir+'R_temporal_factor.'+fig_format,
                format=fig_format, bbox_inches='tight')

    # ----------------------------------------------------------
    # Graphs vs depth for the ice core
    # ----------------------------------------------------------

    for name in ic:

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_ylabel('depth (m)')
        ax.invert_yaxis()
        ax.plot(ic[name]['x0'], ic[name]['depth'], color='r')
        if ic[name]['comp'] is not None and ~np.isnan(ic[name]['cp_x']).all():
            ax.plot(ic[name]['cp_x'], ic[name]['cp_depth'], color='r',
                    linestyle='dashed')
        ax.set_xlabel(r'$x$ origin (km)', color='r')
        ax.spines['bottom'].set_color('r')
        ax.tick_params(axis='x', colors='r')

        ax2 = ax.twiny()
        ax2.spines.bottom.set_visible(False)
        ax2.plot(ic[name]['age']/1000, ic[name]['depth'], color='b')
        ax2.plot(ic[name]['age_int']/1000, ic[name]['depth'], color='b',
                 linestyle='dotted')
        if ic[name]['comp'] is not None and \
                ~np.isnan(ic[name]['cp_age']).all():
            ax2.plot(ic[name]['cp_age'], ic[name]['cp_depth'], color='b',
                     linestyle='dashed')
        ax2.set_xlabel('age (kyr)', color='b')
        ax2.spines['top'].set_color('b')
        ax2.tick_params(axis='x', colors='b')
        if 'fig_max_age' in ic[name]:
            ax2.set_xlim(0, ic[name]['fig_max_age'])
        
        ax3 = ax.twiny()
        ax3.spines['top'].set_position(('axes', 1.1))
        ax3.spines.bottom.set_visible(False)
        ax3.plot(ic[name]['tau'], ic[name]['depth'][:-1], color='orange')
        ax3.plot(ic[name]['tau_diff'], ic[name]['depth'][:-1], color='orange',
                 linestyle='dotted')
        if ic[name]['comp'] is not None and \
                ~np.isnan(ic[name]['cp_tau']).all():
            ax3.plot(ic[name]['cp_tau'], ic[name]['cp_depth'], color='orange',
                     linestyle='dashed')
        ax3.set_xlabel('thinning function (no unit)', color='orange')
        ax3.spines['top'].set_color('orange')
        ax3.tick_params(axis='x', colors='orange')

        ax4 = ax.twiny()
        ax4.spines['top'].set_position(('axes', 1.2))
        ax4.spines.bottom.set_visible(False)
        ax4.plot(ic[name]['Omega'], ic[name]['depth'][:], color='violet')
        # if ic[name]['comp'] is not None and \
        #         ~np.isnan(ic[name]['cp_tau']).all():
        #     ax4.plot(ic[name]['cp_tau'], ic[name]['cp_depth'], color='orange',
        #              linestyle='dashed')
        ax4.set_xlabel('$\Omega$ (no unit)', color='violet')
        ax4.spines['top'].set_color('violet')
        ax4.tick_params(axis='x', colors='violet')

        plt.savefig(datadir+name+'_ice_core_vs_depth.'+fig_format,
                    format=fig_format, bbox_inches='tight')

    # ----------------------------------------------
    # Graphs vs age for the ice core
    # ----------------------------------------------

    for name in ic:

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_xlabel('age (kyr)')
        ax.set_ylabel('layer thickness (m/yr)')
        ax.stairs(ic[name]['a0'], ic[name]['age']/1000, baseline=None,
                  label='accumulation')
        ax.stairs(ic[name]['tau'] * ic[name]['a0'], ic[name]['age']/1000,
                  baseline=None, label='layer thickness')
        ax.legend()

        plt.savefig(datadir+name+'_ice_core_vs_age.'+fig_format,
                    format=fig_format, bbox_inches='tight')

    # Showing the figures
    plt.show()

# Program execution time
MESSAGE = 'Program execution time: '+str(time.perf_counter()-START_TIME)+' s.'
print(MESSAGE)
if os.name != 'nt':
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max memory usage: '+str(mem)+' kbytes')