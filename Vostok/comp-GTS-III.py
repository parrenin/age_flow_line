import numpy as np
import matplotlib.pyplot as plt

datadir = './'
fig_format = 'pdf'

depth_ic, age_ic, tau_ic, a0_ic, x0_ic, steady_a0_ic =\
    np.loadtxt('ice_core_output.txt', unpack=True)

GTS_depth, GTS_age, GTS_x = np.loadtxt('GTS-III.txt', unpack=True)
GTS_x = 370 - GTS_x

# ----------------------------------------------------------
# Graphs vs depth for the ice core
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_ylabel('depth (m)')
ax.invert_yaxis()
ax.plot(x0_ic, depth_ic, color='r')
ax.plot(GTS_x, GTS_depth, color='r', linestyle='dashed')
ax.set_xlabel(r'$x$ origin (km)', color='r')
ax.spines['bottom'].set_color('r')
ax.tick_params(axis='x', colors='r')

ax2 = ax.twiny()
ax2.spines.bottom.set_visible(False)
ax2.plot(age_ic/1000, depth_ic, color='b')
ax2.plot(GTS_age, GTS_depth, color='b', linestyle='dashed')
ax2.set_xlabel('age (kyr)', color='b')
ax2.spines['top'].set_color('b')
ax2.tick_params(axis='x', colors='b')

ax3 = ax.twiny()
ax3.spines['top'].set_position(('axes', 1.1))
ax3.spines.bottom.set_visible(False)
ax3.plot(tau_ic, depth_ic, color='g')
ax3.set_xlabel('thinning function (no unit)', color='g')
ax3.spines['top'].set_color('g')
ax3.tick_params(axis='x', colors='g')

plt.savefig(datadir+'comp-GTS-III.'+fig_format,
            format=fig_format, bbox_inches='tight')

# ----------------------------------------------------------
# Graphs vs depth for the ice core
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_ylabel('depth (m)')
ax.invert_yaxis()
ax.plot(x0_ic-np.interp(depth_ic, GTS_depth, GTS_x), depth_ic, color='r')
ax.set_xlabel(r'$x$ origin (km)', color='r')
ax.spines['bottom'].set_color('r')
ax.tick_params(axis='x', colors='r')

ax2 = ax.twiny()
ax2.spines.bottom.set_visible(False)
ax2.plot(age_ic/1000 - np.interp(depth_ic, GTS_depth, GTS_age),
         depth_ic, color='b')
ax2.set_xlabel('age (kyr)', color='b')
ax2.spines['top'].set_color('b')
ax2.tick_params(axis='x', colors='b')

plt.savefig(datadir+'comp-GTS-III-diff.'+fig_format,
            format=fig_format, bbox_inches='tight')

plt.show()
