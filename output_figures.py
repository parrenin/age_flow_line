#−*−coding: utf−8−*− 

# La visualisation est plus rapide sans la sauvegarde des figures 



#------------------------------------
# Executing ice_flow_model.py 
#------------------------------------

exec(open('modele_ordre_1.py').read())



#----------------------------------------
# Figure 1: Tube width Y 
#----------------------------------------
plt.figure(1)
plt . plot (pi , Y[1:] , '-')
plt.legend([r'Tube width: Y (m)'], loc='best')
plt.xlabel(r'$\pi$', fontsize = 18)
plt.ylabel(r'$Y \ (m)$', fontsize = 18)
plt.grid()
plt.savefig('output_figures/Tube_width_theta_%d.pdf' %imax)



#---------------------------------------------------
# Figure 2: Accumulation
#---------------------------------------------------
plt.figure(2)
plt . plot (pi , a[1:] , '-')
plt.legend([r'accumulation: a (m/year)'], loc='best')
plt.xlabel(r'$\pi$', fontsize = 18)
plt.ylabel(r'$a \ (m/year)$', fontsize = 18)
plt.savefig('output_figures/Accumulation_theta_%d.pdf' %imax)
plt.grid()



#----------------------------------------
# Figure 3: Melting rate
#----------------------------------------
plt.figure(3)
plt . plot (pi , m[1:] , '-')
plt.legend([ r'melting: m (m/year)'], loc='best')
plt.xlabel(r'$\pi$',fontsize = 18)
plt.ylabel(r'$m \ (m/year)$',fontsize = 18)
plt.savefig('output_figures/Melting_theta_%d.pdf' %imax)
plt.grid()



#----------------------------------------------------------
# Visualisation maillage (pi,theta)
#----------------------------------------------------------
plt.figure(4)
plt.vlines(pi, theta_min, theta_max)
for i in range(0,imax+1):
  plt.hlines(mat_theta[i,:], mat_pi[0,:], np.zeros((imax+1,)) )
plt.xlabel(r'$\pi$',fontsize=18)
plt.ylabel(r'$\theta$',fontsize=18)
plt.savefig('output_figures/Maillage_pi_theta_%d.pdf' %imax)



#----------------------------------------------------------
# Visualisation du maillage (x,z)
#----------------------------------------------------------
z_ie_min = np.zeros((imax+2))
for i in range(imax+2):
  z_ie_min[i] = np.asarray(  np.amin( z_ie[:,i][~np.isnan(z_ie[:,i])] )   )

plt.figure(5, figsize = (15,5))
for i in range(0,imax+1):
  plt.plot(x[:], z_ie[i,:],  ls = '-', marker = '.', color='black')
plt.vlines(x, z_ie_min, S_ie)
plt.xlabel(r'$X$',fontsize=18)
plt.ylabel(r'$Z$',fontsize=18)
plt.savefig('output_figures/Maillage_x_z_%d.pdf' %imax)



#-------------------------------------------------------------------------
# Visualisation des paramètres de l'écoulement sur le maillage (pi,theta)
#-------------------------------------------------------------------------
fig, ax = plt.subplots()
fig.set_size_inches(15,5)

# Twin the x-axis twice to make independent y-axes.
axes = [ ax.twinx(), ax.twinx(), ax.twinx()]

# Make some space on the right side for the extra y-axis.
fig.subplots_adjust(right=0.8)

# Move the last y-axis spine over to the right by 20% of the width of the axes
axes[0].spines['right'].set_position(('axes', 1.01))
axes[0].spines['right'].set_color('Green')
axes[1].spines['right'].set_position(('axes', 1.08))
axes[1].spines['right'].set_color('Red')
axes[2].spines['right'].set_position(('axes', 1.2))
axes[2].spines['right'].set_color('Blue')


# To make the border of the right-most axis visible, we need to turn the frame
# on. This hides the other plots, however, so we need to turn its fill off.
axes[0].set_frame_on(True)
axes[0].patch.set_visible(False)
axes[1].set_frame_on(True)
axes[1].patch.set_visible(False)
axes[2].set_frame_on(True)
axes[2].patch.set_visible(False)


# And finally we get to plot ...
colors = ('Green', 'Red', 'Blue')

datas = [ Y[1:], a[1:], m[1:] ]

ynames = [ 'Y (m)' , 'a (m/year)' , 'm (m/year)' ]


ax.vlines(pi, theta_min, theta_max, linewidth = 0.2)
for i in range(0,imax+1):
  ax.hlines(mat_theta[i,:], mat_pi[0,:], np.zeros((imax+1,)), linewidth = 0.2 )
ax.set_xlim(np.amin(theta_min), np.amax(theta_max))
ax.set_xlabel(r'$\pi$',fontsize=18)
ax.set_ylabel(r'$\theta$',fontsize=18)
for ax, color, data, yname in zip(axes, colors, datas, ynames):
    ax.plot(pi, data, color=color)
    ax.set_ylabel(  yname , color=color)
    ax.tick_params(axis='y', colors=color)
plt.savefig('output_figures/flow_parameters_pi_theta_%d.pdf' %imax)



#-------------------------------------------------------------------------
# Visualisation des paramètres de l'écoulement sur le maillage (x,z)
#-------------------------------------------------------------------------
fig, ax = plt.subplots()
fig.set_size_inches(15,5)
fig.subplots_adjust(right=0.8)

datas = [ Y, a, m ]
axes = [ ax.twinx(), ax.twinx(), ax.twinx()]

# Move the last y-axis spine over to the right by 20% of the width of the axes
axes[0].spines['right'].set_position(('axes', 1.01))
axes[0].spines['right'].set_color('Green')
axes[1].spines['right'].set_position(('axes', 1.08))
axes[1].spines['right'].set_color('Red')
axes[2].spines['right'].set_position(('axes', 1.2))
axes[2].spines['right'].set_color('Blue')


# To make the border of the right-most axis visible, we need to turn the frame
# on. This hides the other plots, however, so we need to turn its fill off.
axes[0].set_frame_on(True)
axes[0].patch.set_visible(False)
axes[1].set_frame_on(True)
axes[1].patch.set_visible(False)
axes[2].set_frame_on(True)
axes[2].patch.set_visible(False)

for i in range(0,imax+1):
  ax.plot(x[:], z_ie[i,:],  ls = '-', marker = '.', color='black')
ax.vlines(x, z_ie_min, S_ie)
ax.set_xlabel(r'$X$',fontsize=18)
ax.set_ylabel(r'$Z$',fontsize=18)

for ax, color, data, yname in zip(axes, colors, datas, ynames):
    ax.plot(x, data, color=color)
    ax.set_ylabel(  yname , color=color)
    ax.tick_params(axis='y', colors=color)
plt.savefig('output_figures/flow_parameters_x_z_%d.pdf' %imax)



#----------------------------------------------------------
# Visualisation des lignes isochrones en (x,z)
#----------------------------------------------------------
fig = plt.figure(8, figsize = (12,6))
ax  = fig.add_subplot(111)

ax.plot ( x, S, label = 'Surface', color = '0')

for i in range(len(Age_iso)) :
    ax.plot ( mat_x_iso[i,:] , mat_z_iso[i,:], label = str(Age_iso[i]) + ' year', color = plt.cm.cool(20*i) )
ax.set_xlabel(r'$x$',fontsize = 19)
ax.set_ylabel(r'$z$',fontsize = 19)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.plot ( x, B, label = 'Bedrock', color = '0')

ax.grid()

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('output_figures/lignes_isochrones_x_z_%d.pdf' %imax)



#---------------------------------------------------------------------
# Lignes isochrones en (pi,theta)
#---------------------------------------------------------------------
fig = plt.figure(9, figsize = (12,6))
ax  = fig.add_subplot(111)
for i in range(len(Age_iso)) :
    ax.plot ( pi , mat_theta_iso[i,1:], marker='.', label = str(Age_iso[i]) + ' year', color = plt.cm.cool(30*i) )
ax.set_xlabel(r'$\pi$',fontsize = 19)
ax.set_ylabel(r'$\theta$',fontsize = 19)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('output_figures/lignes_isochrones_pi_theta_%d.pdf' %imax)



#---------------------------------------------------------------------
# Age-Profondeur dans le forage
#---------------------------------------------------------------------
plt.figure(10)
plt.plot( Age, depth_corrected,'-')
plt.gca().invert_yaxis()
plt.xlabel(r'$age\ (yr\ b \ 1997)$', fontsize = 15)
plt.ylabel(r'$depth \ (m)$', fontsize = 15)
plt.savefig('output_figures/Age_depth_%d.pdf' %imax)



#---------------------------------------------------------------------
# R(t) - Age 
#---------------------------------------------------------------------
plt.figure(11)
plt.plot( Age, R_t,'-')
plt.xlabel(r'$time \ (yr\ b\ 1997 )$', fontsize = 15 )
plt.ylabel(r'$R(t)$', fontsize = 15)
plt.savefig('output_figures/R_t_age_%d.pdf' %imax)



#---------------------------------------------------------------------
# Fonction d'amincissement
#---------------------------------------------------------------------
plt.figure(12, figsize=(12,8))

plt.plot( tau_ie_middle, depth_corrected[:-1],ls='-', \
label = r'$ordre\ 1 - lin\'eaire$')
plt.plot( tau_ie_middle_sp, depth_corrected[:-1],ls='-',\
label = r'$ordre\ 1 - spline\ cubique$')
plt.plot( tau_ie_middle_sp_2, depth_corrected[:-1],ls='-',\
label = r'$ordre\ 1 - spline\ cubique - d\'eriv\'ee\ impos\'ee$')
plt.ylim([max(depth_corrected),-200.])
plt.xlabel( r'$\tau_{ie} \ (yr \ b1997) $', fontsize = 18)
plt.ylabel(r'$depth \ (m)$', fontsize = 18)
plt.legend(loc = 'upper left')
plt.grid()



#----------------------------------------------------------
# Visualisation des lignes de courant
#----------------------------------------------------------
plt.figure(13, figsize = (15,7))

plt.plot ( x, S, label = 'Surface', color = '0')

for i in range(2,imax+1): # add a step to plot less trajecories
  plt.plot(x[i:,], np.diagonal(mat_z,i), color='blue')
  plt.plot(x[2:-i+1,], np.diagonal(mat_z,-i+2)[2:], color='blue')
plt.plot(x[2:,], np.diagonal(mat_z,1)[1:], color='blue')

plt.vlines(x[0], B[0], S[0], color= 'blue') # ice divide
plt.vlines(x[1], B[1], S[1], color= 'blue') # vertical flow

plt.plot(0,0, label = "Trajectoires", color = 'blue')

plt.plot ( x, B, label = 'Socle', color = '0')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim([-5, x[imax+1] + 5])
plt.xlabel(r'$X$',fontsize = 19)
plt.ylabel(r'$Z$',fontsize = 19)
plt.grid()



#----------------------------------------------------------
# Ice Origin
#----------------------------------------------------------
plt.figure(14)
plt.plot(mat_x0[:,imax+1], mat_depth[:,imax+1], ls='-', marker='.')
plt.gca().invert_yaxis()
plt.xlabel(r'$ICE\ ORIGIN \ (km)$', fontsize = 15)
plt.ylabel(r'$DEPTH \ (m)$', fontsize = 15)
plt.savefig('output_figures/ice_origin_%d.pdf' %imax)


plt.show()




