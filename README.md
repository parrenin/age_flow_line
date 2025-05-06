# age_flow_line
A numerical age model along a flow line of an ice sheet

# What this manual is and is not?

This manual is a documentation on how to use the age_flow_line software.  
It is _not_ a description of the age_flow_line principles and assumptions. Please read to the scientific articles
describing age_flow_line for that purpose:\
Parrenin, F., Bazin, L., Capron, E., Landais, A., Lemieux-Dudon, B. and Masson-Delmotte, V.:
IceChrono1: a probabilistic model to compute a common and optimal chronology for several ice cores,
_Geosci. Model Dev._, 8(5), 1473–1492, doi:10.5194/gmd-8-1473-2015, 2015.  
It is _not_ an operating system or python documentation.
Please use your operating system or python documentation instead.

# Where can I get help on age_flow_line?

You can also directly email to Frédéric Parrenin: frederic.parrenin@univ-grenoble-alpes.fr

# How to download age_flow_line?

Go here:  
https://github.com/parrenin/age_flow_line/  
and click on the donwload button.  
In the downloaded folder, you will find the following files:
- README.md		: is the current documentation of age_flow_line.
- LICENCE		: is the age_flow_line licence file.
- age_flow_line.py		: is the main program that you will run.
- Clean.py		: is a python script to clean a model experiment directory
- DC-BELDC		: is an example experiment directory: it contains all the necessary
numerical settings and input files for the flow line between Dome C and Little Dome C.

# What do I need to run age_flow_line?

age_flow_line is a scientific python3 software, therefore you need a scipy distribution.  
age_flow_line is developed and tested using the anaconda distribution, therefore we recommend it.  
Anaconda can be downloaded here (use the python3 version):  
https://www.anaconda.com/download

Paleochrono probably works on other scipy distributions, provided they contain the following python
modules:  
- sys
- os
- time
- math
- numpy
- matplotlib
- scipy
- resource
- yaml

# How to run age_flow_line?

Assuming you use anaconda, you can go in the spyder shell and type the following commands in the
ipython interpreter:

```
cd path-to-age_flow_line
run age_flow_line.py exp_directory/
```

where `path-to-age_flow_line` is the directory containing age_flow_line and `exp_directory` is the name of
your experiment directory. 
The `DC-LDC` experiment directory is provided for you convenience.
It takes a few seconds to run on a recent computer.

# What is the structure of an experiment direcotry?

You can have a look at the provided `DC-LDC` directory.

You have one parameter file `parameters.yml` which contains general parameters for the
experiment.

Then you have several input file in `.txt` format:
- `accumulation.txt`: the surface accumulation along the flow line
- `melting.txt`: the basal melting along the flow line
- `thickness.txt`: the thickness along the flow line
- `surface.txt`: the surface elevation along the flow line
- `tube_width.txt`: the flow tube width along the flow line
- `sliding.txt`: the basal sliding along the flow line
- `p_Lliboutry.txt`: the p exponent of the Lliboutry velocity profile along the flow line
- `relative_density.txt`: a depth vs relative density profile
- `temporal_factor.txt`: the accumulation/melting relative temporal variations

If you want to set up a new flow tube experiment, we suggest to copy an existing experiment directory such as DC-BELDC.
Then you can incrementally modify the `parameters.yml` parameter file and the `.txt` data files.

# What are the outputs of a run?

If the run went correctly, it has created output files for the whole flow line:
- `flow_line_output.txt`: is the output for the whole flow line
- `age_pi_theta.pdf`: is the figure for the age field in the (pi,theta) coordinate system
- `age_x_z.pdf`: is the same using the (x,z) coordinate system
- `age_x_depth.pdf`: is the same using the (x,depth) coordinate system
- `boundary_conditions_x.pdf`: is the figure with the boundary conditions as a function of x
- `calculated_quantities_x.pdf`: is the figure with some 1D quantities along the flow line
- `iso-omega_lines_x_z.pdf`: are the iso-omega lines in (x,z)
- `iso-omega_lines_x_depth.pdf`: same using the depth
- `mesh_pi_theta.pdf`: is the mesh in (pi,theta)
- `mesh_x_z.pdf`: is the mesh in (x,z)
- `mesh_x_depth.pdf`: is the mesh in (x,depth)
- `R_temporal_factor.pdf`: is the figure with the accu/melting temporal factor
- `stream_lines_x_z.pdf`: are the stream lines / trajectories in (x,z)
- `stream_lines_x_depth.pdf`: same using (x,depth)
- `thinning_analytical_x_z.pdf`: is the thinning function calculated analytically in (x,z)
- `thinning_analytical_x_depth.pdf`: same in (x,depth)
- `thinning_x_z.pdf`: is the thinning function calculated by finite difference in (x,z) coordinate system

If you have defined virtual ice cores, it has also created some output files:
- `IC_ice_core_output.txt`: is the output for the _IC_ ice core
- `IC_ice_core_vs_depth.pdf`: figure with a few quantities as a function of the depth in the _IC_ ice core
- `IC_ice_core_vs_age.pdf`: figure with a few quantities as a function of the age in the _IC_ ice core

# What to do if something goes wrong?

Some errors can be eliminated by restarting the kernel in spyder (under "Console">"Restart kernel").\
If the problem persist, please post an email to the author or on the mailing list with the error message appearing on the command line.
