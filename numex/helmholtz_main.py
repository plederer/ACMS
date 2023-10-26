# LIBRARIES

from ngsolve import *
from netgen.geom2d import *

import numpy 
import scipy.linalg
import scipy.sparse as sp

from netgen.occ import *
# from ngsolve.webgui import Draw
import matplotlib.pyplot as plt

from helping_functions import *
from helmholtz_aux import *


maxH=0.02

#Generate mesh: unit disco with 8 subdomains
mesh, dom_bnd = unit_disc(maxH)

# PROBLEM SETTING
# PROBLEM = 1: plane wave solution (Example 5.1, Tables 5.2-5.5), 
# exact solution available and adjustable kwave 
# PROBLEM = 2: localised interior source (Example 5.2, Table 5.6)
# no exact solution, use of bubbles
# PROBLEM = 3: periodic structure (NOT YET IMPLEMENTED -> mesh needs to change)

problem = 2



# ATTENTION: if the mesh is too coarse, we cannot have many bubbles/modes
order_v = [1] 
Bubble_modes = [1]
Edge_modes = [1] 
plot_error = 0


# Variables setting
kappa, omega, beta, f, g, sol_ex = problem_definition(problem)

# Compute ground truth solution with FEM of order 3 on the initialised mesh
# If available, the exact solution is used  (sol_ex == 1)  
grad_uex = ground_truth(mesh, dom_bnd, kappa, omega, beta, f, g, sol_ex)

# Solve ACMS system and compute H1 error
h1_error, gfu_acms = acms_solution(mesh, dom_bnd, Bubble_modes, Edge_modes, order_v, kappa, omega, beta, f, g, grad_uex)

# Plot H1 error
convergence_plots(plot_error, h1_error, mesh, Edge_modes, Bubble_modes,order_v)
