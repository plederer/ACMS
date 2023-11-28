# LIBRARIES
from helmholtz_aux import *

# PROBLEM SETTING
# PROBLEM = 1: plane wave solution (Example 5.1, Tables 5.2-5.5), 
# exact solution available and adjustable kwave 
# PROBLEM = 2: localised interior source (Example 5.2, Table 5.6)
# no exact solution, use of bubbles
# PROBLEM = 3: localised boundary source (Example 5.3, Table 5.7)
# no exact solution, 
# periodic structure (NOT YET IMPLEMENTED -> mesh needs to change)

problem = 1 #float(input("Choose the problem. \n 1 = plane wave. \n 2 = local interior source. \n 3 = localised boundary source. \n Problem =  "))

# ATTENTION: if the mesh is too coarse, we cannot have many bubbles/modes
maxH = 0.05 #float(input("maxH: "))

order_v = [1,2,3] #list(map(int, input("Order of approximation. Vector = ").split())) # Vector [1, 2, 3]
print("Order of approximation is ", order_v)

Bubble_modes = [1] #list(map(int, input("Number of bubble modes. Vector = ").split())) # Vector [2,4,8,16,32,64,128]
print("Number of bubble modes is ", Bubble_modes)

Edge_modes = [8] #list(map(int, input("Number of edge modes. Vector = ").split())) # Vector [2,4,8,16,32,64,128]
print("Number of edge modes is ", Edge_modes)


# Generates the mesh 
# Creates variables associated with the problem
# Computes a ground truth solution with FEM of order 3 on the generated mesh
# Computes ACMS solution and saves the error 
#       both with the ground truth solution and with the exact solution, if available
# Saves the error on file named "file_name.npy" and plots it if specified (now always 0)


# Errors_FEM, Errors_exact = main(maxH, problem, order_v, Bubble_modes, Edge_modes) 


plt.rcParams.update({'font.size':12})

for h in maxH/(2**np.arange(0,3,1)):
    print(h)
    Errors_FEM, Errors_exact = main(h, problem, order_v, Bubble_modes, Edge_modes) 
    
    l2rel_error = np.reshape(Errors_exact['L2_Relative_error'], (len(Bubble_modes), len(order_v)*len(Edge_modes)))
    System_dofs = Errors_exact['nDoFs']
    print(l2rel_error[0])
    print(System_dofs)
    
    plt.loglog(System_dofs, l2rel_error[0], label=('H=%.2f, $I_e$=%i' %(h, Edge_modes[0])), marker = 'o') 
    
plt.legend()
plt.title('$L^2$ relative errors: p= %i,...,%i' %(order_v[0],order_v[-1]))
plt.xlabel('System size / dofs')  
plt.yticks([7*10**(-4), 6*10**(-4), 5*10**(-4), 4*10**(-4)], ['$7 \cdot 10^{-4}$','$6 \cdot 10^{-4}$','$5 \cdot 10^{-4}$','$4 \cdot 10^{-4}$'])
plt.xticks([400,1000,5000,22000], [400,1000,5000,22000])
plt.show()





################################################################
################################################################


# from decimal import MAX_EMAX
# from ngsolve import *
# from netgen.geom2d import *

# import numpy 
# import scipy.linalg
# import scipy.sparse as sp

# from netgen.occ import *
# # from ngsolve.webgui import Draw
# import matplotlib.pyplot as plt

# from datetime import datetime
# from pathlib import Path

# from helping_functions import *

# #Generate mesh: unit disco with 8 subdomains
# mesh, dom_bnd = unit_disc(maxH)


# plot_error = 0

# # Variables setting
# kappa, omega, beta, f, g, sol_ex, u_ex = problem_definition(problem)

# # Compute ground truth solution with FEM of order 3 on the initialised mesh
# # If available, the exact solution is used  (sol_ex == 1)  
# grad_uex = ground_truth(mesh, dom_bnd, kappa, omega, beta, f, g)

# # Solve ACMS system and compute H1 error
# h1_error, gfu_acms, h1_error_ex = acms_solution(problem, mesh, dom_bnd, Bubble_modes, Edge_modes, order_v, kappa, omega, beta, f, g, grad_uex, sol_ex, u_ex)

# # Save error on file named "file_name.npy" 
# # It needs to be loaded to be readable
# print("Error with FEM solution as ground truth")
# H1_error, file_name = save_error_file(problem, h1_error, order_v, Bubble_modes, Edge_modes, 0)
# if sol_ex == 1:
#     print("Error with exact solution")
#     H1_error_ex, file_name = save_error_file(problem, h1_error_ex, order_v, Bubble_modes, Edge_modes, 1)

# # Plot H1 error
# convergence_plots(plot_error, h1_error, mesh, Edge_modes, Bubble_modes, order_v)
