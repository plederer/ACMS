# LIBRARIES
from helmholtz_aux import *
# import netgen.gui

# ngsglobals.msg_level = 1

# PROBLEM SETTING
# PROBLEM = 1: plane wave solution (Example 5.1, Tables 5.2-5.5), exact solution available and adjustable kwave 
# PROBLEM = 2: localised interior source (Example 5.2, Table 5.6), no exact solution, use of bubbles
# PROBLEM = 3: localised boundary source (Example 5.3, Table 5.7), no exact solution, periodic structure (NOT YET IMPLEMENTED -> mesh needs to change)

# problem = float(input("Choose the problem. \n 1 = plane wave. \n 2 = local interior source. \n 3 = localised boundary source. \n Problem =  "))
# omega = float(input("Wavenumber k: "))
# # ATTENTION: if the mesh is too coarse, we cannot have many bubbles/modes
# maxH = float(input("maxH: "))
# Href = int(input("Number of mesh refinements refH (0 is no refinements): "))
# order_v = list(map(int, input("Order of approximation. Vector = ").split())) # Vector [1, 2, 3]
# print("Order of approximation is ", order_v)
# Bubble_modes = list(map(int, input("Number of bubble modes. Vector = ").split())) # Vector [2,4,8,16,32,64,128]
# print("Number of bubble modes is ", Bubble_modes)
# Edge_modes = list(map(int, input("Number of edge modes. Vector = ").split())) # Vector [2,4,8,16,32,64,128]
# print("Number of edge modes is ", Edge_modes)

# For testing
problem = 5
ACMS_flag = 0
omega = 0.484/10
Href = 0
maxH = 0.05 #025 # * 4
order_v = [1]
Bubble_modes = [0]
Edge_modes = [4]


error_table = 0
table_content_aux = ""
table_header = ""
table_end = ""

# SetNumThreads(12)
with TaskManager():
    for h in maxH/(2**np.arange(0, Href + 1 , 1)):
        print(h)
        # Variables setting
        mesh, dom_bnd, alpha, kappa, beta, f, g, sol_ex, u_ex, Du_ex, mesh_info = problem_definition(problem, maxH, omega)

        # Solve ACMS system and compute errors
        ndofs, dofs, errors_dictionary, solution_dictionary = acms_solution(mesh, dom_bnd, alpha, Bubble_modes, Edge_modes, order_v, kappa, omega, beta, f, g, u_ex, Du_ex, mesh_info)    
        
        gfu_acms = solution_dictionary["gfu_acms"]
        # gfu_fem = solution_dictionary["gfu_fem"]
        # grad_fem = solution_dictionary["grad_fem"]
        
        Draw(gfu_acms, mesh, "uacms")
        # Draw(gfu_fem, mesh, "ufem")
        
        
        # if error_table == 1:
        #     file_name, Errors = error_table_save(maxH, problem, order_v, Bubble_modes, Edge_modes, mesh, kappa, errors_dictionary, ndofs, dofs, u_ex, sol_ex, gfu_fem, grad_fem)
        #     file_path = f"./Results/" + file_name + ".npz"
        #     table_header, table_content, table_end = process_file(file_path, ACMS_flag)
        #     table_content_aux += table_content + "\\\\\n"
        
print(table_header + table_content_aux + table_end)    

