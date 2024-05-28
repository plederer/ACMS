# LIBRARIES
from helmholtz_aux import *
# import netgen.gui

# ngsglobals.msg_level = 1

# PROBLEM SETTING
# PROBLEM = 1: plane wave solution (Example 5.1, Tables 5.2-5.5), exact solution available and adjustable kwave 
# PROBLEM = 2: localised interior source (Example 5.2, Table 5.6), no exact solution, use of bubbles
# PROBLEM = 3: localised boundary source (Example 5.3, Table 5.7), no exact solution, periodic structure (NOT YET IMPLEMENTED -> mesh needs to change)

problem = float(input("Choose the problem. \n 1 = PW. \n 2 = LIS. \n 3 = LBS. \n 4 = Crystal Sq. \n 5 = Crystal \n Problem =  "))
omega = float(input("Wavenumber k: "))
maxH = float(input("maxH: "))
Href = int(input("Number of mesh refinements refH (0 is no refinements): "))
order_v = list(map(int, input("Order of approximation. Vector = ").split())) # Vector [1, 2, 3]
print("Order of approximation is ", order_v)
# Bubble_modes = list(map(int, input("Number of bubble modes. Vector = ").split())) # Vector [2,4,8,16,32,64,128]
# print("Number of bubble modes is ", Bubble_modes)
Edge_modes = list(map(int, input("Number of edge modes. Vector = ").split())) # Vector [2,4,8,16,32,64,128]
print("Number of edge modes is ", Edge_modes)
if problem == 5:
    incl = int(input("Number of inclusions in one direction per cell incl (Power of 2): "))
else:
    incl = 0


# For testing
# problem = 5
# incl = 1
# omega = 1 #0.484/10
# Href = 0
# maxH = 0.2
# order_v = [1]
# Bubble_modes = [0]
# Edge_modes = [1]


Bubble_modes = [0]
ACMS_flag = 0   #1 = exact sol 0 = fem error

error_table = 1
table_content_aux = ""
table_header = ""
table_end = ""


SetNumThreads(12)
with TaskManager():
    for h in maxH/(2**np.arange(0, Href + 1 , 1)):
        print(h)
        # Variables setting
        mesh, dom_bnd, mesh_info, variables_dictionary = problem_definition(problem, incl, h, omega, Bubble_modes, Edge_modes, order_v)
        
        #FEM solution with same order of approximation
        solution_dictionary = ground_truth(mesh, dom_bnd, variables_dictionary, 10)
        
        # Solve ACMS system and compute errors
        variables_dictionary, solution_dictionary, errors_dictionary = acms_main(mesh, mesh_info, dom_bnd, variables_dictionary, solution_dictionary)    
        
        
        if error_table == 1:
            file_name, Errors = error_table_save(maxH, mesh, variables_dictionary, solution_dictionary, errors_dictionary)
            file_path = f"./Results/" + file_name + ".npz"
            table_header, table_content, table_end = process_file(file_path, ACMS_flag)
            table_content_aux += table_content + "\\\\\n"
        
        
print(table_header + table_content_aux + table_end)    

