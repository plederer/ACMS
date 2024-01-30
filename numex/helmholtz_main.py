# LIBRARIES
from helmholtz_aux import *
# import netgen.gui

# PROBLEM SETTING
# PROBLEM = 1: plane wave solution (Example 5.1, Tables 5.2-5.5), 
# exact solution available and adjustable kwave 
# PROBLEM = 2: localised interior source (Example 5.2, Table 5.6)
# no exact solution, use of bubbles
# PROBLEM = 3: localised boundary source (Example 5.3, Table 5.7)
# no exact solution, 
# periodic structure (NOT YET IMPLEMENTED -> mesh needs to change)

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

# for testing
problem = 4
ACMS_flag = 0
omega = 1
Href = 3
maxH = 0.025
order_v = [1,2,3]
Bubble_modes = [1]
Edge_modes = [1,2,4,8]


# Generates the mesh 
# Creates variables associated with the problem
# Computes a ground truth solution with FEM of order 3 on the generated mesh
# Computes ACMS solution and saves the error 
#       both with the ground truth solution and with the exact solution, if available
# Saves the error on file named "file_name.npy" and plots it if specified (now always 0)


table_content_aux = ""

for h in maxH/(2**np.arange(0, Href + 1 , 1)):
    print(h)
    file_name, Errors = main(h, problem, omega, order_v, Bubble_modes, Edge_modes) 
    
    file_path = f"./Results/" + file_name + ".npz"
    table_header, table_content, table_end = process_file(file_path, ACMS_flag)
    table_content_aux += table_content + "\\\\\n"
    
print(table_header + table_content_aux + table_end)    




# plt.rcParams.update({'font.size':12})
#     Bubble_modes_aux = Errors['Dictionary'][()]['bubbles'][1]
#     print(Bubble_modes_aux)
#     Edge_modes_aux = Errors['Dictionary'][()]['edges'][1]
#     print(Edge_modes_aux)
#     l2rel_error = np.reshape(Errors['L2_Relative_error'], (len(Bubble_modes_aux), len(order_v)*len(Edge_modes_aux)))
#     System_dofs = Errors['nDoFs']
#     plt.loglog(System_dofs, l2rel_error[0], label=('H=%.2f, $I_e$=%i' %(h, Edge_modes_aux[0])), marker = 'o') 

# plt.legend()
# plt.title('$L^2$ relative errors: p= %i,...,%i' %(order_v[0],order_v[-1]))
# plt.xlabel('System size / dofs')  
# plt.xticks(System_dofs, System_dofs)
# plt.savefig('./'+str(Errors["FileName"])+".png")
# #The figure is saved with the same name as the last created file (namely with the finest meshsize)
# plt.show()
