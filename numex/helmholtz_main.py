# LIBRARIES
from helmholtz_aux import *
# import netgen.gui
# from ngsolve.eigenvalues import PINVIT

problem = float(input("Choose the problem. \n 1 = PW. \n 2 = LIS. \n 3 = LBS. \n 4 = Crystal Sq. \n 5 = Crystal \n Problem =  "))
# omega = float(input("Wavenumber k: "))
omega_v = list(map(float, input("Wavenumber k vector = ").split()))
maxH = float(input("maxH: "))
Href = int(input("Number of mesh refinements refH (0 is no refinements): "))
order_v = list(map(int, input("Order of approximation. Vector = ").split())) 
# Bubble_modes = list(map(int, input("Number of bubble modes. Vector = ").split()))

edge_sweep = int(input("Do you want to sweep over the number of edge modes = 1y/0n "))
if edge_sweep:
    min_edge_modes = int(input("Min number of edge modes = "))
    max_edge_modes = int(input("Max number of edge modes = "))
    Edge_modes = np.arange(min_edge_modes, max_edge_modes+1, 1).tolist()
else:
    Edge_modes = list(map(int, input("Number of edge modes. Vector = ").split()))


if problem == 5:
    Ncell = int(input("Number of cells in one direction: "))
    incl = int(input("Number of inclusions in one direction per cell incl (Power of 2): "))
    ACMS_flag = 0 #FEM vs ACMS error
elif problem == 1:
    Ncell = 0
    incl = 0
    ACMS_flag = int(input("Error against exact solution = 1 or FEM solution = 0. "))
else:
    Ncell = 0
    incl = 0
    ACMS_flag = 0

# # FOR TESTING
# problem = 1
# Ncell = 1
# incl = 1
# omega_v = [1,2]#np.arange(1,4,1)
# Href = 0
# maxH = 0.2
# order_v = [2]
# Bubble_modes = [0]
# Edge_modes = [1,2,3] #8,16, 32, 64]
# # # Edge_modes = np.arange(1,2+1,1).tolist()
# # # print(Edge_modes)
# ACMS_flag = 0

Bubble_modes = [0]

error_table = 1
table_content_l2_aux = ""
table_content_h1_aux = ""
table_header = ""
table_separation = ""
table_end = ""

relerr = []

SetNumThreads(12)
with TaskManager():
    for omega in omega_v:
        for h in maxH/(2**np.arange(0, Href + 1 , 1)):
            print(h)
            # Variables setting
            mesh, variables_dictionary = problem_definition(problem, Ncell, incl, h, omega, Bubble_modes, Edge_modes, order_v, load_mesh = True)

            #FEM solution with same order of approximation
            solution_dictionary = ground_truth(mesh, variables_dictionary, 10)
            
            # Solve ACMS system and compute errors
            variables_dictionary, solution_dictionary, errors_dictionary = acms_main(mesh, variables_dictionary, solution_dictionary)
            print("l2 error", errors_dictionary["l2_error"])
            
            if error_table == 1:
                file_name = create_error_file(variables_dictionary)
                Errors = save_error_file(file_name, mesh, variables_dictionary, solution_dictionary, errors_dictionary)
                file_path = f"./Results/" + file_name + ".npz"
                table_header, table_content_l2, table_separation, table_content_h1, table_end = process_file(file_path, ACMS_flag)
                table_content_l2_aux += table_content_l2 + "\\\\\n"
                table_content_h1_aux += table_content_h1 + "\\\\\n"
            
            
            if ACMS_flag == 1:
                u_ex = variables_dictionary["u_ex"]
                l2_error = errors_dictionary["l2_error_ex"]
                l2_norm = Integrate ( InnerProduct(u_ex, u_ex), mesh, order = 10)
                print("exact", l2_norm)
            else: 
                gfu_gt = solution_dictionary["gfu_fem"]
                l2_error = errors_dictionary["l2_error"]
                l2_norm = Integrate ( InnerProduct(gfu_gt, gfu_gt), mesh, order = 10)
                print("gt", l2_norm)
                  
            dim = variables_dictionary["dim"]
            l2_norm = sqrt(l2_norm.real)
            l2_error_rel = np.dot(l2_error, 1/l2_norm)   
            relerr.append(l2_error_rel)
            print("Rel err", relerr)
            
print(table_header + table_content_l2_aux + table_separation + table_content_h1_aux + table_end)    

# relerr_reshaped = np.reshape(relerr, (np.size(omega_v), np.size(Edge_modes)))
# print(relerr_reshaped)

folder = "omega_sweep_l2_errors"

if not os.path.exists(folder):
    os.mkdir(folder)

dirname = os.path.dirname(__file__)

pickle_name = ""
if problem == 1:
    pickle_name += "circle_"
else:
    pickle_name += "crystal_Ncells_" + str(int(Ncell))  + "_incl_" + str(int(incl))  + "_"

pickle_name += "omega_"
for om in omega_v:
    pickle_name += str(om) + "_"

pickle_name += "maxH_" + str(maxH) + "_"
pickle_name += "order_" + str(int(order_v[0]))+ "_"
pickle_name += "EM_"+ str(int(Edge_modes[0]))+ "_" + str(int(Edge_modes[-1]))+ "_"
            
save_file = os.path.join(dirname, folder + "/" + pickle_name + ".out")
picklefile = open(save_file, "wb")

data = [omega_v, maxH, order_v, Edge_modes, relerr]
pickle.dump(data, picklefile)
picklefile.close()


from print_errors import SaveTable

SaveTable(pickle_name, dirname, folder)






# mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
# mesh.ngmesh.Save("mytestmesh.vol.gz")
# mesh = Mesh("mytestmesh.vol.gz")
# Draw(mesh)
 
# from netgen.read_gmsh import ReadGmsh
# from netgen.meshing import *
# import the Gmsh file to a Netgen mesh object
# mesh = ReadGmsh("coarse_test")
# mesh = Mesh(mesh)
# # mesh.ngmesh.Save("newmesh.vol")
        # mesh2 = open("newmesh.vol")
        # mesh2 = Mesh(mesh2)
        # Draw(mesh2)
        # input()
        # quit()

# PROBLEM SETTING
# PROBLEM = 1: plane wave solution (Example 5.1, Tables 5.2-5.5), exact solution available and adjustable kwave 
# PROBLEM = 2: localised interior source (Example 5.2, Table 5.6), no exact solution, use of bubbles
# PROBLEM = 3: localised boundary source (Example 5.3, Table 5.7), no exact solution, periodic structure (NOT YET IMPLEMENTED -> mesh needs to change)
# PROBLEM = 4: crystal configuration with square inclusions. ATTENTION: incl==0 does not work later - CHECK.
# PROBLEM = 5: crystal configuration with circular inclusions. We can choose the number of inclusions per cell. 


# Eigenvalue computation
# alpha = variables_dictionary["alpha"]
# omega = variables_dictionary["omega"]
# beta = variables_dictionary["beta"]
# dom_bnd = variables_dictionary["dom_bnd"]

# fes = H1(mesh, order = 3, complex = True) #dirichlet = dom_bnd) 
# u, v = fes.TnT()
# a = BilinearForm(fes)
# a += alpha * grad(u) * grad(v) * dx() 
# a += -1J * omega * beta * u * v * ds(dom_bnd, bonus_intorder = 10)

# m = BilinearForm(fes)
# m +=  u * v * dx()
# m.Assemble()

# pre = Preconditioner(a, "direct")
# a.Assemble()

# u = GridFunction(fes)

# evals, evecs = PINVIT(a.mat, m.mat, pre=pre, num =1)
# print(evals)
# u.vec.data = evecs[0]

# Draw(u,mesh,"u")

# r = sqrt(x**2 + y**2)
# sol = -cos(r * pi/2)
# Draw(sol, mesh, "sol")
# # input()
        