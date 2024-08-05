# LIBRARIES
from helmholtz_aux import *

do_draw = False
if do_draw:
    import netgen.gui
from ngsolve.eigenvalues import PINVIT

# problem = 5 #float(input("Choose the problem. \n 1 = PW. \n 2 = LIS. \n 3 = LBS. \n 4 = Crystal Sq. \n 5 = Crystal \n Problem =  "))
# # omega = float(input("Wavenumber k: "))
# omega_v = 1 #list(map(float, input("Wavenumber k vector = ").split()))
# maxH = float(input("maxH: "))
# Href = int(input("Number of mesh refinements refH (0 is no refinements): "))
# order_v = list(map(int, input("Order of approximation. Vector = ").split())) 
# # Bubble_modes = list(map(int, input("Number of bubble modes. Vector = ").split()))
# Edge_modes = list(map(int, input("Number of edge modes. Vector = ").split()))


# if problem == 5:
#     Ncell = int(input("Number of cells in one direction: "))
#     incl = int(input("Number of inclusions in one direction per cell incl (Power of 2): "))
#     ACMS_flag = 0 #FEM vs ACMS error
# elif problem == 1:
#     Ncell = 0
#     incl = 0
#     ACMS_flag = int(input("Error against exact solution = 1 or FEM solution = 0. "))
# else:
#     Ncell = 0
#     incl = 0
#     ACMS_flag = 0

# FOR TESTING
problem = 5
Ncell = 16
incl = 1

# a = 2 * (0.5-0.126) + 2
ss = 0.8
ee = 2
omega_v = [1.12] #list(np.arange(ss,ee,0.005)) #list([i/10 for i in range(10,50)]

Href = 0
maxH = 0.2
order_v = [4]
Bubble_modes = [0]
Edge_modes = [2,4,8,16]
ACMS_flag = 0

Bubble_modes = [0]

error_table = 1
table_content_l2_aux = ""
table_content_h1_aux = ""
table_header = ""
table_separation = ""
table_end = ""


r  = 0.126     # radius of inclusion
Lx = 1 * incl   #* 0.484 #"c"
Ly = Lx        #0.685 #"a
Nx = Ncell // incl # number of cells in x direction
Ny = Nx       # number of cells in y direction
alpha_outer = 1/12.1 #SILICON
alpha_inner = 1 #0 #AIR        
# alpha_outer = 1  #AIR
# alpha_inner = 1./12.1 #0 #SILICON        
layers = 0

wg = 3

ix = [i for i in range(layers)] + [Nx - 1 - i for i in range(layers)]
iy = [i for i in range(layers)] + [Ny - 1 - i for i in range(layers)]

defects = np.ones((Nx,Ny))
for i in ix:
    for j in range(Ny): 
        defects[i,j] = 0.0

for j in iy:
    for i in range(Nx): 
        defects[i,j] = 0.0 

load_mesh = True
mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer, alpha_inner, defects, layers, load_mesh)
V = H1(mesh, order = order_v[0], complex = True)

# print(mesh.Get)

ints_right = []
ints_left = []
# SetNumThreads(12)

if not os.path.exists("timings"):
    os.mkdir("timings")

dirname = os.path.dirname(__file__)

# TaskManager().__enter__
# for omega in omega_v:
omega = omega_v[0]
order = order_v[0]
for EE in Edge_modes:
    print("Edgemodes = ", EE)
           
    kappa = omega    #**2 * alpha 
    k_ext = omega    #**2 # * alpha=1
    k = k_ext * CF((1,0)) #CF = CoefficientFunction
    beta = - k_ext / omega
    f = 0 
    sigma = 1
    off = Ncell/2 - 0.5 * incl
    peak = exp(-(y-off)**2*sigma)
    # print("off = ", off)
    # Draw(y-off, mesh, "yy")
    g = 1j * (k_ext - k * specialcf.normal(2)) * exp(-1j * (k[0] * x + k[1] * y)) *peak # Incoming plane wave 
    # Draw(g, mesh, "g")
    u_ex = 0
    sol_ex = 0
    Du_ex = 0
    gamma = 1
    
    # solution_dictionary = ground_truth(mesh, variables_dictionary, 10)
    if False:
        start = time.time()
        
        V = H1(mesh, order = 2, complex = True) 
        print("V.ndof = ", V.ndof)
        u, v = V.TnT()

        a = BilinearForm(V)
        a += alpha * grad(u) * grad(v) * dx() 
        a += - gamma * kappa**2 * u * v * dx()
        a += -1J * omega * beta * u * v * ds(dom_bnd, bonus_intorder = 10)
        with TaskManager():
            a.Assemble()

        l = LinearForm(V)
        l += f * v * dx(bonus_intorder=10)
        l += g * v * ds(dom_bnd,bonus_intorder=10)
        with TaskManager():
            l.Assemble()
        
        
        gfu_fem = GridFunction(V)
        with TaskManager():
            ainv = a.mat.Inverse(V.FreeDofs(), inverse = "sparsecholesky")
            gfu_fem.vec.data = ainv * l.vec
        

        Draw(gfu_fem, mesh,"ufem")
        print(Norm(gfu_fem.vec))
        quit()
        # uinc = exp(-1J*omega * x)
        # Draw(uinc, mesh, "u_inc")
        # Draw(gfu_fem - uinc, mesh,"u_scatter")

        # input()
    if True:
        acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = kappa, f = f, g = g, beta = beta, gamma = gamma)
        
        
        # input()
                
        edge_basis = acms.calc_edge_basis()
        # print("edge_basis = ", edge_basis)
        if edge_basis:
            # start = time.time()
            acms.CalcHarmonicExtensions()
            
            # assemble_start = time.time()
            # for m in acms.doms:
            #     acms.Assemble_localA(m)
            acms.Assemble()
            # print("assemble = ", time.time() - assemble_start)

            gfu, num, usmall = compute_acms_solution(mesh, V, acms, edge_basis, setglobal=False)
            
            acms.PrintTiminigs()

        ex_data = {"maxH": maxH, "incl": incl, "Ncell": Ncell,"order": order, "Ie": EE, "ne" : acms.ndofemax, "acmsndof" : acms.acmsdofs}
        timings = acms.timings

        pickle_name =   "maxH:" +     str(maxH) + "_" + \
                        "Nx:" +       str(Nx) + "_" + \
                        "Ny:" +       str(Ny) + "_" + \
                        "incl:" +     str(incl) + "_" + \
                        "Ncell:" +    str(Ncell) + "_" + \
                        "layers:" +   str(layers) + "_" + \
                        "order:" +   str(order) + "_" + \
                        "Ie:" +      str(EE) + "_" + \
                        ".dat"
        picklefile = open(save_file, "wb")
        save_file = os.path.join(dirname, "timings/" + pickle_name)
        data = [ex_data, timings]
        pickle.dump(data, picklefile)
        picklefile.close()