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
Ncell = 14
incl = 2

# a = 2 * (0.5-0.126) + 2
ss = 1.2
ee = 3
omega_v = list(np.arange(ss,ee,0.005)) #list([i/10 for i in range(10,50)]

Href = 0
maxH = 0.2
order_v = [2]
Bubble_modes = [0]
Edge_modes = [8]
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
alpha_inner = 10 #0 #AIR        
# alpha_outer = 1  #AIR
# alpha_inner = 1./12.1 #0 #SILICON        
layers = 0

ix = [3] #[i for i in range(layers)] + [Nx - 1 - i for i in range(layers)]
iy = [3] #[i for i in range(layers)] + [Ny - 1 - i for i in range(layers)]

defects = np.ones((Nx,Ny))
for i in ix:
    for j in range(3+1): 
        defects[i,j] = 0.0

for j in iy:
    for i in range(3+1): 
        defects[i,j] = 0.0 

mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer, alpha_inner, defects, layers)
V = H1(mesh, order = order_v[0], complex = True)

ints_right = []
ints_left = []
# SetNumThreads(12)
with TaskManager():
    for omega in omega_v:
        print("omega = ", omega)
        
        kappa = omega    #**2 * alpha 
        k_ext = omega    #**2 # * alpha=1
        k = k_ext * CF((1,0)) #CF = CoefficientFunction
        beta = - k_ext / omega
        f = 0 
        sigma = 1
        peak = exp(-(y-Ny+1)**2*sigma)
        g = 1j * (k_ext - k * specialcf.normal(2)) * exp(-1j * (k[0] * x + k[1] * y)) *peak # Incoming plane wave 
        # Draw(g, mesh, "g")
        u_ex = 0
        sol_ex = 0
        Du_ex = 0
        gamma = 1
        
        # solution_dictionary = ground_truth(mesh, variables_dictionary, 10)
        if True:
            start = time.time()
            
            V = H1(mesh, order = 3, complex = True) 
            
            u, v = V.TnT()

            a = BilinearForm(V)
            a += alpha * grad(u) * grad(v) * dx() 
            a += - gamma * kappa**2 * u * v * dx()
            a += -1J * omega * beta * u * v * ds(dom_bnd, bonus_intorder = 10)
            a.Assemble()

            l = LinearForm(V)
            l += f * v * dx(bonus_intorder=10)
            l += g * v * ds(dom_bnd,bonus_intorder=10)
            l.Assemble()
            
            
            gfu_fem = GridFunction(V)
            ainv = a.mat.Inverse(V.FreeDofs(), inverse = "sparsecholesky")
            gfu_fem.vec.data = ainv * l.vec
            

            Draw(gfu_fem, mesh,"ufem")
            # uinc = exp(-1J*omega * x)
            # Draw(uinc, mesh, "u_inc")
            # Draw(gfu_fem - uinc, mesh,"u_scatter")

        if False:
            acms = ACMS(order = order_v[0], mesh = mesh, bm = 0, em = Edge_modes[0], bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = kappa, f = f, g = g, beta = beta, gamma = gamma)
                        
            edge_basis = acms.calc_edge_basis()
            # print("edge_basis = ", edge_basis)
            if edge_basis:

                start = time.time()
                acms.CalcHarmonicExtensions()
                                            
                assemble_start = time.time()
                for m in acms.doms:
                    acms.Assemble_localA(m)
                # print("assemble = ", time.time() - assemble_start)

            gfu, num, usmall = compute_acms_solution(mesh, V, acms, edge_basis, setglobal=True)
            # Draw(gfu, mesh, "uacms")
            # Draw(x, mesh, "x")
        if do_draw:
            input()
        # intval_left = acms.IntegrateACMS("dom_bnd_left_V", usmall)
        # intval_right = acms.IntegrateACMS("dom_bnd_right_V", usmall)
        intval_left = 0
        intval_right = 0
        rr = gfu_fem.real**2  + gfu_fem.imag**2
        for i, edgename in enumerate(mesh.GetBoundaries()):
            if "dom_bnd_left_V" in edgename:
                # print(edgename)
                # print(i)
                intval_left+= Integrate(rr, mesh, definedon = mesh.Boundaries(edgename))
            # if "dom_bnd_right_V" in edgename:
            if "dom_bnd_bottom_H" in edgename:
                # print(edgename)
                # print(i)
                intval_right+= Integrate(rr, mesh, definedon = mesh.Boundaries(edgename))

        # intval_left+= Integrate(rr, mesh, definedon = mesh.Boundaries("dom_bnd_left_V3"))
        # intval_right+= Integrate(rr, mesh, definedon = mesh.Boundaries("dom_bnd_right_V10"))

        # test = GridFunction(H1(mesh, order = 1, dirichlet=".*"))
        # test.Set(1, definedon = mesh.Boundaries("dom_bnd_right_V10"))
        # Draw(test)
        # input()
        # print("integral = ", intval)
        ints_left.append(sqrt(intval_left))
        ints_right.append(sqrt(intval_right))

# print("ints_right = ", ints_right)
# print("ints_left = ", ints_left)

import matplotlib.pyplot as plt

plt.plot(omega_v, ints_left, label = "left")
plt.plot(omega_v, ints_right, label = "right")
plt.legend()
plt.show()        
        