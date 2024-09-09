# LIBRARIES
from helmholtz_aux import *

calc_fem = True
calc_acms = False

problem = 5


ss = 0.5
ee = 2
omega_v = list(np.arange(ss,ee,0.001))

Href = 0
maxH = 0.2
order = 3
EE = 4
ACMS_flag = 0

Bubble_modes = [0]

incl = 2
Nx = 10 // incl # number of cells in x direction
Ny = 10 // incl  
Ncell = Nx * Ny * incl**2 


r  = 0.25    # radius of inclusion
Lx = 1 * incl   #* 0.484 #"c"
Ly = Lx        #0.685 #"a
alpha_outer = 1/12.1 #SILICON
alpha_inner = 1 #0 #AIR        
# alpha_outer = 1  #AIR
# alpha_inner = 1./12.1 #0 #SILICON        
layers = 0


ix = [i for i in range(layers)] + [Nx - 1 - i for i in range(layers)]
iy = [Ny//(2)] # #[i for i in range(layers)] + [Ny - 1 - i for i in range(layers)]

defects = np.ones((Nx,Ny))
for i in ix:
    for j in range(Ny): 
        defects[i,j] = 0.0

for j in iy:
    for i in range(Nx): 
        defects[i,j] = 0.0 

load_mesh = True
mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer, alpha_inner, defects, layers, load_mesh)


# input()
# print(mesh.Get)


ints_right = []
ints_left = []

ints_right_acms = []
ints_left_acms = []
# SetNumThreads(12)

# TaskManager().__enter__
for omega in omega_v:
    # omega = omega_v[0]
    # for EE in Edge_modes:
    # print("Edgemodes = ", EE)
    print("omega = ", omega)
    
    kappa = omega    #**2 * alpha 
    k_ext = omega    #**2 # * alpha=1
    k = k_ext * CF((1,0)) #CF = CoefficientFunction
    beta = - k_ext / omega
    f = 0 
    sigma = 1
    off = Ny*incl/2 - 0.5 * incl
    peak = exp(-(y-off)**2*sigma)
    # print("off = ", off)
    # Draw(y-off, mesh, "yy")
    g_list = []
    gg = 1j * omega * exp(-1j * omega * x) * peak
    for bnd in mesh.GetBoundaries():
        if "dom_bnd_left" in bnd:
            g_list.append(gg)
        else:
            g_list.append(0)
    g = CF(g_list)

    # g = 1j * (k_ext - k * specialcf.normal(2)) * exp(-1j * (k[0] * x + k[1] * y)) * peak # Incoming plane wave 
    # Draw(g, mesh, "g")
    u_ex = 0
    sol_ex = 0
    Du_ex = 0
    gamma = 1
    
    # solution_dictionary = ground_truth(mesh, variables_dictionary, 10)
    if calc_fem:
        start = time.time()
        
        V = H1(mesh, order = order, complex = True) 
        # print("V.ndof = ", V.ndof)
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
    if calc_acms:
        ###############################################################
        # reconstruct the acms_solution only on the cells next to the domain boundary on the left and the right side
        domsba = BitArray(len(mesh.GetMaterials()))
        domsba[:] = 0
        for i, edgename in enumerate(mesh.GetBoundaries()):
            if "dom_bnd_left_V" in edgename or "dom_bnd_right_V" in edgename:
                domsba = domsba | mesh.Boundaries(edgename).Neighbours(VOL).Mask()
        
        doms = []
        for ii, d in enumerate(domsba):
            if d == 1:
                doms.append(mesh.ngmesh.GetMaterial(ii+1))
        
        # draw all domains!
        # doms = None
        ###############################################################
              
        acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = kappa, f = f, g = g, beta = beta, gamma = gamma, save_doms = doms)
                  
                
        edge_basis = acms.calc_edge_basis()
        # print("edge_basis = ", edge_basis)
        if edge_basis:
            acms.Assemble()
            usmall = acms.Solve() 

            gfu_acms = GridFunction(acms.Vc)

            acms.SetGlobalFunction(gfu_acms, usmall, doms)
        acms.PrintTiminigs(all = False)
        
    # # intval_left = acms.IntegrateACMS("dom_bnd_left_V", usmall)
    # # intval_right = acms.IntegrateACMS("dom_bnd_right_V", usmall)
    intval_left = 0
    intval_right = 0
    intval_left_acms = 0
    intval_right_acms = 0
    rr = 0
    rr_acms = 0
    if calc_fem:
        rr = gfu_fem.real**2  + gfu_fem.imag**2
    if calc_acms:
        rr_acms = gfu_acms.real**2  + gfu_acms.imag**2
    for i, edgename in enumerate(mesh.GetBoundaries()):
        if "dom_bnd_left_V" in edgename:
            if calc_fem:
                intval_left+= Integrate(rr, mesh, definedon = mesh.Boundaries(edgename))
            if calc_acms:
                intval_left_acms+= Integrate(rr_acms, mesh, definedon = mesh.Boundaries(edgename))
        if "dom_bnd_right_V" in edgename:
            if calc_fem:
                intval_right+= Integrate(rr, mesh, definedon = mesh.Boundaries(edgename))
            if calc_acms:
                intval_right_acms+= Integrate(rr_acms, mesh, definedon = mesh.Boundaries(edgename))

    ints_left.append(sqrt(intval_left))
    ints_right.append(sqrt(intval_right))

    ints_left_acms.append(sqrt(intval_left_acms))
    ints_right_acms.append(sqrt(intval_right_acms))


folder = "sweep"

if not os.path.exists(folder):
    os.mkdir(folder)

dirname = os.path.dirname(__file__)


file_name = "sweep_" +  "maxH:" +     str(maxH) + "_" + \
            "Ncell:" +    str(Ncell) + "_" + \
            "Incl:" +    str(incl) + "_" + \
            "Nx:" +    str(Nx) + "_" + \
            "Ny:" +    str(Ny) + "_" + \
            "order:" +   str(order) + "_" + \
            "Ie:" +      str(EE) + "_" + \
            "omega_min:" +      str(ss) + "_" +\
            "omega_max:" +      str(ee)

save_file = os.path.join(dirname, folder + "/" +  file_name + ".dat")
file = open(save_file, "w")

header = "w\t"
if calc_fem:
    header += "left\tright\t"
if calc_acms:
    header += "leftA\trightA\t"

header = header[:-1] + "\n"
file.write(header)
for i in range(len(omega_v)):
    line = str(omega_v[i]) + "\t"
    if calc_fem:
        line += str(ints_left[i]) + "\t" + str(ints_right[i]) + "\t"
    if calc_acms:
        line += str(ints_left_acms[i]) + "\t" + str(ints_right_acms[i]) + "\t"
    line = line[:-1] + "\n"
    file.write(line)
file.close

import matplotlib.pyplot as plt

plt.plot(omega_v, ints_left, label = "left")
plt.plot(omega_v, ints_right, label = "right")
plt.legend()
plt.show()        
        