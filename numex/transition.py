# LIBRARIES
from helmholtz_aux import *
import matplotlib.pyplot as plt

do_draw = False
save_fig = True
draw_fig = False
calc_fem = False
calc_acms = True
vtk_do = False

if do_draw:
    import netgen.gui
from ngsolve.eigenvalues import PINVIT


problem = 5
incl = 2
Nx = 10 // incl # number of cells in x direction
Ny = 10 // incl       # number of cells in y direction

Ncell = Nx * Ny * incl**2 


omega = 1.48 # 1.8288

Href = 0
maxH = 0.2
order = 4
EE = 16



r  = 0.25 #0.126     # radius of inclusion
Lx = 1 * incl   #* 0.484 #"c"
Ly = Lx        #0.685 #"a

alpha_outer = 1/12.1 #SILICON
alpha_inner = 1 #0 #AIR             
layers = 0

ix = [i for i in range(layers)] + [Nx - 1 - i for i in range(layers)]
iy = [Ny//(2)] #
# iy = [i for i in range(layers)] + [Ny - 1 - i for i in range(layers)]

defects = np.ones((Nx,Ny))
for i in ix:
    for j in range(Ny): 
        defects[i,j] = 0.0

for j in iy:
    for i in range(Nx): 
        defects[i,j] = 0.0 

mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer, alpha_inner, defects, layers, load_mesh = False)



if True:
    print("omega = ", omega)
    
    kappa = omega    #**2 * alpha 
    k_ext = omega    #**2 # * alpha=1
    k = k_ext * CF((1,0)) #CF = CoefficientFunction
    beta = - k_ext / omega
    f = 0 
    sigma = 2
    off = Ny*incl/2 - 0.5 * incl
    peak = exp(-(y-off)**2*sigma)
    # print("off = ", off)
    # Draw(y-off, mesh, "yy")
    g = 1j * (k_ext - k * specialcf.normal(2)) * exp(-1j * (k[0] * x + k[1] * y)) * peak # Incoming plane wave 
    Draw(g, mesh, "g")
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
        
        # doms = None #[]
        ###############################################################
              
        acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = kappa, f = f, g = g, beta = beta, gamma = gamma, save_localbasis=doms, save_extensions = doms)
                        
        edge_basis = acms.calc_edge_basis()
        if edge_basis:
            acms.Assemble()
            usmall = acms.Solve() 

            gfu_acms = GridFunction(acms.Vc)

            acms.SetGlobalFunction(gfu_acms, usmall, doms)
            
        acms.PrintTiminigs(all = False)
        
        Draw(gfu_acms, mesh, "uacms")
        
    if do_draw:
        input()

    # number of cell in upper and lower direction plus waveguide, i.e. N_craw = 2 plots the solution over 5 cells
    N_draw = 2

    N = 100 * (2*N_draw + 1)
    center = -0.5 * incl + Ny*incl/2
    off = 0.5 * incl
    
    xx = np.linspace(center - off - N_draw, center + off + N_draw, N)
    wave_in = []
    wave_out = []

    acms_wave_in = []
    acms_wave_out = []


    for xi in xx:
        if calc_fem:
            val = gfu_fem(mesh(-off, xi))
            val_out = gfu_fem(mesh(-off+Nx*incl, xi))

            wave_in.append(sqrt(val.real**2 + val.imag**2))
            wave_out.append(sqrt(val_out.real**2 + val_out.imag**2))

        if calc_acms:
            acms_val = gfu_acms(mesh(-off, xi))
            acms_val_out = gfu_acms(mesh(-off+Nx*incl, xi))

            acms_wave_in.append(sqrt(acms_val.real**2 + acms_val.imag**2))
            acms_wave_out.append(sqrt(acms_val_out.real**2 + acms_val_out.imag**2))

    if calc_fem:    
        plt.plot(xx,wave_in, label = "left")
        plt.plot(xx,wave_out, label = "out")

    if calc_acms:
        plt.plot(xx,acms_wave_in, label = "left_acms")
        plt.plot(xx,acms_wave_out, label = "out_acms")
        

    plt.legend()
    
    file_name = "transition_" +  "maxH:" +     str(maxH) + "_" + \
                                "Ncell:" +    str(Ncell) + "_" + \
                                "Incl:" +    str(Incl) + "_" + \
                                "Nx:" +    str(Nx) + "_" + \
                                "Ny:" +    str(Ny) + "_" + \
                                "order:" +   str(order) + "_" + \
                                "Ie:" +      str(EE) + "_" + \
                                "omega:" +      str(omega) + "_"
                                
    
    if save_fig:
        if not os.path.exists("transition"):
            os.mkdir("transition")

        dirname = os.path.dirname(__file__)
        # file_name = "transition_" +  "maxH:" +     str(maxH) + "_" + \
        #                         "Ncell:" +    str(Ncell) + "_" + \
        #                         "order:" +   str(order) + "_" + \
        #                         "Ie:" +      str(EE) + "_" + \
        #                         "omega:" +      str(omega) + "_" + \
        #                         ".png"

        save_file = os.path.join(dirname, "transition/" +  file_name + ".png")

        plt.savefig(save_file, dpi=400)
    if draw_fig:
        plt.show()

    if vtk_do:
        if not os.path.exists("vtks"):
            os.mkdir("vtks")

        fields = []
        field_names = []

        if calc_fem: 
            # V = VectorH1(mesh, order = order)
            # gf = GridFunction(V)
            # gf.Set(CF((gfu_fem.imag,gfu_fem.imag)))
            
            fields.append(gfu_fem.imag)
            fields.append(gfu_fem.real)
            field_names.append("gfu_fem_imag")
            field_names.append("gfu_fem_real")
        if calc_acms: 
            fields.append(gfu_acms.imag)
            fields.append(gfu_acms.real)
            field_names.append("gfu_acms_imag")
            field_names.append("gfu_acms_real")
        # print(field_names)

        vtk = VTKOutput(ma=mesh, coefs = fields, names =  field_names, filename="./vtks/" + file_name + ".vtk", subdivision=3)
        vtk.Do()



    