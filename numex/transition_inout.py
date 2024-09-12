# LIBRARIES
from helmholtz_aux import *
import matplotlib.pyplot as plt

for ii in range(6):
    for jj in range(6):
        calc_fem = True
        calc_acms = True

        problem = 5
        incl = 2
        Nx = (10 + 20 * ii) // incl # number of cells in x direction
        Ny = (10 + 20 * jj) // incl       # number of cells in y direction

        Ncell = Nx * Ny * incl**2 

        omega = 1.48 # 1.8288
        # omega = 1.26

        Href = 0
        maxH = 0.1
        order = 5
        EE = 8

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

        mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer, alpha_inner, defects, layers, load_mesh = True)



        if True:
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
                
                # draw all domains!
                # doms = None
                ###############################################################
                    
                acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = kappa, f = f, g = g, beta = beta, gamma = gamma, save_doms = doms)
                                
                edge_basis = acms.calc_edge_basis()
                if edge_basis:
                    acms.Assemble()
                    usmall = acms.Solve() 

                    gfu_acms = GridFunction(acms.Vc)

                    acms.SetGlobalFunction(gfu_acms, usmall, doms)
                    
                acms.PrintTiminigs(all = False)
                
                Draw(gfu_acms, mesh, "uacms")
                

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

            xx_real = np.linspace(- off - N_draw, + off + N_draw, N)

            if calc_fem: 
                plt.plot(xx_real,wave_in, label = "left")
                plt.plot(xx_real,wave_out, label = "out")

            if calc_acms:
                plt.plot(xx_real,acms_wave_in, label = "left_acms")
                plt.plot(xx_real,acms_wave_out, label = "out_acms")
                

            plt.legend()
            
            file_name = "transition_" +  "maxH:" +     str(maxH) + "_" + \
                                        "Ncell:" +    str(Ncell) + "_" + \
                                        "Incl:" +    str(incl) + "_" + \
                                        "Nx:" +    str(Nx) + "_" + \
                                        "Ny:" +    str(Ny) + "_" + \
                                        "order:" +   str(order) + "_" + \
                                        "Ie:" +      str(EE) + "_" + \
                                        "omega:" +      str(omega) + "_"

            folder = "transition_inout"
            if not os.path.exists(folder):
                    os.mkdir(folder)

            dirname = os.path.dirname(__file__)
            
            save_file = os.path.join(dirname, folder + "/" +  file_name + ".png")

            plt.savefig(save_file, dpi=400)
            
            save_file = os.path.join(dirname, folder + "/" +  file_name + ".out")
            file = open(save_file, "w")
            header = "y\t"

            if calc_fem:
                header += "left\tright\t"
            if calc_acms:
                header += "leftA\trightA\t"

            header = header[:-1] + "\n"
            file.write(header)
            for i in range(len(xx_real)):
                line = str(xx_real[i]) + "\t"
                if calc_fem:
                    line += str(wave_in[i]) + "\t" + str(wave_out[i]) + "\t"
                if calc_acms:
                    line += str(acms_wave_in[i]) + "\t" + str(acms_wave_out[i]) + "\t"
                line = line[:-1] + "\n"
                file.write(line)
            file.close
            