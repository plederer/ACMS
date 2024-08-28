# LIBRARIES
from helmholtz_aux import *
from ngsolve.eigenvalues import PINVIT


Ncell = 16


maxH = 0.01
order = 6
Edge_modes = [2,4,8,16,32,64,128,256]

Bubble_modes = [0]

Nx = int(sqrt(Ncell))
Ny = Nx           

mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl = -1, r = 0.0, Lx = 1, Ly = 1, load_mesh = True)

# for bnd in mesh.GetBoundaries():
#     print("bnd name = ", bnd)
#     V = H1(mesh, order = 1) #, definedon = mesh.Boundaries(bnd))
#     gfu = GridFunction(V)
#     gfu.Set(1, definedon = mesh.Boundaries(bnd))
#     Draw(gfu)
#     input()



if not os.path.exists("timings_Ie"):
    os.mkdir("timings_Ie")

dirname = os.path.dirname(__file__)

save_timings = True
for EE in Edge_modes:
    print("Edgemodes = ", EE)
     
    acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = 0, mesh_info = mesh_info, alpha = 1, omega = 1, kappa = 1, f = 0, g = 1, beta = 1, gamma = 1, save_localbasis=False, save_extensions = False)
                
    edge_basis = acms.calc_edge_basis()
    
    if edge_basis:
        acms.Assemble()
        usmall = acms.Solve()

        print(Norm(usmall))
        acms.PrintTiminigs()
        # Draw(gfu)
        # input()
        
        # import matplotlib.pyplot as plt
        # plt.spy(acms.asmall)
        # plt.savefig('spyplot_J36_Ie16.png', dpi=400)
        # # plt.show()
        # quit()

        if save_timings:
            ex_data = {"maxH": maxH, "Ncell": Ncell, "order": order, "Ie": EE, "ne" : acms.ndofemax, "acmsndof" : acms.acmsdofs}
            timings = acms.timings

            pickle_name =   "maxH:" +     str(maxH) + "_" + \
                            "Ncell:" +    str(Ncell) + "_" + \
                            "order:" +   str(order) + "_" + \
                            "Ie:" +      str(EE) + "_" + \
                            ".dat"
            
            save_file = os.path.join(dirname, "timings_Ie/" + pickle_name)
            picklefile = open(save_file, "wb")
            data = [ex_data, timings]
            pickle.dump(data, picklefile)
            picklefile.close()

    