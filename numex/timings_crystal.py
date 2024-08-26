# LIBRARIES
from helmholtz_aux import *
from ngsolve.eigenvalues import PINVIT



order = 5
Bubble_modes = [0]
if not os.path.exists("timings_crystal"):
    os.mkdir("timings_crystal")

dirname = os.path.dirname(__file__)
save_timings = True


cases = [(1,  0.05, 32),
         (2,  0.05, 64),
         (4,  0.05, 128),
         (8,  0.05, 256),
         (16, 0.05, 512),
         ]

# cases = [(16, 0.1, 12),
#          ]


for incl, maxH, EE in cases:
    Lx = 1 * incl  
    Nx = 16 // incl 
    defects = np.ones((Nx,Nx))

    mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Nx, alpha_outer = 1/12.1, alpha_inner = 1, defects = defects, incl = incl, r = 0.126, Lx = Lx, Ly = Lx, load_mesh = True)
    # Draw(mesh)
    # input()
    acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = 0, mesh_info = mesh_info, alpha = alpha, omega = 1, kappa = 1, f = 0, g = 1, beta = 1, gamma = 1, save_localbasis=False, save_extensions = False)
                
    edge_basis = acms.calc_edge_basis()
    
    if edge_basis:
        acms.Assemble()
        usmall = acms.Solve()

        print(Norm(usmall))
        acms.PrintTiminigs()

        # import matplotlib.pyplot as plt
        # plt.spy(acms.asmall)
        # plt.savefig('spyplot_J36_Ie16.png', dpi=400)
        # # plt.show()
        # quit()

        if save_timings:
            ex_data = {"maxH": maxH, "Ncell": 16, "order": order, "Ie": EE, "ne" : acms.ndofemax, "acmsndof" : acms.acmsdofs}
            timings = acms.timings

            pickle_name =   "maxH:" +     str(maxH) + "_" + \
                            "Ncell:" +    str(16) + "_" + \
                            "order:" +   str(order) + "_" + \
                            "Ie:" +      str(EE) + "_" + \
                            ".dat"
            save_file = os.path.join(dirname, "timings_crystal/" + pickle_name)
            picklefile = open(save_file, "wb")
            data = [ex_data, timings]
            pickle.dump(data, picklefile)
            picklefile.close()

    