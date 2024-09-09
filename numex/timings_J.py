# LIBRARIES
from helmholtz_aux import *
from ngsolve.eigenvalues import PINVIT


Js = [i**2 for i in range(1,11)] # [1,4,9,16,25,36]

maxH = 0.1
order = 4
Edge_modes = 4 #[4,8,16,32,64,128,256]

if not os.path.exists("timings_J"):
    os.mkdir("timings_J")

dirname = os.path.dirname(__file__)

save_timings = True
for Ncell in Js:
    
    mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx = int(sqrt(Ncell)), Ny = int(sqrt(Ncell)), incl = -1, r = 0.0, Lx = 1, Ly = 1, load_mesh = True)

    acms = ACMS(order = order, mesh = mesh, bm = 0, em = Edge_modes, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = 1, omega = 1, kappa = 1, f = 0, g = 1, beta = 1, gamma = 1, save_doms = [])
                
    edge_basis = acms.calc_edge_basis()
    
    if edge_basis:
        acms.Assemble()
        usmall = acms.Solve()

        print(Norm(usmall))
        acms.PrintTiminigs()
        # Draw(gfu)
        # input()

        if save_timings:
            ex_data = {"maxH": maxH, "Ncell": Ncell, "order": order, "Ie": Edge_modes, "ne" : acms.ndofemax, "acmsndof" : acms.acmsdofs}
            timings = acms.timings

            pickle_name =   "maxH:" +     str(maxH) + "_" + \
                            "Ncell:" +    str(Ncell) + "_" + \
                            "order:" +   str(order) + "_" + \
                            "Ie:" +      str(Edge_modes) + "_" + \
                            ".dat"
            save_file = os.path.join(dirname, "timings_J/" + pickle_name)
            picklefile = open(save_file, "wb")
            data = [ex_data, timings]
            pickle.dump(data, picklefile)
            picklefile.close()

    