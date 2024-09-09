# LIBRARIES
from helmholtz_aux import *
from ngsolve.eigenvalues import PINVIT


Ncell = int(input("Ncell: "))
maxH = float(input("maxH: "))
order = int(input("order: "))
EE = int(input("EdgeModes: ")) #,4,8,16,32,64,128,256]


Bubble_modes = [0]

Nx = int(sqrt(Ncell))
Ny = Nx           

mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl = -1, r = 0.0, Lx = 1, Ly = 1, load_mesh = True)


if not os.path.exists("spyplots"):
    os.mkdir("spyplots")

dirname = os.path.dirname(__file__)

acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = 0, mesh_info = mesh_info, alpha = 1, omega = 1, kappa = 1, f = 0, g = 1, beta = 1, gamma = 1, save_doms=[])
            
edge_basis = acms.calc_edge_basis()


if edge_basis:
    acms.Assemble()
    usmall = acms.Solve()

    import matplotlib.pyplot as plt
    plt.spy(acms.asmall)
    figname = "spy_J_" + str(int(sqrt(Ncell))) + "_Ie_" + str(EE) + ".png" 
    plt.savefig( dirname + "/spyplots/" + figname, dpi=400)
    plt.show()
    