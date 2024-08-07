# LIBRARIES
from helmholtz_aux import *
from ngsolve.eigenvalues import PINVIT


Ncell = 16


maxH = 0.1
order = 4
Edge_modes = [2,4,8,16,32,64,128,256]

Bubble_modes = [0]

Nx = int(sqrt(Ncell))
Ny = Nx           

mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl = -1, r = 0.0, Lx = 1, Ly = 1, load_mesh = True)

if not os.path.exists("timings_Ie"):
    os.mkdir("timings_Ie")

dirname = os.path.dirname(__file__)

for EE in Edge_modes:
    print("Edgemodes = ", EE)
     
    acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = 1, omega = 1, kappa = 1, f = 0, g = 1, beta = 1, gamma = 1, save_localbasis=False, save_extensions = False)
                
    edge_basis = acms.calc_edge_basis()
    
    if edge_basis:
        acms.Assemble()
        usmall = acms.Solve()

        print(Norm(usmall))
        acms.PrintTiminigs()
        # Draw(gfu)
        # input()

# TaskManager().__exit__