# LIBRARIES
from geometries import *
from acms_class import *
import numpy as np 

import netgen.gui

incl = 2
Nx = 10 // incl # number of cells in x direction
Ny = 10 // incl # number of cells in y direction

Ncell = Nx * Ny * incl**2 


omega = 1.48 

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

mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer, alpha_inner, defects, layers, load_mesh = True)

k_vec = omega * CF((1,0)) 
f = 0 
sigma = 2
off = Ny*incl/2 - 0.5 * incl
peak = exp(-(y-off)**2*sigma)
# Incoming plane wave times peak
g = 1j * (omega - k_vec * specialcf.normal(2)) * exp(-1j * (k_vec[0] * x + k_vec[1] * y)) * peak 
   
acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = omega, f = f, g = g, beta = -1, gamma = 1)
                
edge_basis = acms.calc_edge_basis()
if edge_basis:
    acms.Assemble()
    usmall = acms.Solve() 

    gfu_acms = GridFunction(acms.Vc)

    acms.SetGlobalFunction(gfu_acms, usmall)
    
acms.PrintTiminigs(all = False)

Draw(gfu_acms.real, mesh, "uacms.real")
