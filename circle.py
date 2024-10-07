# LIBRARIES
from geometries import *
from acms_class import *
import numpy as np 

import netgen.gui


omega = 1

maxH = 0.2
order = 4
EE = 16


unit_disc

mesh, dom_bnd, alpha, mesh_info = unit_disc(maxH)

k_vec = omega * CF((1,0)) 
f = 0 
sigma = 2

g = 1j * (omega - k_vec * specialcf.normal(2)) * exp(-1j * (k_vec[0] * x + k_vec[1] * y)) 
   
acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = omega, f = f, g = g, beta = -1, gamma = 1)
                
edge_basis = acms.calc_edge_basis()
if edge_basis:
    acms.Assemble()
    usmall = acms.Solve() 

    gfu_acms = GridFunction(acms.Vc)

    acms.SetGlobalFunction(gfu_acms, usmall)
    
acms.PrintTiminigs(all = False)

Draw(gfu_acms.real, mesh, "uacms.real")
