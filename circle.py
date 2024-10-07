# LIBRARIES
from geometries import *
from acms_class import *
import numpy as np 

import netgen.gui

omega = 1

maxH = 0.2
order = 4
EE = 16

mesh, dom_bnd, alpha, mesh_info = unit_disc(maxH)

k = omega * CF((0.6,0.8))
f = 0
u_ex = exp(-1J * (k[0] * x + k[1] * y))
g = -1j * (k[0] * x + k[1] * y) * u_ex - 1j * omega * u_ex
   
acms = ACMS(order = order, mesh = mesh, bm = 0, em = EE, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = omega, f = f, g = g, beta = 1, gamma = 1)
                
edge_basis = acms.calc_edge_basis()
if edge_basis:
    acms.Assemble()
    usmall = acms.Solve() 

    gfu_acms = GridFunction(acms.Vc)

    acms.SetGlobalFunction(gfu_acms, usmall)
    
acms.PrintTiminigs(all = False)

Draw(u_ex.real, mesh, "u_ex.real")
Draw(u_ex.real - gfu_acms.real, mesh, "error.real")
Draw(gfu_acms.real, mesh, "uacms.real")
