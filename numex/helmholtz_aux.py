# LIBRARIES

from ngsolve import *
from netgen.geom2d import *


import numpy 
import scipy.linalg
import scipy.sparse as sp

from netgen.occ import *
# from ngsolve.webgui import Draw
# from netgen.webgui import Draw as DrawGeo

from helping_functions import *
from helmholtz_savefiles import *

# import matplotlib.pyplot as plt

import time



##################################################################
##################################################################
##################################################################
##################################################################

def GetMeshinfo(mesh):
    dir_edges = ""
    edge_basis = []

    for e in range(len(mesh.GetBoundaries())):
        if not "inner_edge" in mesh.ngmesh.GetBCName(e):
            dir_edges += mesh.ngmesh.GetBCName(e) + "|"
            edge_basis.append((e, mesh.ngmesh.GetBCName(e)))

    dir_edges = dir_edges[:-1]
    
    
    vertex_basis = []
    for v in range(len(mesh.GetBBoundaries())):
        if not "inner_vertex" in mesh.ngmesh.GetCD2Name(v):
            vertex_basis.append((v, mesh.ngmesh.GetCD2Name(v)))

    return {"dir_edges": dir_edges, "verts": vertex_basis, "edges": edge_basis}


##################################################################
##################################################################


def unit_disc(maxH):
    
    l = sqrt(2) # Edge of square
    r = 1 # Circle radius

    circ = WorkPlane().Circle(0, 0, r).Face()
    rect = WorkPlane().Rotate(45).RectangleC(l, l).Face()
    quadUR = MoveTo(r/2,r/2).RectangleC(r, r).Face()
    # quadUR.edges[0] = 
    quadUL = MoveTo(-r/2,r/2).RectangleC(r, r).Face()
    quadLR = MoveTo(r/2,-r/2).RectangleC(r, r).Face()
    quadLL = MoveTo(-r/2,-r/2).RectangleC(r, r).Face()

    triangleUR = rect - quadUR 
    triangleUR.faces.edges[0].name = "test_D"
    triangleUR.faces.edges[2].name = "test_V"
    triangleUR.faces.edges[3].name = "test_H"
    triangleUR.faces.edges[1].name = "test_D"
    triangleUR.faces.edges[4].name = "test_D"

    
    triangleUL = rect - quadUL
    triangleUL.faces.edges[1].name = "test_H"
    triangleUL.faces.edges[3].name = "test_D"
    triangleLR = rect - quadLR
    triangleLR.faces.edges[4].name = "test_V"
    triangleLL = rect - quadLL

    circ.edges.name = "dom_bnd_C"
    outer = circ - rect

    shape = Glue([triangleUR, triangleUL, triangleLR, triangleLL, outer])
    # shape = triangleUR
    # DrawGeo(shape)

    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh = maxH))
    mesh.Curve(10)
    
    
    
    # Draw(mesh)
    # print(mesh.GetBoundaries())
    # V = H1(mesh, dirichlet = ".*")
    # gfu = GridFunction(V)
    # gfu.Set(1, definedon = mesh.Boundaries("dom_bnd_C"))
    # # gfu.Set(1, definedon = mesh.Materials("outer1"))
    # Draw(gfu)

    nmat = len(mesh.GetMaterials())
    nbnd = len(mesh.GetBoundaries())
    nvert = len(mesh.GetBBoundaries())

    for i in range(nmat):
        mesh.ngmesh.SetMaterial(i+1,"D" + str(i))

    dom_bnd = ""

    bi = 0
    for i in range(nbnd):
        if not "dom_bnd" in mesh.ngmesh.GetBCName(i): # != "dom_bnd":
            if "V" in mesh.ngmesh.GetBCName(i):
                mesh.ngmesh.SetBCName(i,"E_V" + str(i))
            elif "H" in mesh.ngmesh.GetBCName(i):
                mesh.ngmesh.SetBCName(i,"E_H" + str(i))
            else:
                mesh.ngmesh.SetBCName(i,"E_D" + str(i))
        else:
            mesh.ngmesh.SetBCName(i,"dom_bnd_C_" + str(bi))
            dom_bnd += "dom_bnd_C_" + str(bi) + "|"
            bi+=1
    dom_bnd = dom_bnd[:-1]
    
    for i in range(nvert):
        mesh.ngmesh.SetCD2Name(i+1,"V" + str(i))

        
    # print(mesh.nv)
    # print(mesh.GetMaterials()) # 8 subdomains
    # print(mesh.GetBoundaries()) # 12 edges
    # print(mesh.GetBBoundaries()) # 5 vertices

    alpha = 1 #mesh.MaterialCF(1, default=0)

    mesh_info = GetMeshinfo(mesh)
    mesh_info["dom_bnd"] = dom_bnd

    return mesh, dom_bnd, alpha, mesh_info




def crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer, alpha_inner, defects = np.ones((0,0)), layers = 0):
    
    if len(defects) == 0:
        defects = np.ones((Nx,Ny))

    #Crystal type: 0 = full crystal # 1 = air defect in crystal  # 2 = air
    crystaltype = [["outer","outer"], 
                   ["outer","inner"],
                   ["inner", "inner"]]

    domain = [MoveTo(Lx*i,Ly*j).RectangleC(Lx,Ly).Face() for i in range(Nx) for j in range(Ny)]
    
    if incl ==1: #Circular inclusion
        inclusion = [MoveTo(0,0).Circle(Lx*i,Ly*j, r).Face() for i in range(Nx) for j in range(Ny)]
                
                
    # elif incl == 2:
    #     inclusion = []
    #     for i in range(Nx):
    #         for j in range(Ny):
    #             Mx = Lx*i
    #             My = Ly * j
    #             c1 = MoveTo(0,0).Circle(Mx - Lx/4,My - Ly/4, r).Face()
    #             c2 = MoveTo(0,0).Circle(Mx + Lx/4,My - Ly/4, r).Face()
    #             c3 = MoveTo(0,0).Circle(Mx + Lx/4,My + Ly/4, r).Face()
    #             c4 = MoveTo(0,0).Circle(Mx - Lx/4,My + Ly/4, r).Face()
    #             inclusion.append(Glue([c1,c2,c3,c4]))
                
    elif incl >= 2: #Circular inclusions (incl on one side)
        inclusion = []
        for i in range(Nx):
            for j in range(Ny):   
                Mx = Lx*i
                My = Ly*j
                exponent = 2
                circular_incl = [MoveTo(0,0).Circle(Mx + (-1)**ee * Lx/incl * (kk + 1/2), My + (-1)**ff * Ly/incl * (ll + 1/2), r).Face()  for ee in range(exponent)  for ff in range(exponent) for kk in range(incl//2) for ll in range(incl//2)]
                inclusion.append(Glue([circ for circ in circular_incl]))

    else: #Square inclusion (incl = 0)
        inclusion = [MoveTo(Lx*i,Ly*j).RectangleC(r, r).Face() for i in range(Nx) for j in range(Ny)]
      
    # outer = [domain[i*Ny+j]-inclusion[i*Ny+j] for i in range(Nx) for j in range(Ny)]
    # inner = [domain[i*Ny+j]*inclusion[i*Ny+j] for i in range(Nx) for j in range(Ny)]
    outer = []
    inner = []

    def GetCell(i,j):
        outerdom = domain[i*Ny+j]
        # outerdom.faces.name = "outer"+str(i*Ny+j)
        outerdom.faces.name = crystaltype[int(defects[i,j])][0]+str(i*Ny+j)
        outerdom = outerdom - inclusion[i*Ny+j]
        outerdom.faces.edges.Min(Y).name = "E_H"
        outerdom.faces.edges.Max(Y).name = "E_H"
        outerdom.faces.edges.Min(X).name = "E_V"
        outerdom.faces.edges.Max(X).name = "E_V"

        innerdom = domain[i*Ny+j]*inclusion[i*Ny+j]
        innerdom.faces.edges.name="inner_edge"+str(i*Ny+j)
        innerdom.faces.vertices.name="inner_vertex"+str(i*Ny+j)
        innerdom.faces.name=crystaltype[int(defects[i,j])][1]+str(i*Ny+j)

        if (j == 0) :
            outerdom.faces.edges.Min(Y).name = "dom_bnd_H"
        if (j == (Ny-1)) :
            outerdom.faces.edges.Max(Y).name = "dom_bnd_H"
        if (i == 0):
            outerdom.faces.edges.Min(X).name = "dom_bnd_V"
        if (i == (Nx-1)) :
            outerdom.faces.edges.Max(X).name = "dom_bnd_V"
        
        if layers > 0:
            if (j == layers) and (i >= layers) and (i <= Nx-1-layers):
                outerdom.faces.edges.Min(Y).name = "crystal_bnd_bottom_H"
            if (j == (Ny-1-layers)) and (i >= layers) and (i <= Nx-1-layers) :
                outerdom.faces.edges.Max(Y).name = "crystal_bnd_top_H"
            if (i == layers) and (j >= layers) and (j <= Ny-1-layers):
                outerdom.faces.edges.Min(X).name = "crystal_bnd_left_V"
            if (i == (Nx-1-layers)) and (j >= layers) and (j <= Ny-1-layers) :
                outerdom.faces.edges.Max(X).name = "crystal_bnd_right_V"

        return innerdom, outerdom
    
    
    for i in range(layers, Nx-layers):
        for j in range(layers, Ny-layers):
            innerdom, outerdom = GetCell(i,j)
            outer.append(outerdom)
            inner.append(innerdom)
    
    if layers > 0:

        for i in [ii for ii in range(0, layers)] + [ii for ii in range(Nx-layers,Nx)]:
            for j in range(0,Ny):
                innerdom, outerdom = GetCell(i,j)
                outer.append(outerdom)
                inner.append(innerdom)
        
        for j in [jj for jj in range(0, layers)] + [jj for jj in range(Ny-layers,Ny)]:
            for i in range(layers,Nx-layers):
                innerdom, outerdom = GetCell(i,j)
                outer.append(outerdom)
                inner.append(innerdom)
        
    
    outershapes = [out_dom for out_dom in outer]
    innershapes = [in_dom for in_dom in inner]

    crystalshape = Glue(outershapes + innershapes)
    mesh = Mesh(OCCGeometry(crystalshape, dim=2).GenerateMesh(maxh = maxH))
    mesh.Curve(10)
    Draw(mesh)
    # input()
    
    nmat = len(mesh.GetMaterials())
    nbnd = len(mesh.GetBoundaries())
    nvert = len(mesh.GetBBoundaries())
    
    # print(mesh.GetBoundaries())
    # V = H1(mesh, dirichlet = ".*")
    # gfu = GridFunction(V)
    # gfu.Set(1, definedon = mesh.Boundaries("dom_bnd_H"))
    # Draw(gfu)
    # input()
    
    
    dom_bnd = ""
    bi = 0
    for i in range(nbnd): #
        if not "dom_bnd" in mesh.ngmesh.GetBCName(i): # != "dom_bnd":
            if not "inner_edge" in mesh.ngmesh.GetBCName(i): # != "dom_bnd":
                if not "crystal_bnd" in mesh.ngmesh.GetBCName(i):
                    if "V" in mesh.ngmesh.GetBCName(i):
                        mesh.ngmesh.SetBCName(i,"E_V" + str(i))
                    else:
                        mesh.ngmesh.SetBCName(i,"E_H" + str(i))
                else:
                    name = mesh.ngmesh.GetBCName(i)
                    # print(name)
                    mesh.ngmesh.SetBCName(i,name + str(i))
        else:
            if "V" in mesh.ngmesh.GetBCName(i):
                mesh.ngmesh.SetBCName(i,"dom_bnd_V" + str(bi))
                dom_bnd += "dom_bnd_V" + str(bi) + "|"
            else:
                mesh.ngmesh.SetBCName(i,"dom_bnd_H" + str(bi))
                dom_bnd += "dom_bnd_H" + str(bi) + "|"
            
            bi+=1
    dom_bnd = dom_bnd[:-1]
    
    for i in range(nvert): #Removing vertices on circles
        if not "inner_vertex" in mesh.ngmesh.GetCD2Name(i):
            mesh.ngmesh.SetCD2Name(i+1,"V" + str(i))

    Draw(mesh)
    
    
    
    
    
    

    # ########################
    # definition of diffusion coefficient: alpha_outer = 1/12.1 #SILICON  # alpha_inner = 1 #AIR
    coeffs = {}

    for d in range(len(mesh.GetMaterials())):
        dom_name = mesh.ngmesh.GetMaterial(d+1) 
        if "outer" in dom_name:
            coeffs[dom_name] = alpha_outer
        else:
            coeffs[dom_name] = alpha_inner

    alpha_cf = mesh.MaterialCF(coeffs, default=0)
    
    alpha = GridFunction(L2(mesh, order = 0))
    alpha.Set(alpha_cf)
    
    
    # ########################
    # rename inner domains give them the same name as the outer one has  inner name just used 
    for d in range(nmat):
        if "inner" in mesh.ngmesh.GetMaterial(d+1): # the +1 comes from the asking the negten mesh instead of the ngsolve mesh!
            offset = int(nmat/(1+incl**2))
            ii = int((d-offset)/incl**2) 
            mesh.ngmesh.SetMaterial(d+1, mesh.ngmesh.GetMaterial(ii+1))
    
    mesh_info = GetMeshinfo(mesh)
    mesh_info["dom_bnd"] = dom_bnd
    
    
    # print(mesh.GetMaterials())
    # V = H1(mesh, dirichlet = ".*")
    # gfu = GridFunction(V)
    # gfu.Set(1, definedon = mesh.Materials("outer0"))
    # Draw(gfu)
    # input()
    
    
    Draw(alpha, mesh, "alpha")
    
    
    return mesh, dom_bnd, alpha, mesh_info



#################################################################
#################################################################
# V = H1(mesh, dirichlet = ".*")
    # gfu = GridFunction(V)
    # # gfu.Set(1, definedon = mesh.Boundaries("crystal_bnd_bottom_H"))
    # gfu.Set(1, definedon = mesh.Materials("outer1"))
    # Draw(gfu)
    # alpha = 1   
    # 
    # # elif incl == 2:
    #     inclusion = []
    #     for i in range(Nx):
    #         for j in range(Ny):
    #             Mx = Lx*i
    #             My = Ly * j
    #             c1 = MoveTo(0,0).Circle(Mx - Lx/4,My - Ly/4, r).Face()
    #             c2 = MoveTo(0,0).Circle(Mx + Lx/4,My - Ly/4, r).Face()
    #             c3 = MoveTo(0,0).Circle(Mx + Lx/4,My + Ly/4, r).Face()
    #             c4 = MoveTo(0,0).Circle(Mx - Lx/4,My + Ly/4, r).Face()
    #             inclusion.append(Glue([c1,c2,c3,c4]))
                # c1 = [MoveTo(0,0).Circle(Mx - Lx/incl * (kk + 1/2), My - Ly/incl * (ll + 1/2), r).Face()  for kk in range(incl//2) for ll in range(incl//2)]
                # c2 = [MoveTo(0,0).Circle(Mx - Lx/incl * (kk + 1/2), My + Ly/incl * (ll + 1/2), r).Face()  for kk in range(incl//2) for ll in range(incl//2)]
                # c3 = [MoveTo(0,0).Circle(Mx + Lx/incl * (kk + 1/2), My - Ly/incl * (ll + 1/2), r).Face()  for kk in range(incl//2) for ll in range(incl//2)]
                # c4 = [MoveTo(0,0).Circle(Mx + Lx/incl * (kk + 1/2), My + Ly/incl * (ll + 1/2), r).Face()  for kk in range(incl//2) for ll in range(incl//2)]
                
                # aux1 = Glue([circ for circ in c1])
                # aux2 = Glue([circ for circ in c2])
                # aux3 = Glue([circ for circ in c3])
                # aux4 = Glue([circ for circ in c4])
                # inclusion.append(Glue([aux1, aux2, aux3, aux4]))
#################################################################
#################################################################



def problem_definition(problem, incl, maxH, omega, Bubble_modes, Edge_modes, order_v):

    if problem ==1:  #Problem setting - PLANE WAVE
        # #Generate mesh: unit disco with 8 subdomains
        mesh, dom_bnd, alpha, mesh_info = unit_disc(maxH)
        kappa = omega
        k = kappa * CF((0.6,0.8)) #CF = CoefficientFunction
        beta = 1
        f = 0
        u_ex = exp(-1J * (k[0] * x + k[1] * y))
        g = -1j * (k[0] * x + k[1] * y) * u_ex - 1j *beta * kappa * u_ex
        Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y)))
        sol_ex = 1
        gamma = 1
    

    elif problem == 2:  #Problem setting - INTERIOR SOURCE
        # #Generate mesh: unit disco with 8 subdomains
        mesh, dom_bnd, alpha, mesh_info = unit_disc(maxH) 
        class Point: 
            def __init__(self):
                self.x = 0
                self.y = 0
        kappa = omega
        beta = 1
        P = Point() # p = (1/3, 1/3)
        P.x = 1/3
        P.y = 1/3
        f = exp(-200 * ((x-P.x)**2 + (y-P.y)**2)) 
        g = 0
        u_ex = 0
        sol_ex = 0
        Du_ex = 0
        gamma = 1

    elif problem == 3:  #Problem setting - BOUNDARY SOURCE
        # #Generate mesh: unit disco with 8 subdomains
        mesh, dom_bnd, alpha, mesh_info = unit_disc(maxH) 
        class Point: 
            def __init__(self):
                self.x = 0
                self.y = 0
        kappa = omega  # omega = 16
        beta = 1
        P = Point() #p = (-1/sqrt(2), 1/sqrt(2))
        P.x = - 1/sqrt(2)
        P.y = 1/sqrt(2)
        f = 0 
        g = exp(-200 * ((x-P.x)**2 + (y-P.y)**2))
        u_ex = 0
        sol_ex = 0
        Du_ex = 0
        gamma = 1


    elif problem == 4:  #Problem setting - PERIODIC CRYSTAL - Squared Inclusions
        r  = 0.05  # radius of inclusion

        Nx = 9 # number of cells in x
        Ny = 9 # number of cells in y

        Lx = 1/9 
        Ly = 1/9
        
        incl = 0 # squared
        alpha_outer = 1
        alpha_inner = 12
        
        mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer, alpha_inner)
        
        class Point: 
            def __init__(self):
                self.x = 0
                self.y = 0
        P = Point() # p = (0, 1/2)
        P.x = -Lx/2
        P.y = (Ny-1)/2 * Ly

        # omega = 10
        kappa = omega
        k = kappa * CF((1,0)) #CF = CoefficientFunction
        beta = 1 
        f = 0 
        g = exp(-1J * (k[0] * x + k[1] * y)) * exp(-100 * ((x-P.x)**2 + (y-P.y)**2)) #Paper test
        Draw(g, mesh, "g")
        u_ex = 0
        sol_ex = 0
        Du_ex = 0
        gamma = 1
        
        
        
    elif problem == 5:  #Problem setting - PERIODIC CRYSTAL - Circular Inclusions
        
        r  = 0.126     # radius of inclusion
        Lx = 1 * incl   #* 0.484 #"c"
        Ly = Lx        #0.685 #"a
        Nx = 16 // incl # number of cells in x direction
        Ny = Nx        # number of cells in y direction
        alpha_outer = 1/12.1 #SILICON
        alpha_inner = 1 #0 #AIR        
        layers = 0 
        
        ix = [i for i in range(layers)] + [Nx - 1 - i for i in range(layers)]
        iy = [i for i in range(layers)] + [Ny - 1 - i for i in range(layers)]
        
        defects = np.ones((Nx,Ny))
        for i in ix: 
            for j in range(Ny): 
                defects[i,j] = 0.0
        
        for j in iy:
            for i in range(Nx): 
                defects[i,j] = 0.0

        mesh, dom_bnd, alpha, mesh_info = crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer, alpha_inner, defects, layers)
        
        # print(Integrate(x, mesh, definedon = mesh.Boundaries("measure_edge_left")))
        # print(Integrate(x, mesh, definedon = mesh.Boundaries("measure_edge_right")))
        # omega = Lx /0.6
        kappa = omega    #**2 * alpha 
        k_ext = omega    #**2 # * alpha=1
        k = k_ext * CF((1,0)) #CF = CoefficientFunction
        beta = - k_ext / omega
        f = 0 
        g = 1j * (k_ext - k * specialcf.normal(2)) * exp(-1j * (k[0] * x + k[1] * y)) # Incoming plane wave 
        Draw(g, mesh, "g")
        u_ex = 0
        sol_ex = 0
        Du_ex = 0
        gamma = 1
        
        
    dim = (len(order_v), len(Edge_modes), len(Bubble_modes))
        
    variables_dictionary = {
        'problem'      : problem,
        'alpha'        : alpha, 
        'omega'        : omega,
        'kappa'        : kappa, 
        'beta'         : beta, 
        'gamma'        : gamma, 
        'f'            : f, 
        'g'            : g,
        'sol_ex'       : sol_ex,
        'u_ex'         : u_ex,
        'Du_ex'        : Du_ex,
        'Bubble_modes' : Bubble_modes, 
        'Edge_modes'   : Edge_modes, 
        'order_v'      : order_v,
        'dim'          : dim,
        'vertices'     : mesh.nv,
        'meshsize'     : maxH
    }
    
    return mesh, dom_bnd, mesh_info, variables_dictionary
    # return mesh, dom_bnd, alpha, kappa, beta, gamma, f, g, sol_ex, u_ex, Du_ex, mesh_info


##################################################################
##################################################################
##################################################################
##################################################################



def ground_truth(mesh, dom_bnd, variables_dictionary, ord):
    #  RESOLUTION OF GROUND TRUTH SOLUTION
    #Computing the FEM solution /  ground truth solution with higher resolution
    
    alpha = variables_dictionary["alpha"]
    omega = variables_dictionary["omega"]
    kappa = variables_dictionary["kappa"]
    beta = variables_dictionary["beta"]
    gamma = variables_dictionary["gamma"]
    f = variables_dictionary["f"]
    g = variables_dictionary["g"]
    
    # SetNumThreads(16)
    with TaskManager():
        start = time.time()
        
        V = H1(mesh, order = ord, complex = True) 
        
        u, v = V.TnT()

        a = BilinearForm(V)
        a += alpha * grad(u) * grad(v) * dx() 
        a += - gamma * kappa**2 * u * v * dx()
        a += -1J * omega * beta * u * v * ds(dom_bnd, bonus_intorder = 10)
        a.Assemble()

        l = LinearForm(V)
        l += f * v * dx(bonus_intorder=10)
        l += g * v * ds(dom_bnd,bonus_intorder=10)
        l.Assemble()

        gfu_fem = GridFunction(V)
        ainv = a.mat.Inverse(V.FreeDofs(), inverse = "sparsecholesky")
        gfu_fem.vec.data = ainv * l.vec
        print("FEM computation = ", time.time() - start)

        Draw(gfu_fem, mesh,"ufem")
        grad_fem = Grad(gfu_fem)
        
    solution_dictionary = {
        'gfu_fem' :  gfu_fem,
        'grad_fem':  grad_fem
    }
    
    return solution_dictionary #gfu_fem, grad_fem


##################################################################
##################################################################
##################################################################
##################################################################

def compute_acms_solution(mesh, V, acms, edge_basis):    
    
    gfu = GridFunction(V)
    
    if (edge_basis == False):
        gfu = 0
        num = 0
    else:         
        setupstart = time.time()
        
        num = acms.acmsdofs #len(basis)
        print("finished setup", time.time() - setupstart)        
        invstart = time.time()
        asmall = acms.asmall
        print("calc asmall = ", time.time() - invstart)
        
        ainvsmall = Matrix(numpy.linalg.inv(asmall))
        f_small = acms.fsmall
        usmall = ainvsmall * f_small
        
        gfu.vec[:] = 0.0
        # print("norm of usmall = ", Norm(usmall))

        acms.SetGlobalFunction(gfu, usmall)
        Draw(gfu, mesh, "uacms")
     
        print("finished_acms")
    return gfu, num

##################################################################
##################################################################
# int_bnd_name = "crystal_bnd_right" #"dom_bnd"
        # # integral = acms.IntegrateACMS(bndname = "crystal_bnd_right", coeffs = usmall)
        # integral = acms.IntegrateACMS(bndname = int_bnd_name, coeffs = usmall)
        # print("myint = ", integral)
        
           # intval = 0
        # for edgename in mesh.GetBoundaries():
        #         if int_bnd_name in edgename:
        #             # print(edgename)
        #             intval += Integrate(gfu, mesh, definedon = mesh.Boundaries(edgename))
        #             # print(intval)
        # print("error integral = ", intval - integral)

##################################################################
##################################################################

 
 
 
  
def acms_main(mesh, mesh_info, dom_bnd, variables_dictionary, solution_dictionary):
    #  ACMS RESOLUTION

    l2_error = []
    l2_error_ex = []
    h1_error = []
    h1_error_ex = []
    l2_error_NodInt = []
    h1_error_NodInt = []
    l2_error_FEMex = []
    h1_error_FEMex = []
    dofs =[]
    ndofs = []
    
   
    Bubble_modes = variables_dictionary["Bubble_modes"]
    Edge_modes = variables_dictionary["Edge_modes"]
    order_v = variables_dictionary["order_v"]
    
    alpha = variables_dictionary["alpha"]
    omega = variables_dictionary["omega"]
    kappa = variables_dictionary["kappa"]
    beta = variables_dictionary["beta"]
    gamma = variables_dictionary["gamma"]
    f = variables_dictionary["f"]
    g = variables_dictionary["g"]
    u_ex = variables_dictionary["u_ex"]
    Du_ex = variables_dictionary["Du_ex"]
    
    gfu_fem = solution_dictionary["gfu_fem"]
    grad_fem = solution_dictionary["grad_fem"]
    
    # SetNumThreads(16)
    with TaskManager():
        for order in order_v:
            print("Order =",  order)      
              
            V = H1(mesh, order = order, complex = True)
            ndofs.append(V.ndof)

            Iu = GridFunction(V) #Nodal interpolant            
                        
            if V.ndof < 10000000:
                for EM in Edge_modes:
                    print("Edge modes =", EM)
                    for BM in Bubble_modes:
                        
                        start = time.time()
                        
                        #Computing full basis with max number of modes 
                        # bi = bonus int order - should match the curved mesh order
                        acms = ACMS(order = order, mesh = mesh, bm = BM, em = EM, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = kappa, f = f, g = g, beta = beta, gamma = gamma)
                        
                        edges_time = time.time() 
                        edge_basis = acms.calc_edge_basis()
                        print("Edge basis functions computation in --- %s seconds ---" % (time.time() - edges_time))
                        print("time to compute harmonic extensions = ", time.time() - start)
                        # print(edge_basis)
                        
                        if edge_basis:
                            start = time.time()
                            acms.CalcHarmonicExtensions()
                                                        
                            assemble_start = time.time()
                            for m in acms.doms:
                                acms.Assemble_localA(m)
                            print("assemble = ", time.time() - assemble_start)
        
                        gfu, num = compute_acms_solution(mesh, V, acms, edge_basis)
                        dofs.append(num)
                        l2_error, l2_error_ex, h1_error, h1_error_ex = append_acms_errors(mesh, gfu, gfu_fem, u_ex, grad_fem, Du_ex, l2_error, l2_error_ex, h1_error, h1_error_ex)

                l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex = append_NI_FEM_errors(mesh, gfu_fem, u_ex, Du_ex, Iu, l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex)
           
            else:
                for EM in Edge_modes:
                    for BM in Bubble_modes:
                        l2_error, l2_error_ex, h1_error, h1_error_ex = append_acms_errors(mesh, 0, 0, 0, 0, 0, l2_error, l2_error_ex, h1_error, h1_error_ex)

                l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex = append_NI_FEM_errors(mesh, 0, 0, 0, 0, l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex)
                
    
    # print("L2 error = ", l2_error)   
    # print("L2 error exact= ", l2_error_ex)
    # print("H1 error = ", h1_error)
    # print("H1 error exact= ", h1_error_ex)
    # print(l2_error_NodInt)
    # print(h1_error_NodInt)
    # print(l2_error_FEMex)
    # print(l2_error_FEMex)
    
    errors_dictionary = {
        'l2_error':        l2_error, 
        'l2_error_ex':     l2_error_ex, 
        'h1_error':        h1_error, 
        'h1_error_ex':     h1_error_ex, 
        'l2_error_NodInt': l2_error_NodInt, 
        'h1_error_NodInt': h1_error_NodInt, 
        'l2_error_FEMex':  l2_error_FEMex, 
        'h1_error_FEMex':  h1_error_FEMex
    }
    
    solution_dictionary.update({'gfu_acms':  gfu})
    variables_dictionary.update({'ndofs':  ndofs})
    variables_dictionary.update({'dofs':  dofs})
    
    return variables_dictionary, solution_dictionary, errors_dictionary





##################################################################
##################################################################
##################################################################
##################################################################



def compute_h1_error(gfu, grad_fem, mesh):
    if gfu == 0 or grad_fem == 0:
        h1_error_aux = 0
    else:
        #Computing error
        diff = grad_fem - Grad(gfu)
        h1_error_aux = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 15)) #Needs to do complex conjugate
        h1_error_aux = h1_error_aux.real
    return h1_error_aux

##################################################################
##################################################################
##################################################################
##################################################################



def compute_l2_error(gfu, gfu_fem, mesh):
    if gfu == 0 or gfu_fem == 0 :
        l2_error_aux = 0
    else:
        #Computing error
        diff = gfu_fem - gfu
        l2_error_aux = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 15)) #Needs to do complex conjugate
        l2_error_aux = l2_error_aux.real
    return l2_error_aux


##################################################################
##################################################################
##################################################################
##################################################################


def append_acms_errors(mesh, gfu, gfu_fem, u_ex, grad_fem, Du_ex, l2_error, l2_error_ex, h1_error, h1_error_ex):
    
    # print("Energy = ", sqrt(Integrate(gfu_fem**2, mesh)))
    l2_error_aux = compute_l2_error(gfu, gfu_fem, mesh)
    l2_error.append(l2_error_aux)
    h1_error_aux = compute_h1_error(gfu, grad_fem, mesh)
    h1_error.append(h1_error_aux)
                    
    l2_error_ex_aux = compute_l2_error(gfu,  u_ex, mesh)
    l2_error_ex.append(l2_error_ex_aux)
    h1_error_ex_aux = compute_h1_error(gfu, Du_ex, mesh)
    h1_error_ex.append(h1_error_ex_aux)
                                
    return l2_error, l2_error_ex, h1_error, h1_error_ex


##################################################################
##################################################################
##################################################################
##################################################################


def append_NI_FEM_errors(mesh, gfu_fem, u_ex, Du_ex, Iu, l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex):
    
    Iu.Set(u_ex, dual=True)
    l2_error_NodInt_aux = compute_l2_error(Iu,  u_ex, mesh)
    l2_error_NodInt.append(l2_error_NodInt_aux)
    h1_error_NodInt_aux = compute_h1_error(Iu, Du_ex, mesh)
    h1_error_NodInt.append(h1_error_NodInt_aux)
    #Error with FEM
    l2_error_FEMex_aux = compute_l2_error(gfu_fem,  u_ex, mesh)
    l2_error_FEMex.append(l2_error_FEMex_aux)
    h1_error_FEMex_aux = compute_h1_error(gfu_fem, Du_ex, mesh)
    h1_error_FEMex.append(h1_error_FEMex_aux)
                     
    return l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex






