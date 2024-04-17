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
    quadUL = MoveTo(-r/2,r/2).RectangleC(r, r).Face()
    quadLR = MoveTo(r/2,-r/2).RectangleC(r, r).Face()
    quadLL = MoveTo(-r/2,-r/2).RectangleC(r, r).Face()

    triangleUR = rect - quadUR 
    triangleUL = rect - quadUL
    triangleLR = rect - quadLR
    triangleLL = rect - quadLL

    circ.edges.name = "dom_bnd"
    outer = circ - rect

    shape = Glue([outer, triangleUR, triangleUL, triangleLR, triangleLL])
    # DrawGeo(shape)

    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh = maxH))
    mesh.Curve(10)
    # Draw(mesh)
    
    nmat = len(mesh.GetMaterials())
    nbnd = len(mesh.GetBoundaries())
    nvert = len(mesh.GetBBoundaries())

    for i in range(nmat):
        mesh.ngmesh.SetMaterial(i+1,"D" + str(i))

    dom_bnd = ""

    bi = 0
    for i in range(nbnd):
        if not "dom_bnd" in mesh.ngmesh.GetBCName(i): # != "dom_bnd":
            mesh.ngmesh.SetBCName(i,"E" + str(i))
        else:
            mesh.ngmesh.SetBCName(i,"dom_bnd_" + str(bi))
            dom_bnd += "dom_bnd_" + str(bi) + "|"
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

    # 0 = full crystal
    # 1 = air defect in crystal
    # 2 = air

    crystaltype = [["outer","outer"], 
                   ["outer","inner"],
                   ["inner", "inner"]]

    domain = [MoveTo(Lx*i,Ly*j).RectangleC(Lx,Ly).Face() for i in range(Nx) for j in range(Ny)]
    
    if incl ==1: #Circular inclusion
        inclusion = [MoveTo(0,0).Circle(Lx*i,Ly*j, r).Face() for i in range(Nx) for j in range(Ny)]
    else: #Square inclusion
        inclusion = [MoveTo(Lx*i,Ly*j).RectangleC(r, r).Face() for i in range(Nx) for j in range(Ny)]

    # outer = [domain[i*Ny+j]-inclusion[i*Ny+j] for i in range(Nx) for j in range(Ny)]
    # inner = [domain[i*Ny+j]*inclusion[i*Ny+j] for i in range(Nx) for j in range(Ny)]
    outer = []
    inner = []
    for i in range(Nx):
        for j in range(Ny):
            outerdom = domain[i*Ny+j]
            # outerdom.faces.name = "outer"+str(i*Ny+j)
            outerdom.faces.name = crystaltype[int(defects[i,j])][0]+str(i*Ny+j)
            outerdom = outerdom - inclusion[i*Ny+j]
            outerdom.faces.edges.Min(Y).name = "E_V"
            outerdom.faces.edges.Max(Y).name = "E_V"
            outerdom.faces.edges.Min(X).name = "E_H"
            outerdom.faces.edges.Max(X).name = "E_H"

            innerdom = domain[i*Ny+j]*inclusion[i*Ny+j]
            innerdom.faces.edges.name="inner_edge"+str(i*Ny+j)
            innerdom.faces.vertices.name="inner_vertex"+str(i*Ny+j)
            innerdom.faces.name=crystaltype[int(defects[i,j])][1]+str(i*Ny+j)

            if (j == 0) :
                outerdom.faces.edges.Min(Y).name = "dom_bnd_V"
            if (j == (Ny-1)) :
                outerdom.faces.edges.Max(Y).name = "dom_bnd_V"
            if (i == 0):
                outerdom.faces.edges.Min(X).name = "dom_bnd_H"
            if (i == (Nx-1)) :
                outerdom.faces.edges.Max(X).name = "dom_bnd_H"
            
            if layers > 0:
                if (j == layers) and (i >= layers) and (i <= Nx-1-layers):
                    outerdom.faces.edges.Min(Y).name = "crystal_bnd_bottom_V"
                if (j == (Ny-1-layers)) and (i >= layers) and (i <= Nx-1-layers) :
                    outerdom.faces.edges.Max(Y).name = "crystal_bnd_top_V"
                if (i == layers) and (j >= layers) and (j <= Ny-1-layers):
                    outerdom.faces.edges.Min(X).name = "crystal_bnd_left_H"
                if (i == (Nx-1-layers)) and (j >= layers) and (j <= Ny-1-layers) :
                    outerdom.faces.edges.Max(X).name = "crystal_bnd_right_H"
                
            outer.append(outerdom)
            inner.append(innerdom)

    
    outershapes = [out_dom for out_dom in outer]
    innershapes = [in_dom for in_dom in inner]
    
    crystalshape = Glue(outershapes + innershapes)
    mesh = Mesh(OCCGeometry(crystalshape, dim=2).GenerateMesh(maxh = maxH))
    mesh.Curve(10)
    

    nmat = len(mesh.GetMaterials())
    nbnd = len(mesh.GetBoundaries())
    nvert = len(mesh.GetBBoundaries())
    
                
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
    # definition of diffusion coefficient
    coeffs = {}
    # alpha_outer = 1/12.1 #SILICON
    # alpha_inner = 1 #AIR

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
        if "inner" in mesh.ngmesh.GetMaterial(d+1):
            mesh.ngmesh.SetMaterial(d+1, "outer" + str(d-int(nmat/2)))
    
    mesh_info = GetMeshinfo(mesh)
    mesh_info["dom_bnd"] = dom_bnd
    
    Draw(alpha, mesh, "alpha")
    
    # print(mesh.GetMaterials())
    # print(mesh.GetBoundaries())
    # print(mesh.GetBBoundaries())
    return mesh, dom_bnd, alpha, mesh_info



#################################################################
#################################################################
#################################################################
#################################################################



def problem_definition(problem, maxH, omega):

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
        
        
        
    elif problem == 5:  #Problem setting - PERIODIC CRYSTAL - Circular Inclusions
        
        r  = 0.126 # radius of inclusion
        Lx = 0.484 #"c"
        Ly = Lx #0.685 #"a"

        Nx = int(input("Number of cells on each direction: "))
        #20 # number of cells in x
        Ny = Nx # number of cells in y
        
        incl = 1 #circular
        alpha_outer = 1/12.1 #SILICON
        alpha_inner = 1 #0 #AIR

        
        layers = 1
        
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
        # quit()
        # omega = Lx /0.6
        kappa = omega #**2 * alpha 
        # k = kappa * CF((0.6,0.8)) #CF = CoefficientFunction
        # / 30.4878
        k_ext = omega #**2 # * alpha=1
        k = k_ext * CF((1,0)) #CF = CoefficientFunction
        beta = - k_ext / omega
        f = 0 
        g = 1j * (k_ext - k * specialcf.normal(2)) * exp(-1j * (k[0] * x + k[1] * y)) # Incoming plane wave 
        Draw(g, mesh, "g")
        u_ex = 0
        sol_ex = 0
        Du_ex = 0
        
        

    return mesh, dom_bnd, alpha, kappa, beta, f, g, sol_ex, u_ex, Du_ex, mesh_info


##################################################################
##################################################################
##################################################################
##################################################################



def ground_truth(mesh, dom_bnd, alpha, kappa, omega, beta, f, g, ord):
    #  RESOLUTION OF GROUND TRUTH SOLUTION
    #Computing the FEM solution /  ground truth solution with higher resolution
    
    # SetNumThreads(8)
    with TaskManager():
        V = H1(mesh, order = ord, complex = True)
        
        u, v = V.TnT()

        a = BilinearForm(V)
        a += alpha * grad(u) * grad(v) * dx() 
        a += - kappa**2 * u * v * dx()
        a += -1J * omega * beta * u * v * ds(dom_bnd, bonus_intorder = 10)
        a.Assemble()

        l = LinearForm(V)
        l += f * v * dx(bonus_intorder=10)
        l += g * v * ds(dom_bnd,bonus_intorder=10)
        l.Assemble()

        gfu_fem = GridFunction(V)
        ainv = a.mat.Inverse(V.FreeDofs(), inverse = "sparsecholesky")
        gfu_fem.vec.data = ainv * l.vec
        print("FEM finished")

        Draw(gfu_fem, mesh,"ufem")
        grad_fem = Grad(gfu_fem)
    
    return gfu_fem, grad_fem


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
    
    print("Energy = ", sqrt(Integrate(gfu_fem**2, mesh)))
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


##################################################################
##################################################################
##################################################################
##################################################################

 
 
def compute_acms_solution(mesh, V, acms, BM, EM):    
    
    gfu = GridFunction(V)
    
    setupstart = time.time()
    
    num = acms.acmsdofs #len(basis)
    print("finished setup", time.time() - setupstart)
    print("ndofs = ", num)
    
    invstart = time.time()
    asmall = acms.asmall
    print("calc asmall = ", time.time() - invstart)
    
    ainvsmall = Matrix(numpy.linalg.inv(asmall))
    f_small = acms.fsmall
    usmall = ainvsmall * f_small
    
    gfu.vec[:] = 0.0
    print("norm of usmall = ", Norm(usmall))

    Draw(gfu, mesh, "uacms")
    print("finished_acms")

        
    return gfu, num

##################################################################
##################################################################
##################################################################
##################################################################

 
 
 
  
def acms_solution(mesh, dom_bnd, alpha, Bubble_modes, Edge_modes, order_v, kappa, omega, beta, f, g, u_ex, Du_ex, mesh_info):
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
    max_bm = Bubble_modes[-1]
    max_em = Edge_modes[-1]

        
    # SetNumThreads(1)
    with TaskManager():
        for order in order_v:
            print(order)
            
            #FEM solution with same order of approximation
            gfu_fem, grad_fem = ground_truth(mesh, dom_bnd, alpha, kappa, omega, beta, f, g, order)
            
            V = H1(mesh, order = order, complex = True)
            ndofs.append(V.ndof)

            Iu = GridFunction(V) #Nodal interpolant            
            
            if V.ndof < 1000000:
                
                #Computing full basis with max number of modes 
                # bi = bonus int order - should match the curved mesh order
                acms = ACMS(order = order, mesh = mesh, bm = max_bm, em = max_em, bi = mesh.GetCurveOrder(), mesh_info = mesh_info, alpha = alpha, omega = omega, kappa = kappa, f = f, g = g)

                start = time.time()
                acms.CalcHarmonicExtensions(kappa = kappa)
                
                edges_time = time.time() 
                acms.calc_edge_basis()
                print("Edge basis functions computation in --- %s seconds ---" % (time.time() - edges_time))
                print("time to compute harmonic extensions = ", time.time() - start)

                assemble_start = time.time()
                for m in acms.doms:
                    acms.Assemble_localA(m)
                print("assemble = ", time.time() - assemble_start)
                            
                for EM in Edge_modes:
                        for BM in Bubble_modes:
                            if (EM <= acms.edge_modes) and (BM <= acms.bubble_modes):
                                gfu, num = compute_acms_solution(mesh, V, acms, BM, EM)
                                dofs.append(num)
                                l2_error, l2_error_ex, h1_error, h1_error_ex = append_acms_errors(mesh, gfu, gfu_fem, u_ex, grad_fem, Du_ex, l2_error, l2_error_ex, h1_error, h1_error_ex)

                l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex = append_NI_FEM_errors(mesh, gfu_fem, u_ex, Du_ex, Iu, l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex)
           
            else:
                for EM in Edge_modes:
                    for BM in Bubble_modes:
                        l2_error, l2_error_ex, h1_error, h1_error_ex = append_acms_errors(mesh, 0, 0, 0, 0, 0, l2_error, l2_error_ex, h1_error, h1_error_ex)

                l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex = append_NI_FEM_errors(mesh, 0, 0, 0, 0, l2_error_NodInt, h1_error_NodInt, l2_error_FEMex, h1_error_FEMex)
                
    
    print(l2_error)   
    print(l2_error_ex)
    print(h1_error)
    print(h1_error_ex)
    print(l2_error_NodInt)
    print(h1_error_NodInt)
    print(l2_error_FEMex)
    print(l2_error_FEMex)
    
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
    
    solution_dictionary = {
        'gfu_acms':  gfu,
        'gfu_fem' :  gfu_fem,
        'grad_fem':  grad_fem
    }
    

    return ndofs, dofs, errors_dictionary, solution_dictionary





##################################################################
##################################################################
##################################################################
##################################################################





def error_table_save(maxH, problem, order_v, Bubble_modes, Edge_modes, mesh, kappa, errors_dictionary, ndofs, dofs, u_ex, sol_ex, gfu_fem, grad_fem):
    plot_error = 0
    
    # Save both H1 and H1-relative errors on file named "file_name.npz" 
    dim = (len(order_v), len(Edge_modes), len(Bubble_modes))
    dictionary = {
        1            : ["The keys are: meshsize, order, bubbles, edges, vertices, problem, wavenumber."],
        'meshsize'   : ["The mesh size is", maxH],
        'order'      : ["The order of approximation is",  order_v],
        'bubbles'    : ["The number of bubble functions is", Bubble_modes],
        'edges'      : ["The number of edge modes is", Edge_modes],
        'vertices'   : ["The number of vertices is", mesh.nv],
        'problem'    : ["Chosen problem", problem],
        "wavenumber" : ["Chosen wavenumber is", kappa]
    }
    
    
    if sol_ex == 0:
        print("Error with FEM of order max as ground truth solution")
        file_name = create_error_file(problem, kappa, maxH, order_v, Bubble_modes, Edge_modes, 0)
        Errors = save_error_file(file_name, dictionary, mesh, errors_dictionary["l2_error"], errors_dictionary["h1_error"], dim, ndofs, dofs, gfu_fem, grad_fem)
        convergence_plots(plot_error, dofs, errors_dictionary["h1_error"], mesh, Edge_modes, Bubble_modes, order_v)
        
    elif sol_ex == 1:
        print("Error with exact solution")
        Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y))) #If we have analytical solution defined
        #Error with FEM
        file_name = create_error_file(problem, kappa, maxH, order_v, Bubble_modes, Edge_modes, 1)
        Errors = save_error_file_exact(file_name, dictionary, mesh, errors_dictionary, dim, ndofs, dofs, u_ex, Du_ex, gfu_fem, grad_fem)
        convergence_plots(plot_error, dofs, errors_dictionary["h1_error_ex"], mesh, Edge_modes, Bubble_modes, order_v)

    return file_name, Errors
     
        
        
        
        
#  def main(maxH, problem, omega, order_v, Bubble_modes, Edge_modes):
#     plot_error = 0
#     # Variables setting
#     mesh, dom_bnd, alpha, kappa, beta, f, g, sol_ex, u_ex, Du_ex, mesh_info = problem_definition(problem, maxH, omega)
#     # Draw(mesh)

#     # Compute ground truth solution with FEM of order max on the initialised mesh
#     gfu_fem, grad_fem = ground_truth(mesh, dom_bnd, alpha, kappa, omega, beta, f, g, order_v[-1])

#     # Solve ACMS system and compute H1 error
#     ndofs, dofs, errors_dictionary = acms_solution(mesh, dom_bnd, alpha, Bubble_modes, Edge_modes, order_v, kappa, omega, beta, f, g, gfu_fem, u_ex, Du_ex, mesh_info)    
    
    
#     # Save both H1 and H1-relative errors on file named "file_name.npz" 
#     # It needs to be loaded to be readable
#     dim = (len(order_v), len(Edge_modes), len(Bubble_modes))
#     dictionary = {
#         1            : ["The keys are: meshsize, order, bubbles, edges, vertices, problem, wavenumber."],
#         'meshsize'   : ["The mesh size is", maxH],
#         'order'      : ["The order of approximation is",  order_v],
#         'bubbles'    : ["The number of bubble functions is", Bubble_modes],
#         'edges'      : ["The number of edge modes is", Edge_modes],
#         'vertices'   : ["The number of vertices is", mesh.nv],
#         'problem'    : ["Chosen problem", problem],
#         "wavenumber" : ["Chosen wavenumber is", kappa]
#     }
    
    
#     if sol_ex == 0:
#         print("Error with FEM of order max as ground truth solution")
#         file_name = create_error_file(problem, kappa, maxH, order_v, Bubble_modes, Edge_modes, 0)
#         Errors = save_error_file(file_name, dictionary, mesh, errors_dictionary["l2_error"], errors_dictionary["h1_error"], dim, ndofs, dofs, gfu_fem, grad_fem)
#         convergence_plots(plot_error, dofs, errors_dictionary["h1_error"], mesh, Edge_modes, Bubble_modes, order_v)
        
#     elif sol_ex == 1:
#         print("Error with exact solution")
#         Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y))) #If we have analytical solution defined
#         #Error with FEM
#         file_name = create_error_file(problem, kappa, maxH, order_v, Bubble_modes, Edge_modes, 1)
#         Errors = save_error_file_exact(file_name, dictionary, mesh, errors_dictionary, dim, ndofs, dofs, u_ex, Du_ex, gfu_fem, grad_fem)
#         convergence_plots(plot_error, dofs, errors_dictionary["h1_error_ex"], mesh, Edge_modes, Bubble_modes, order_v)

#     return file_name, Errors
     
        
       
##################################################################
##################################################################
##################################################################
##################################################################


def convergence_plots(plot_error, dofs, h1_error, mesh, Edge_modes, Bubble_modes, order_v):

    ## Convergence plots

    if plot_error ==1:

        h1_error = np.reshape(h1_error, (len(order_v)*len(Edge_modes), len(Bubble_modes)))
        dofs = np.reshape(dofs, (len(order_v)*len(Edge_modes), len(Bubble_modes)))


        #Bubbles
        plt.rcParams.update({'font.size':15})
        for p in range(len(order_v)):
            for i in range(len(Edge_modes)):
                plt.loglog(Bubble_modes, h1_error[p*len(Edge_modes) + i,:], label=('Edge modes=%i' %Edge_modes[i]))
        plt.title('$H^1$ errors: increased bubbles deg=%i' %p)
        plt.legend()
        plt.xlabel('Bubbles')

        #Edges
        plt.rcParams.update({'font.size':15})
        for p in range(len(order_v)):
            for i in range(len(Bubble_modes)):
                plt.loglog(Edge_modes, h1_error[p*len(Edge_modes):(p+1)*len(Edge_modes),i], label=('Bubbles=%i' %Bubble_modes[i]))
        plt.title('$H^1$ errors: increased edge modes deg=%i' %p)
        plt.legend()
        plt.xlabel('Edge modes')

        plt.show()