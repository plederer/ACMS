# LIBRARIES

from ngsolve import *
from netgen.geom2d import *

import numpy 
import scipy.linalg
import scipy.sparse as sp

from netgen.occ import *
from helping_functions import *
from helmholtz_savefiles import *

# from ngsolve.webgui import Draw
import matplotlib.pyplot as plt

import time



##################################################################
##################################################################
##################################################################
##################################################################


def unit_disc(maxH):
    
    # GEOMETRY
    geo = SplineGeometry()
    Points = [(0,-1), (1,-1), (1,0), 
            (1,1), (0,1), (-1,1),
            (-1,0), (-1,-1), (0,0)]

    bcs_edge = ["c0", "c1", "c2", "c3", 
                "m0", "m1", "m2", "m3",
                "m4", "m5", "m6", "m7"]

    for i, pnt in enumerate(Points):
        geo.AddPoint(*pnt, name = "V" + str(i))

    geo.Append(["spline3", 0, 1, 2], leftdomain=1, rightdomain=0, bc="c0")
    geo.Append(["spline3", 2, 3, 4], leftdomain=2, rightdomain=0, bc="c1")
    geo.Append(["spline3", 4, 5, 6], leftdomain=3, rightdomain=0, bc="c2")
    geo.Append(["spline3", 6, 7, 0], leftdomain=4, rightdomain=0, bc="c3")
    geo.Append(["line", 0, 2], leftdomain=5, rightdomain=1, bc="m0")
    geo.Append(["line", 2, 4], leftdomain=6, rightdomain=2, bc="m1")
    geo.Append(["line", 4, 6], leftdomain=7, rightdomain=3, bc="m2")
    geo.Append(["line", 6, 0], leftdomain=8, rightdomain=4, bc="m3")

    geo.Append(["line", 8, 0], leftdomain=5, rightdomain=8, bc="m4")
    geo.Append(["line", 8, 2], leftdomain=6, rightdomain=5, bc="m5")
    geo.Append(["line", 8, 4], leftdomain=7, rightdomain=6, bc="m6")
    geo.Append(["line", 8, 6], leftdomain=8, rightdomain=7, bc="m7")

    # geo = SplineGeometry()
    # geo.AddCircle ( (0, 0), r=1, leftdomain=1, rightdomain=0, )

    ngmesh = geo.GenerateMesh(maxh = maxH)

    mesh = Mesh(ngmesh)
    mesh.Curve(order = 10)
    for i in range(8):
        mesh.ngmesh.SetMaterial(i+1,"om" + str(i))

    Draw(mesh)
    print(mesh.nv)
    # print(mesh.GetMaterials())
    # print(mesh.GetBoundaries())
    # print(mesh.GetBBoundaries())

    dom_bnd = "c0|c1|c2|c3"

    return mesh, dom_bnd


##################################################################
##################################################################
##################################################################
##################################################################



def problem_definition(problem, omega):

    if problem ==1:  #Problem setting - PLANE WAVE

        # omega = float(input("Wavenumber k: "))
        kappa = omega
        k = kappa * CF((0.6,0.8)) #CF = CoefficientFunction
        beta = 1
        f = 0
        u_ex = exp(-1J * (k[0] * x + k[1] * y))
        g = -1j * (k[0] * x + k[1] * y) * u_ex - 1j *beta * kappa * u_ex
        Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y)))
        sol_ex = 1

    elif problem == 2:  #Problem setting - INTERIOR SOURCE
            
        class Point: 
            def __init__(self):
                self.x = 0
                self.y = 0

        # omega = 1
        kappa = omega
        beta = 1
        p = (1/3, 1/3)
        P = Point()
        P.x = 1/3
        P.y = 1/3
        f = exp(-200 * ((x-P.x)**2 + (y-P.y)**2)) 
        g = 0
        u_ex = 0
        sol_ex = 0





    elif problem == 3:  #Problem setting - BOUNDARY SOURCE
            
        class Point: 
            def __init__(self):
                self.x = 0
                self.y = 0

        # omega = 16
        kappa = omega
        beta = 1
        p = (-1/sqrt(2), 1/sqrt(2))
        P = Point()
        P.x = - 1/sqrt(2)
        P.y = 1/sqrt(2)
        f = 0 
        g = exp(-200 * ((x-P.x)**2 + (y-P.y)**2))
        u_ex = 0
        sol_ex = 0

    return kappa, beta, f, g, sol_ex, u_ex


##################################################################
##################################################################
##################################################################
##################################################################



def ground_truth(mesh, dom_bnd, kappa, omega, beta, f, g, ord):
    #  RESOLUTION OF GROUND TRUTH SOLUTION
    #Computing the FEM solution /  ground truth solution with higher resolution
    
    SetNumThreads(8)
    with TaskManager():
        V = H1(mesh, order = 3, complex = True)
        u, v = V.TnT()

        a = BilinearForm(V)
        a += grad(u) * grad(v) * dx() 
        a += - kappa**2 * u * v * dx()  
        a += -1J * omega * beta * u * v * ds(dom_bnd)
        a.Assemble()

        l = LinearForm(V)
        l += f * v * dx(bonus_intorder=10)
        l += g * v * ds(dom_bnd,bonus_intorder=10)
        l.Assemble()

        gfu_fem = GridFunction(V)
        ainv = a.mat.Inverse(V.FreeDofs(), inverse = "sparsecholesky")
        gfu_fem.vec.data = ainv * l.vec
        print("FEM finished")

        grad_fem = Grad(gfu_fem)
    
    return gfu_fem, grad_fem


##################################################################
##################################################################
##################################################################
##################################################################



def compute_h1_error(gfu, grad_fem, mesh):
    #Computing error
    diff = grad_fem - Grad(gfu)
    h1_error_aux = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 15)) #Needs to do complex conjugate
    # Draw(gfu, mesh, "u_acms")
    h1_error_aux = h1_error_aux.real
    return h1_error_aux

##################################################################
##################################################################
##################################################################
##################################################################



def compute_l2_error(gfu, gfu_fem, mesh):
    #Computing error
    diff = gfu_fem - gfu
    l2_error_aux = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 15)) #Needs to do complex conjugate
    # Draw(gfu, mesh, "u_acms")
    l2_error_aux = l2_error_aux.real
    return l2_error_aux



##################################################################
##################################################################
##################################################################
##################################################################
  
def acms_solution(mesh, dom_bnd, Bubble_modes, Edge_modes, order_v, kappa, omega, beta, f, g, gfu_fem, sol_ex, u_ex):
    #  ACMS RESOLUTION

    l2_error = []
    l2_error_ex = []
    h1_error = []
    h1_error_ex = []
    l2_error_NodInt = []
    h1_error_NodInt = []
    dofs =[]
    ndofs = []
    max_bm = Bubble_modes[-1]
    max_em = Edge_modes[-1]

    # GROUND TRUTH SOLUTION
    grad_fem = Grad(gfu_fem)
    
    
    
    SetNumThreads(8)

    with TaskManager():
        for order in order_v:
            print(order)
            # start_time = time.time()
            V = H1(mesh, order = order, complex = True)
            
            Iu = GridFunction(V) #Nodal interpolant
            
            u, v = V.TnT()
            
            ndofs.append(V.ndof)
            
            if V.ndof < 1000000:
                
                a = BilinearForm(V)
                a += grad(u) * grad(v) * dx()
                a += - kappa**2 * u * v * dx()
                a += -1J * omega * beta * u * v * ds(dom_bnd, bonus_intorder = 10)
                a.Assemble()

                l = LinearForm(V)
                l += f * v * dx(bonus_intorder=10)
                l += g * v * ds(dom_bnd, bonus_intorder=10) #Could be increased with kappa (TBC)
                l.Assemble()

                gfu = GridFunction(V)
                #Computing full basis with max number of modes 
                # bi = bonus int order - should match the curved mesh order
                # print("max = ", max_em)
                acms = ACMS(order = order, mesh = mesh, bm = max_bm, em = max_em, bi = 10)
                acms.CalcHarmonicExtensions(kappa = kappa)
                acms.calc_basis()
            

                for EM in Edge_modes:

                        for BM in Bubble_modes:
                            #Vc = H1(mesh, order = order, complex = True)
                            # start_time = time.time()

                            if (EM <= acms.edge_modes) and (BM <= acms.bubble_modes):
                                gfu = GridFunction(V)
                                basis = MultiVector(gfu.vec, 0)

                                for bv in acms.basis_v:
                                    gfu.vec.FV()[:] = bv
                                    basis.Append(gfu.vec)

                                for e, label in enumerate(mesh.GetBoundaries()):
                                    for i in range(EM):
                                        gfu.vec.FV()[:] = acms.basis_e[e * acms.edge_modes + i]
                                        basis.Append(gfu.vec)


                                for d, dom in enumerate(mesh.GetMaterials()):
                                    for i in range(BM):
                                        gfu.vec.FV()[:] = acms.basis_b[d * acms.bubble_modes + i]
                                        basis.Append(gfu.vec)


                                num = len(basis)
                                dofs.append(num)

                                asmall = InnerProduct (basis, a.mat * basis, conjugate = False) #Complex
                                ainvsmall = Matrix(numpy.linalg.inv(asmall))
                                f_small = InnerProduct(basis, l.vec, conjugate = False)
                                usmall = ainvsmall * f_small
                                gfu.vec[:] = 0.0
                                gfu.vec.data = basis * usmall
                                print("finished_acms")

                                l2_error_aux = compute_l2_error(gfu, gfu_fem, mesh)
                                l2_error.append(l2_error_aux)
                                h1_error_aux = compute_h1_error(gfu, grad_fem, mesh)
                                h1_error.append(h1_error_aux)

                                if sol_ex == 1:
                                    Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y))) #If we have analytical solution defined
                                    l2_error_ex_aux = compute_l2_error(gfu,  u_ex, mesh)
                                    l2_error_ex.append(l2_error_ex_aux)
                                    h1_error_ex_aux = compute_h1_error(gfu, Du_ex, mesh)
                                    h1_error_ex.append(h1_error_ex_aux)
                            else:
                                l2_error.append(0)
                                h1_error.append(0)
                                dofs.append(V.ndof)
                                # ndofs.append(V.ndof)

                                if sol_ex == 1:
                                    l2_error_ex.append(0)
                                    h1_error_ex.append(0)
                            


                if sol_ex == 1:
                    Iu.Set(u_ex, dual=True)
                    l2_error_NodInt_aux = compute_l2_error(Iu,  u_ex, mesh)
                    l2_error_NodInt.append(l2_error_NodInt_aux)
                    h1_error_NodInt_aux = compute_h1_error(Iu, Du_ex, mesh)
                    h1_error_NodInt.append(h1_error_NodInt_aux)
                    
                    
            else:
                for EM in Edge_modes:
                    for BM in Bubble_modes:
                        l2_error.append(0)
                        h1_error.append(0)
                        dofs.append(V.ndof)
                        ndofs.append(V.ndof)

                        if sol_ex == 1:
                            l2_error_ex.append(0)
                            h1_error_ex.append(0)
                            
                if sol_ex == 1:
                    l2_error_NodInt.append(0)
                    h1_error_NodInt.append(0)
    
    
    return ndofs, dofs, Edge_modes, Bubble_modes, l2_error, l2_error_ex, h1_error, h1_error_ex, l2_error_NodInt, h1_error_NodInt





##################################################################
##################################################################
##################################################################
##################################################################





def main(maxH, problem, omega, order_v, Bubble_modes, Edge_modes):
    # Variables setting
    kappa, beta, f, g, sol_ex, u_ex = problem_definition(problem, omega)
    plot_error = 0
    
    
    # #Generate mesh: unit disco with 8 subdomains
    mesh, dom_bnd = unit_disc(maxH)


    # Compute ground truth solution with FEM of order 3 on the initialised mesh
    
    gfu_fem, grad_fem = ground_truth(mesh, dom_bnd, kappa, omega, beta, f, g, order_v[-1])
    
    # Solve ACMS system and compute H1 error
    ndofs, dofs, Edge_modes, Bubble_modes, l2_error_fem, l2_error_ex, h1_error_fem, h1_error_ex, l2_error_NodInt, h1_error_NodInt = acms_solution(mesh, dom_bnd, Bubble_modes, Edge_modes, order_v, kappa, omega, beta, f, g, gfu_fem, sol_ex, u_ex)    
    
    # Save both H1 and H1-relative errors on file named "file_name.npz" 
    # It needs to be loaded to be readable
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
        print("Error with FEM of order 3 as ground truth solution")
        file_name = create_error_file(problem, kappa, maxH, order_v, Bubble_modes, Edge_modes, 0)
        Errors = save_error_file(file_name, dictionary, mesh, l2_error_fem, h1_error_fem, dim, ndofs, dofs, gfu_fem, grad_fem)
        convergence_plots(plot_error, dofs, h1_error_fem, mesh, Edge_modes, Bubble_modes, order_v)
        
    elif sol_ex == 1:
        print("Error with exact solution")
        Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y))) #If we have analytical solution defined
        #Error with FEM
        l2_error_FEMex = compute_l2_error(gfu_fem,  u_ex, mesh)
        h1_error_FEMex = compute_h1_error(gfu_fem, Du_ex, mesh)
        file_name = create_error_file(problem, kappa, maxH, order_v, Bubble_modes, Edge_modes, 1)
        Errors = save_error_file_exact(file_name, dictionary, mesh, l2_error_ex, h1_error_ex, l2_error_fem, h1_error_fem, l2_error_FEMex, h1_error_FEMex, l2_error_NodInt, h1_error_NodInt, dim, ndofs, dofs, u_ex, Du_ex, gfu_fem, grad_fem)
        convergence_plots(plot_error, dofs, h1_error_ex, mesh, Edge_modes, Bubble_modes, order_v)

    return file_name, Errors
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

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