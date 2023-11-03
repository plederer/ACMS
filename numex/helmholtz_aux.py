# LIBRARIES

from ngsolve import *
from netgen.geom2d import *

import numpy 
import scipy.linalg
import scipy.sparse as sp

from netgen.occ import *
from helping_functions import *

# from ngsolve.webgui import Draw
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path




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

    ngmesh = geo.GenerateMesh(maxh=maxH)

    mesh = Mesh(ngmesh)
    for i in range(8):
        mesh.ngmesh.SetMaterial(i+1,"om" + str(i))

    Draw(mesh)
    print(mesh.nv)
    print(mesh.GetMaterials())
    print(mesh.GetBoundaries())
    print(mesh.GetBBoundaries())

    dom_bnd = "c0|c1|c2|c3"

    return mesh, dom_bnd


##################################################################
##################################################################


def problem_definition(problem):

    if problem ==1:  #Problem setting - PLANE WAVE
        
        omega = 1
        kappa = omega
        k = kappa * CF((0.6,0.8)) #CF = CoefficientFunction
        beta = 1
        f = 0
        u_ex = exp(-1J * (k[0] * x + k[1] * y))
        g = -1j * kappa * (k[0] * x + k[1] * y) * u_ex - 1j *beta * u_ex
        Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y)))
        sol_ex = 1

    elif problem == 2:  #Problem setting - INTERIOR SOURCE
            
        class Point: 
            def __init__(self):
                self.x = 0
                self.y = 0

        omega = 1
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

        omega = 16
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

    return kappa, omega, beta, f, g, sol_ex, u_ex


##################################################################
##################################################################


def ground_truth(mesh, dom_bnd, kappa, omega, beta, f, g):
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

        gfu_ex = GridFunction(V)
        ainv = a.mat.Inverse(V.FreeDofs(), inverse = "sparsecholesky")
        gfu_ex.vec.data = ainv * l.vec
        print("FEM finished")

        grad_uex = Grad(gfu_ex)
    
    return gfu_ex, grad_uex


##################################################################
##################################################################


def compute_h1_error(gfu, grad_uex, mesh):
    #Computing error
    diff = grad_uex - Grad(gfu)
    h1_error_aux = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 10)) #Needs to do complex conjugate
    # Draw(gfu, mesh, "u_acms")
    h1_error_aux = h1_error_aux.real
    return h1_error_aux

##################################################################
##################################################################


def compute_l2_error(gfu, gfu_ex, mesh):
    #Computing error
    diff = gfu_ex - gfu
    l2_error_aux = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 10)) #Needs to do complex conjugate
    # Draw(gfu, mesh, "u_acms")
    l2_error_aux = l2_error_aux.real
    return l2_error_aux


##################################################################
##################################################################


def create_error_file(problem, maxH, order_v, Bubble_modes, Edge_modes, err_type):
    
    problem_dict = {
        1 : "PW",
        2 : "LIS",
        3 : "PerCrys"
    }

    err_type_dict = {
        0 : "FEMsol",
        1 : "EXACTsol"
    }

    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"L2-H1_errors_{err_type_dict[err_type]}_{problem_dict[problem]}_meshH{maxH}_o{order_v[-1]}_b{Bubble_modes[-1]}_e{Edge_modes[-1]}_{date_time}"
    # print(file_name)

    return file_name



def save_error_file(file_name, mesh, l2_error, h1_error, dim, gfu_ex, grad_uex):
    
    save_dir = Path('./Results') #Saves local folder name
    save_dir.mkdir(exist_ok=True) #Creates folder Results if it does not exists already
    file_path = save_dir.joinpath(file_name) # Full path where to save the results file (no .npy)
    
    # 3 dimensional vector with a matrix in bubbles-edges for each order 
    # dim = len(order_v), len(Edge_modes), len(Bubble_modes)))
    l2_error_3d = np.reshape(l2_error, (dim))
    h1_error_3d = np.reshape(h1_error, (dim))

    #Computing norm of the FEM solution to use for relative errors
    l2_norm_FEM = Integrate ( InnerProduct(gfu_ex, gfu_ex), mesh, order = 10)
    h1_norm_FEM = l2_norm_FEM +  Integrate ( InnerProduct(grad_uex,grad_uex), mesh, order = 10)
    l2_norm_FEM = sqrt(l2_norm_FEM.real)
    h1_norm_FEM = sqrt(h1_norm_FEM.real)

    #Relative errors
    l2_error_rel = np.dot(l2_error, 1/l2_norm_FEM)
    l2_error_rel_3d = np.reshape(l2_error_rel, (dim))
    h1_error_rel = np.dot(h1_error, 1/h1_norm_FEM)
    h1_error_rel_3d = np.reshape(h1_error_rel, (dim))


    # Save both 3d objects in the same file. They are assigned names: H1_FEM_error, H1_FEM_Relative_error
    np.savez(file_path, L2_error = l2_error_3d, L2_Relative_error = l2_error_rel_3d, H1_error = h1_error_3d, H1_Relative_error = h1_error_rel_3d)

    # Loading file to print errors
    Errors = np.load(save_dir.joinpath(file_name + ".npz"))
    print("L2 error")
    print(Errors['L2_error'])
    print("L2 relative error")
    print(Errors['L2_Relative_error'])
    print("H1 error")
    print(Errors['H1_error'])
    print("H1 relative error")
    print(Errors['H1_Relative_error'])

    return Errors


##################################################################
##################################################################
  


def acms_solution(mesh, dom_bnd, Bubble_modes, Edge_modes, order_v, kappa, omega, beta, f, g, gfu_ex, sol_ex, u_ex):
    #  ACMS RESOLUTION

    l2_error = []
    l2_error_ex = []
    h1_error = []
    h1_error_ex = []
    dofs =[]
    max_bm = Bubble_modes[-1]
    max_em = Edge_modes[-1]

    # GROUND TRUTH SOLUTION
    grad_uex = Grad(gfu_ex)
    
    SetNumThreads(8)

    with TaskManager():
        for order in order_v:
            print(order)
            V = H1(mesh, order = order, complex = True)
            u, v = V.TnT()


            a = BilinearForm(V)
            a += grad(u) * grad(v) * dx()
            a += - kappa**2 * u * v * dx()
            a += -1J * omega * beta * u * v * ds(dom_bnd)
            a.Assemble()

            l = LinearForm(V)
            l += f * v * dx(bonus_intorder=10)
            l += g * v * ds(dom_bnd, bonus_intorder=10)
            l.Assemble()

            gfu = GridFunction(V)
            #Computing full basis with max number of modes 
            acms = ACMS(order = order, mesh = mesh, bm = max_bm, em = max_em)
            acms.CalcHarmonicExtensions(kappa = kappa)
            acms.calc_basis()

            for EM in Edge_modes:
                    for BM in Bubble_modes:
                        #Vc = H1(mesh, order = order, complex = True)
                        gfu = GridFunction(V)
                        basis = MultiVector(gfu.vec, 0)

                        for bv in acms.basis_v:
                            gfu.vec.FV()[:] = bv
                            basis.Append(gfu.vec)

                        for e, label in enumerate(mesh.GetBoundaries()):
                            for i in range(EM):
                                gfu.vec.FV()[:] = acms.basis_e[e * max_em + i]
                                basis.Append(gfu.vec)


                        for d, dom in enumerate(mesh.GetMaterials()):
                            for i in range(BM):
                                gfu.vec.FV()[:] = acms.basis_b[d * max_bm + i]
                                basis.Append(gfu.vec)


                        num = len(basis)
                        dofs.append(num)


                        asmall = InnerProduct (basis, a.mat * basis, conjugate = False) #Complex

                        asmall_np = np.zeros((num, num), dtype=numpy.complex128)
                        asmall_np = asmall.NumPy()

                        # SetNumThreads(1)
                        ainvs_small_np = numpy.linalg.inv(asmall_np)

                        ainvsmall = Matrix(num,num,complex=True)


                        for i in range(num):
                            for j in range(num):
                                ainvsmall[i,j] = ainvs_small_np[i,j]

                        f_small = InnerProduct(basis, l.vec, conjugate = False)

                        usmall = ainvsmall * f_small
                        gfu.vec[:] = 0.0

                        gfu.vec.data = basis * usmall
                       # Draw(gfu-gfu_ex, mesh, "error")

                        print("finished_acms")
                        
                        l2_error_aux = compute_l2_error(gfu,  gfu_ex, mesh)
                        l2_error.append(l2_error_aux)
                        

                        h1_error_aux = compute_h1_error(gfu, grad_uex, mesh)
                        h1_error.append(h1_error_aux)

                        if sol_ex == 1:
                            Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y))) #If we have analytical solution defined
                            l2_error_ex_aux = compute_l2_error(gfu,  u_ex, mesh)
                            l2_error_ex.append(l2_error_ex_aux)
                            h1_error_ex_aux = compute_h1_error(gfu, Du_ex, mesh)
                            h1_error_ex.append(h1_error_ex_aux)
                            

    
    return gfu, l2_error, l2_error_ex, h1_error, h1_error_ex

 



##################################################################
##################################################################


def convergence_plots(plot_error, h1_error, mesh, Edge_modes, Bubble_modes,order_v):

    ## Convergence plots

    if plot_error ==1:

        h1_error = np.reshape(h1_error, (len(order_v)*len(Edge_modes), len(Bubble_modes)))
        dofs = np.reshape(dofs, (len(order_v)*len(Edge_modes), len(Bubble_modes)))


        #Bubbles
        plt.rcParams.update({'font.size':15})
        for d in range(len(order_v)):
            for i in range(len(Edge_modes)):
                plt.loglog(Bubble_modes, h1_error[d*len(Edge_modes) + i,:], label=('Edge modes=%i' %Edge_modes[i]))
        plt.title('$H^1$ errors: increased bubbles deg=%i' %order)
        plt.legend()
        plt.xlabel('Bubbles')

        #Edges
        plt.rcParams.update({'font.size':15})
        for d in range(len(order_v)):
            for i in range(len(Bubble_modes)):
                plt.loglog(Edge_modes, h1_error[d*len(Edge_modes):(d+1)*len(Edge_modes),i], label=('Bubbles=%i' %Bubble_modes[i]))
        plt.title('$H^1$ errors: increased edge modes deg=%i' %order)
        plt.legend()
        plt.xlabel('Edge modes')


###############################################
# ##############################################





def main(maxH, problem, order_v, Bubble_modes, Edge_modes):
    # Variables setting
    kappa, omega, beta, f, g, sol_ex, u_ex = problem_definition(problem)
    plot_error = 0

    # #Generate mesh: unit disco with 8 subdomains
    mesh, dom_bnd = unit_disc(maxH)


    # Compute ground truth solution with FEM of order 3 on the initialised mesh
    gfu_ex, grad_uex = ground_truth(mesh, dom_bnd, kappa, omega, beta, f, g)

    # Solve ACMS system and compute H1 error
    gfu, l2_error, l2_error_ex, h1_error, h1_error_ex = acms_solution(mesh, dom_bnd, Bubble_modes, Edge_modes, order_v, kappa, omega, beta, f, g, gfu_ex, sol_ex, u_ex)

    # Save both H1 and H1-relative errors on file named "file_name.npz" 
    # It needs to be loaded to be readable
    print("Error with FEM of order 3 as ground truth solution")
    file_name = create_error_file(problem, maxH, order_v, Bubble_modes, Edge_modes, 0)
    dim = (len(order_v), len(Edge_modes), len(Bubble_modes))
    Errors_FEM = save_error_file(file_name, mesh, l2_error, h1_error, dim, gfu_ex, grad_uex)
    
    # Plot H1 error
    convergence_plots(plot_error, h1_error, mesh, Edge_modes, Bubble_modes, order_v)
    
    # If available, the exact solution is used  (sol_ex == 1)  
    if sol_ex == 1:
        H1_error_ex = []
        Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y))) #If we have analytical solution defined
        print("Error with exact solution")
        file_name_exact = create_error_file(problem, maxH, order_v, Bubble_modes, Edge_modes, 1)
        Errors_exact = save_error_file(file_name_exact, mesh, l2_error_ex, h1_error_ex, dim, u_ex, Du_ex)

        # Plot H1 error
        convergence_plots(plot_error, h1_error_ex, mesh, Edge_modes, Bubble_modes, order_v)
    

    # return gfu_acms, H1_error, H1_error_ex
     
        
        




"""
Some content saved in comments:


#Saving the error on a file

    # h1_error_3d = np.reshape(h1_error, (len(order_v), len(Edge_modes), len(Bubble_modes)))
    # np.save('H1_error', h1_error_3d)
    # H1_error = np.load('H1_error.npy')
    # print(H1_error)

if sol_ex == 1:
        Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y)))
        grad_uex = Du_ex #If we have analytical solution defined
    elif sol_ex == 0:

    #Computing error with FEM solution
                        # diff = grad_uex - Grad(gfu)
                        # h1_error_aux = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 10))
                        # h1_error.append(h1_error_aux.real)






  #Order
        # plt.rcParams.update({'font.size':15})
        # for d in range(len(order_v)):
        #     print(len(Edge_modes)-1)
        #     print(len(Edge_modes))
        #     print(h1_error[len(Edge_modes)-1:len(Edge_modes):-1,-1])
        #     plt.plot(order_v, h1_error[(d+1)*len(Edge_modes)-1,-1])#, label=('Bubbles=%i' %Bubble_modes[i]))
        # plt.title('$H^1$ errors: increased degree of approximation')
        # plt.legend()
        # plt.xlabel('Edge modes')

        

"""