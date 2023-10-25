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

ngmesh = geo.GenerateMesh(maxh=0.02)

mesh = Mesh(ngmesh)
for i in range(8):
    mesh.ngmesh.SetMaterial(i+1,"om" + str(i))

Draw(mesh)
print(mesh.nv)
# quit()
print(mesh.GetMaterials())
print(mesh.GetBoundaries())
print(mesh.GetBBoundaries())

dom_bnd = "c0|c1|c2|c3"


# PROBLEM SETTING

plot_error = 0
problem = 2
order_v = [1,2] # ATTENTION: if the mesh is too coarse, we cannot have many bubbles/modes
Bubble_modes = [1,2,4,8]
Edge_modes = [1,2,4,8] 

if problem ==1:  #Problem setting - PLANE WAVE
    
    omega = 1
    kappa = omega
    k = kappa * CF((0.6,0.8)) #CF = CoefficientFunction
    beta = 1
    f = 0
    u_ex = exp(-1J * (k[0] * x + k[1] * y))
    g = -1j * kappa * (k[0] * x + k[1] * y) * u_ex - 1j *beta * u_ex
    Du_ex = CF((u_ex.Diff(x), u_ex.Diff(y)))

elif problem == 2:  #Problem setting - INTERIOR SOURCE
        
    class Point: """ Point class for representing and manipulating x,y coordinates. """

    def __init__(self):
        """ Create a new point at the origin """
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


#  RESOLUTION OF GROUND TRUTH SOLUTION
#Computing the FEM solution /  ground truth solution with higher resolution

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

# GROUND TRUTH SOLUTION
grad_uex = Grad(gfu_ex)
# grad_uex = Du_ex #If we have analytical solution defined

#  ACMS RESOLUTION


h1_error = []
dofs =[]
max_bm = Bubble_modes[-1]
max_em = Edge_modes[-1]

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

                    #Computing error
                    diff = grad_uex - Grad(gfu)
                    # h1_error_aux = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 10))
                    h1_error_aux = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 10))
                    #Needs to do complex conjugate
                    Draw(gfu, mesh, "u_acms")
                    h1_error.append(h1_error_aux.real)
            

#Saving the error on a file

h1_error_3d = np.reshape(h1_error, (len(order_v), len(Edge_modes), len(Bubble_modes)))
np.save('H1_error', h1_error_3d)
H1_error = np.load('H1_error.npy')
print(H1_error)


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