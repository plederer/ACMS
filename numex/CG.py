from ngsolve import *
from netgen.geom2d import SplineGeometry

import scipy.linalg
import scipy.sparse as sp
import numpy as np

from helping_functions import *

from ngsolve.krylovspace import MinResSolver, GMResSolver,  CGSolver

SetNumThreads(4)

geo = SplineGeometry()
Points = [(0,0), (1,0), (2,0), 
          (2,1), (1,1), (0,1)]

bcs_edge = ["bottom0", "bottom1", "right", "top1", "top0", "left", "middle"]

for i, pnt in enumerate(Points):
    geo.AddPoint(*pnt, name = "V" + str(i))

geo.Append(["line", 0, 1], leftdomain=1, rightdomain=0, bc="bottom0")
geo.Append(["line", 1, 2], leftdomain=2, rightdomain=0, bc="bottom1")
geo.Append(["line", 2, 3], leftdomain=2, rightdomain=0, bc="right")
geo.Append(["line", 3, 4], leftdomain=2, rightdomain=0, bc="top1")
geo.Append(["line", 4, 5], leftdomain=1, rightdomain=0, bc="top0")
geo.Append(["line", 5, 0], leftdomain=1, rightdomain=0, bc="left")
geo.Append(["line", 1, 4], leftdomain=1, rightdomain=2, bc="middle")

# ngmesh = unit_square.GenerateMesh(maxh=0.1)
mesh = Mesh(geo.GenerateMesh(maxh = 0.1))
mesh.ngmesh.SetMaterial(1,"omega0")
mesh.ngmesh.SetMaterial(2,"omega1")
Draw(mesh)
print(mesh.nv)
# quit()
print(mesh.GetMaterials())
print(mesh.GetBoundaries())
print(mesh.GetBBoundaries())
#input()

order = 3
dom_bnd = "bottom1|right|top1|top0"

with TaskManager():
    V =  H1(mesh, order = order, dirichlet = dom_bnd)
    gfu = GridFunction(V)
    basis = MultiVector(gfu.vec, 0)
    BM = 10
    EM = 5
    acms = ACMS(order = order, mesh = mesh, bm = BM, em = EM)
    acms.CalcHarmonicExtensions()
    # basis_v = MultiVector(gfu.vec, 0)
    basis_e = MultiVector(gfu.vec, 0)
    basis_b = MultiVector(gfu.vec, 0)

    # acms.calc_vertex_basis(basis_v)
    acms.calc_edge_basis(basis_e)
    acms.calc_bubble_basis(basis_b)

    basis = MultiVector(gfu.vec, 0)
    

    for d, dom in enumerate(mesh.GetBoundaries()):
        if (dom == "middle"):
            for i in range(EM):
                basis.Append(basis_e[d * EM + i])
    
    for bb in basis_b:
        basis.Append(bb)

    
    num = len(basis)

    u, v = V.TnT()

    

    # fd = V.FreeDofs() & V.GetDofs(mesh.Boundaries("middle"))
    fd = ~V.GetDofs(mesh.Boundaries(dom_bnd))
    kappa = 0
    a = BilinearForm(V)
    a += grad(u) * grad(v) * dx
    # a += u * v * dx
    # a += - kappa * u * v * dx
    # a += 10**6 * u * v * ds(dom_bnd)
    # a += u * v * dx

    # "type" = local is a Jacobi preconditioner
    # thus, the diagonal of A
    c = Preconditioner(a, type="local")

    a.Assemble()

    # alternative: block jacobi preconditioner
    # list of lists with dof numbers of the blocks
    blocks = []
    for vi in mesh.vertices:
        v_block = []
        # add vertex dof
        v_block += [d for d in V.GetDofNrs(vi) if V.FreeDofs()[d]]
        # print("V = ", v)
        # print("faces = ", v.faces)
        # print("edges = ", v.edges)
        for ed in vi.edges:
            #higer order dofs per edge, does NOT include the vertices!
            # print(V.GetDofNrs(ed))
            v_block += [d for d in V.GetDofNrs(ed) if V.FreeDofs()[d]]
        for fac in vi.faces:
            #higer order dofs per face (=element), does NOT include the vertices and edges!
            # print(V.GetDofNrs(fac))
            v_block += [d for d in V.GetDofNrs(fac) if V.FreeDofs()[d]]
        blocks.append(v_block)

    #remove blocks that are empty, like for vertices on dir_bnd
    blocks = [x for x in blocks if len(x)]

    # create block jacobi smoother
    smoother = a.mat.CreateBlockSmoother(blocks)

    f = LinearForm(V)
    f += x*x * v * dx()
    f.Assemble()


    asmall = InnerProduct (basis, a.mat * basis)
    ainvsmall = Matrix(num,num)
    # for i in range(num):
    #     for j in range(num):
    #         if (i == j):
    #             ainvsmall[i,j] = 1
    #         else:
    #             ainvsmall[i,j] = 0

    # f_small = InnerProduct(basis, f.vec)

    # print(asmall)

    asmall.Inverse(ainvsmall)
    # print(ainvsmall)
    # usmall = ainvsmall * f_small

    gfu.vec[:] = 0.0


    # gfu.vec.data = basis * usmall

    # A_s small matrix on ACMS
    # P Basis transformation
    # C = P A_s^{-1} P^T 
    class myC(BaseMatrix):
                def __init__ (self):
                    super(myC, self).__init__()
                    self.temp = a.mat.CreateColVector()
                def Mult(self, x, y):
                    y[:] = 0
                    # t1 = P^T * x
                    t1 = InnerProduct(basis,x)
                    # t2 = A_s^{-1} * t1
                    t2 = ainvsmall * t1
                    # P * t2
                    y.data = basis * t2
                def Height(self):
                    return V.ndof

                def Width(self):
                    return V.ndof

                def CreateColVector(self):
                    return a.mat.CreateColVector()
    
    # C = myC() + c.mat
    C = myC() + smoother

    
    # C = c.mat
    
    solver = CGSolver(mat = a.mat, pre = C, maxiter = 1000, tol = 1e-14, printrates = "\r")
    
    ainv = a.mat.Inverse(fd)
    # solver = ainv
    gfu_ex = GridFunction(V)
    gfu_ex.vec.data = ainv * f.vec

    gfu.vec.data = solver * f.vec


    Draw(gfu_ex, mesh, "gfu_ex")
    Draw(gfu, mesh, "gfu")

    # print(Norm(gfu.vec))
    Draw(gfu-gfu_ex, mesh, "error")