from ngsolve import *
from netgen.geom2d import SplineGeometry

import scipy.linalg
import scipy.sparse as sp
import numpy as np

from helping_functions import *

SetNumThreads(4)

# Setting the geometry

if False:
    geo = SplineGeometry()
    # Setting global decomposition vertices (coarse mesh)
    Points = [(0,0), (1,0), (2,0), 
            (2,1), (1,1), (0,1)]
    # Setting edgle lables counter-clock wise 
    bcs_edge = ["bottom0", "bottom1", "right", "top1", "top0", "left", "middle"]
    # Labeling vertices V1,...,V6
    for i, pnt in enumerate(Points):
        geo.AddPoint(*pnt, name = "V" + str(i))
        
    # Labeling edges by specifying end points, neighbouring domains (counterclock-wise), label
    geo.Append(["line", 0, 1], leftdomain=1, rightdomain=0, bc="bottom0")
    geo.Append(["line", 1, 2], leftdomain=2, rightdomain=0, bc="bottom1")
    geo.Append(["line", 2, 3], leftdomain=2, rightdomain=0, bc="right")
    geo.Append(["line", 3, 4], leftdomain=2, rightdomain=0, bc="top1")
    geo.Append(["line", 4, 5], leftdomain=1, rightdomain=0, bc="top0")
    geo.Append(["line", 5, 0], leftdomain=1, rightdomain=0, bc="left")
    geo.Append(["line", 1, 4], leftdomain=1, rightdomain=2, bc="middle")

    # ngmesh = unit_square.GenerateMesh(maxh=0.1)
    mesh = Mesh(geo.GenerateMesh(maxh = 0.1))
    # Labeling subdomains in coarse mesh
    mesh.ngmesh.SetMaterial(1,"omega0")
    mesh.ngmesh.SetMaterial(2,"omega1")
    Draw(mesh)
    print(mesh.nv) # Number of vertices?
    print(mesh.GetMaterials()) # Subdomains
    print(mesh.GetBoundaries()) # Edges
    print(mesh.GetBBoundaries()) # Vertices
    #input()
else:
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
    ngmesh = geo.GenerateMesh(maxh=0.1)
    mesh = Mesh(ngmesh)
    for i in range(8):
        mesh.ngmesh.SetMaterial(i+1,"om" + str(i))

    Draw(mesh)
    print(mesh.nv)
    # quit()
    print(mesh.GetMaterials())
    print(mesh.GetBoundaries())
    print(mesh.GetBBoundaries())
    # input()



#######################################################################
# SYSTEM ASSEMBLY


with TaskManager():
    V =  H1(mesh, order = 2, dirichlet = ".*")
    gfu = GridFunction(V)
    max_bm = 100
    acms = ACMS(order = 2, mesh = mesh, bm = max_bm, em = 10)
    acms.CalcHarmonicExtensions()
    
    u, v = V.TnT()
    # dom_bnd = "bottom0|bottom1|right|top1|top0|left"
    dom_bnd = "c0|c1|c2|c3"

    kappa = 0
    a = BilinearForm(V)
    a += grad(u) * grad(v) * dx
    a += - kappa * u * v * dx
    a += 10**6 * u * v * ds(dom_bnd)
    # a += u * v * dx
    a.Assemble()

    f = LinearForm(V)
    f += 1 * v * dx()
    f.Assemble()
    ainv = a.mat.Inverse()
    gfu_ex = GridFunction(V)
    gfu_ex.vec.data = ainv * f.vec
    
    # basis_v = MultiVector(gfu.vec, 0)
    # basis_e = MultiVector(gfu.vec, 0)
    # basis_b = MultiVector(gfu.vec, 0)
    # acms.calc_vertex_basis(basis_v)
    # acms.calc_edge_basis(basis_e)
    # acms.calc_bubble_basis(basis_b)

    acms.calc_basis()

    # acms.calc_vertex_basis(basis_v)
    # acms.calc_edge_basis(basis_e)
    # acms.calc_bubble_basis(basis_b)
    # print(basis_v.dim)
    # quit()

    for BM in [50,60,70,80,90]:
        basis = MultiVector(gfu.vec, 0)
        for bv in acms.basis_v:
            basis.Append(bv)
            # gfu.vec.data = bv
            # Draw(gfu)
            # input()

        for be in acms.basis_e:
            basis.Append(be)
        
        
        for d, dom in enumerate(mesh.GetMaterials()):
            for i in range(BM):
                basis.Append(acms.basis_b[d * max_bm + i])


        num = len(basis)




        asmall = InnerProduct (basis, a.mat * basis)
        ainvsmall = Matrix(num,num)

        f_small = InnerProduct(basis, f.vec)

        # print(asmall)
        asmall.Inverse(ainvsmall)
        usmall = ainvsmall * f_small

        gfu.vec[:] = 0.0

        gfu.vec.data = basis * usmall


        ### big solution

        
        
        Draw(gfu-gfu_ex, mesh, "error")
        Draw(gfu_ex, mesh, "gfu_ex")
        Draw(gfu, mesh, "gfu")

        # print(Norm(gfu.vec))
        
        # if uex is a given Coefficientfunction
        # u_ex = x**2 * ...
        # grad_uex = CF((u_ex.Diff(x),u_ex.Diff(y))) 

        # if uex is given as a Gridfunction
        # i.e. a finer FEM solution...
        # grad_uex = Grad(gfu_ex)

        grad_uex = Grad(gfu_ex)
        diff = grad_uex - Grad(gfu)

        h1_error = sqrt( Integrate ( InnerProduct(diff,diff), mesh, order = 10))
        print(h1_error)