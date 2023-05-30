from ngsolve import *
from netgen.geom2d import SplineGeometry

import scipy.linalg
import scipy.sparse as sp
import numpy as np

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


order = 2

bubble_modes = 60
edge_modes = 5

V = H1(mesh, order = order, dirichlet = ".*")
gfu = GridFunction(V)

###############################################################

edge_extensions = {}
vol_extensions = {}

def GetHarmonicExtensionDomain(dom_name):
    fd_all = V.GetDofs(mesh.Materials(dom_name))
    base_space = H1(mesh, order = order, dirichlet = ".*")
    Vharm = Compress(base_space, fd_all)
    uharm, vharm = Vharm.TnT()
    aharm = BilinearForm(Vharm)
    aharm += grad(uharm)*grad(vharm)*dx(dom_name)
    aharm.Assemble()
    aharm_inv = aharm.mat.Inverse(Vharm.FreeDofs(), inverse = "sparsecholesky")

    # Calc embedding
    ind = Vharm.ndof * [0]
    ii = 0
    for i, b in enumerate(fd_all):
        if b == True:
            ind[ii] = i
            ii += 1
    E = PermutationMatrix(base_space.ndof, ind)

    return Vharm, aharm.mat, aharm_inv, E

def GetHarmonicExtensionEdge(edge_name):
    fd_all = V.GetDofs(mesh.Boundaries(edge_name))
    bnd = ""
    for b in mesh.GetBoundaries():
        if (b != edge_name):
            bnd += b + "|"
    bnd = bnd[:-1]
    base_space = H1(mesh, order = order, dirichlet = bnd)
    Vharm = Compress(base_space, fd_all)
    t = specialcf.tangential(2)

    uharm, vharm = Vharm.TnT()
    aharm = BilinearForm(Vharm)
    aharm += (grad(uharm)*t) * (grad(vharm)*t) * ds(skeleton = True, definedon=mesh.Boundaries(edge_name))
    aharm.Assemble()
    
    aharm_inv = aharm.mat.Inverse(Vharm.FreeDofs(), inverse = "sparsecholesky")

    ind = Vharm.ndof * [0]
    ii = 0
    for i, b in enumerate(fd_all):
        if b == True:
            ind[ii] = i
            ii += 1
    E = PermutationMatrix(base_space.ndof, ind)

    return Vharm, aharm.mat, aharm_inv, E

def CalcHarmonicExtensions():
    for dom_name in mesh.GetMaterials():
        Vharm, aharm, aharm_inv, E = GetHarmonicExtensionDomain(dom_name)
        vol_extensions[dom_name] = [Vharm, aharm, aharm_inv, E]

    for edge_name in mesh.GetBoundaries():
        Vharm, aharm, aharm_inv, E = GetHarmonicExtensionEdge(edge_name)
        edge_extensions[edge_name] = [Vharm, aharm, aharm_inv, E]


def calc_edge_basis(basis):
    for edge_name in mesh.GetBoundaries():
        vertex_dofs = V.GetDofs(mesh.BBoundaries(".*")) 
        fd = V.GetDofs(mesh.Boundaries(edge_name)) & (~vertex_dofs) 
        base_space = H1(mesh, order = order, dirichlet = ".*")
        Vloc = Compress(base_space, fd)

        uloc, vloc = Vloc.TnT()
        t = specialcf.tangential(2)

        aloc = BilinearForm(Vloc)
        aloc += (grad(uloc)*t) * (grad(vloc)*t) * ds(skeleton=True, definedon=mesh.Boundaries(edge_name))
        aloc.Assemble()

        mloc = BilinearForm(Vloc)
        mloc += uloc.Trace() * vloc.Trace() * ds(edge_name)
        mloc.Assemble()

        AA = sp.csr_matrix(aloc.mat.CSR())
        MM = sp.csr_matrix(mloc.mat.CSR())
        ev, evec =scipy.sparse.linalg.eigs(A = AA, M = MM, k = edge_modes, which='SM')
        evec = evec.transpose()

        ind = Vloc.ndof * [0]
        ii = 0
        for i, b in enumerate(fd):
            if b == True:
                ind[ii] = i
                ii += 1
        Eloc = PermutationMatrix(base_space.ndof, ind)

        for e in evec:
            # Vloc.Embed(e.real, gfu.vec)
            gfu.vec.data = Eloc.T * e.real

            nb_dom = mesh.Boundaries(edge_name).Neighbours(VOL)
            gfu_edge = gfu.vec.CreateVector()
        
            for bi, bb in enumerate(mesh.GetMaterials()):
                if nb_dom.Mask()[bi]:
                    Vharm, aharm_mat, aharm_inv, E = vol_extensions[bb]
            
                    gfu_extension = GridFunction(Vharm)
                    res = gfu_extension.vec.CreateVector()

                    gfu_edge.data = gfu.vec
                    # Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)
                    gfu_extension.vec.data = E * gfu_edge
                    
                    res = aharm_mat * gfu_extension.vec
                    gfu_extension.vec.data = - aharm_inv * res
                    # Vharm.Embed(gfu_extension.vec, gfu_edge)
                    gfu_edge.data = E.T * gfu_extension.vec
                    gfu.vec.data += gfu_edge

            basis.Append(gfu.vec)

###############################################################

def calc_vertex_basis(basis):
    for j, vertex_name in enumerate(mesh.GetBBoundaries()):
        gfu_vertex = gfu.vec.CreateVector()
        fd = V.GetDofs(mesh.BBoundaries(vertex_name))

        nb_edges = mesh.BBoundaries(vertex_name).Neighbours(BND)
        nb_dom = mesh.BBoundaries(vertex_name).Neighbours(VOL)

        gfu.vec[:] = 0
        gfu.vec[np.nonzero(fd)[0]] = 1 

        for bi, bb in enumerate(mesh.GetBoundaries()):
            if nb_edges.Mask()[bi]:
                # Vharm, aharm_mat, aharm_inv = GetHarmonicExtensionEdge(bb)
                Vharm, aharm_mat, aharm_inv, E = edge_extensions[bb]
                gfu_extension = GridFunction(Vharm)
                gfu_extension.vec[:] = 0.0
                res = gfu_extension.vec.CreateVector()
                res[:]=0.0

                gfu_vertex[:] = 0
                gfu_vertex[np.nonzero(fd)[0]] = 1 

                gfu_extension.vec.data = E * gfu_vertex
                # Vharm.EmbedTranspose(gfu_vertex, gfu_extension.vec)
                res.data = aharm_mat * gfu_extension.vec
                # # # only harmonic extension to one edge
                # # # has zero vertex value! 
                gfu_extension.vec.data = - aharm_inv * res
                # Vharm.Embed(gfu_extension.vec, gfu_vertex)
                gfu_vertex.data = E.T * gfu_extension.vec
                gfu.vec.data += gfu_vertex
        
        gfu_edge = gfu.vec.CreateVector()
        
        for bi, bb in enumerate(mesh.GetMaterials()):
            if nb_dom.Mask()[bi]:
                Vharm, aharm_mat, aharm_inv, E = vol_extensions[bb]
                gfu_extension = GridFunction(Vharm)
                gfu_extension.vec[:] = 0.0
                res = gfu_extension.vec.CreateVector()
                gfu_edge[:]=0.0
                gfu_edge.data = gfu.vec
                # Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)
                gfu_extension.vec.data = E * gfu_edge
                
                res.data = aharm_mat * gfu_extension.vec
                gfu_extension.vec.data = - aharm_inv * res
                # Vharm.Embed(gfu_extension.vec, gfu_edge)
                gfu_edge.data = E.T * gfu_extension.vec
                gfu.vec.data += gfu_edge

        basis.Append(gfu.vec)

###############################################################

def calc_bubble_basis(basis):
    for mat_name in mesh.GetMaterials():
        fd = V.GetDofs(mesh.Materials(mat_name)) & V.FreeDofs()
        Vloc = Compress(H1(mesh, order = order, dirichlet = ".*"), fd)

        uloc, vloc = Vloc.TnT()
        aloc = BilinearForm(Vloc)
        aloc += grad(uloc) * grad(vloc) * dx()
        aloc.Assemble()

        mloc = BilinearForm(Vloc)
        mloc += uloc * vloc * dx()
        mloc.Assemble()

        AA = sp.csr_matrix(aloc.mat.CSR())
        MM = sp.csr_matrix(mloc.mat.CSR())
        ev, evec =scipy.sparse.linalg.eigs(A = AA, M = MM, k = bubble_modes, which='SM')
        evec = evec.transpose()

        ind = Vloc.ndof * [0]
        ii = 0
        for i, b in enumerate(fd):
            if b == True:
                ind[ii] = i
                ii += 1
        E = PermutationMatrix(V.ndof, ind)
        
        for e in evec:
            gfu.vec[:]=0.0
            # Vloc.Embed(e.real, gfu.vec)
            gfu.vec.data = E.T * e.real
            basis.Append(gfu.vec)

with TaskManager():
    basis = MultiVector(gfu.vec, 0)

    CalcHarmonicExtensions()
    calc_vertex_basis(basis)
    calc_edge_basis(basis)
    calc_bubble_basis(basis)

    # print("AAA")
    # for i in range(len(basis)):
    #     gfu.vec[:] = 0.0
    #     gfu.vec.data = basis[i]
    #     Draw(gfu)
    #     input()

    num = len(basis)



    u, v = V.TnT()

    dom_bnd = "bottom0|bottom1|right|top1|top0|left"

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


    asmall = InnerProduct (basis, a.mat * basis)
    ainvsmall = Matrix(num,num)

    f_small = InnerProduct(basis, f.vec)

    # print(asmall)
    asmall.Inverse(ainvsmall)
    usmall = ainvsmall * f_small

    gfu.vec[:] = 0.0

    gfu.vec.data = basis * usmall


    ### big solution

    ainv = a.mat.Inverse()
    gfu_ex = GridFunction(V)
    gfu_ex.vec.data = ainv * f.vec
    Draw(gfu-gfu_ex, mesh, "error")
    Draw(gfu_ex, mesh, "gfu_ex")
    Draw(gfu, mesh, "gfu")

    print(Norm(gfu.vec))
    