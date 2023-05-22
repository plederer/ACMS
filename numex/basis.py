from ngsolve import *
from netgen.geom2d import SplineGeometry

import scipy.linalg
import scipy.sparse as sp
import numpy as np

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

order = 4

bubble_modes = 10
edge_modes = 15

V = H1(mesh, order = order, dirichlet = ".*")
gfu = GridFunction(V)

###############################################################

edge_extensions = {}
vol_extensions = {}

def GetHarmonicExtensionDomain(dom_name):
    fd_all = V.GetDofs(mesh.Materials(dom_name))
    Vharm = Compress(H1(mesh, order = order, dirichlet = ".*"), fd_all) 
    uharm, vharm = Vharm.TnT()
    aharm = BilinearForm(Vharm)
    aharm += grad(uharm)*grad(vharm)*dx()
    aharm.Assemble()

    aharm_inv = aharm.mat.Inverse(Vharm.FreeDofs(), inverse = "sparsecholesky")
    
    return Vharm, aharm, aharm_inv

def GetHarmonicExtensionEdge(edge_name):
    fd_all = V.GetDofs(mesh.Boundaries(edge_name))
    bnd = ""
    for b in mesh.GetBoundaries():
        if (b != edge_name):
            bnd += b + "|"
    bnd = bnd[:-1]
    Vharm = Compress(H1(mesh, order = order, dirichlet = bnd), fd_all)
    t = specialcf.tangential(2)

    uharm, vharm = Vharm.TnT()
    aharm = BilinearForm(Vharm)
    aharm += (grad(uharm)*t) * (grad(vharm)*t) * ds(skeleton = True, definedon=mesh.Boundaries(edge_name))
    aharm.Assemble()
    aharm_inv = aharm.mat.Inverse(Vharm.FreeDofs(), inverse = "sparsecholesky")
    
    return Vharm, aharm, aharm_inv

def CalcHarmonicExtensions():
    for dom_name in mesh.GetMaterials():
        Vharm, aharm, aharm_inv = GetHarmonicExtensionDomain(dom_name)
        vol_extensions[dom_name] = [Vharm, aharm, aharm_inv]

    for edge_name in mesh.GetBoundaries():
        Vharm, aharm, aharm_inv = GetHarmonicExtensionEdge(edge_name)
        edge_extensions[edge_name] = [Vharm, aharm, aharm_inv]


def calc_edge_basis(basis):
    for edge_name in mesh.GetBoundaries():
        vertex_dofs = V.GetDofs(mesh.BBoundaries(".*")) 
        fd = V.GetDofs(mesh.Boundaries(edge_name)) & (~vertex_dofs) 
        
        Vloc = Compress(H1(mesh, order = order, dirichlet = ".*"), fd)

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

        for e in evec:
            Vloc.Embed(e.real, gfu.vec)
            nb_dom = mesh.Boundaries(edge_name).Neighbours(VOL)
            gfu_edge = gfu.vec.CreateVector()
        
            for i, n in enumerate(mesh.GetMaterials()):
                if nb_dom.Mask()[i]:
                    Vharm, aharm, aharm_inv = vol_extensions[n]
            
                    gfu_extension = GridFunction(Vharm)
                    res = gfu_extension.vec.CreateVector()

                    gfu_edge.data = gfu.vec
                    Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)

                    res = aharm.mat * gfu_extension.vec
                    gfu_extension.vec.data = - aharm_inv * res
                    Vharm.Embed(gfu_extension.vec, gfu_edge)
                    gfu.vec.data += gfu_edge

            basis.Append(gfu.vec)

###############################################################
def calc_vertex_basis(basis):
    for j,vertex_name in enumerate(mesh.GetBBoundaries()):
        gfu_vertex = gfu.vec.CreateVector()
        fd = V.GetDofs(mesh.BBoundaries(vertex_name))

        nb_edges = mesh.BBoundaries(vertex_name).Neighbours(BND)
        
        gfu.vec[:] = 0
        gfu.vec[np.nonzero(fd)[0]] = 1 

        for i, n in enumerate(mesh.GetBoundaries()):
            if nb_edges.Mask()[i]:
                # Vharm, aharm, aharm_inv = CreateHarmonicExtensionEdge(n)
                Vharm, aharm, aharm_inv = edge_extensions[n]
                gfu_extension = GridFunction(Vharm)
                res = gfu_extension.vec.CreateVector()

                gfu_vertex[:] = 0
                gfu_vertex[np.nonzero(fd)[0]] = 1 

                Vharm.EmbedTranspose(gfu_vertex, gfu_extension.vec)
                res = aharm.mat * gfu_extension.vec
                # only harmonic extension to one edge
                # has zero vertex value! 
                gfu_extension.vec.data = - aharm_inv * res
                Vharm.Embed(gfu_extension.vec, gfu_vertex)
                gfu.vec.data += gfu_vertex

        nb_dom = mesh.BBoundaries(vertex_name).Neighbours(VOL)
        gfu_edge = gfu.vec.CreateVector()
        
        for i, n in enumerate(mesh.GetMaterials()):
            if nb_dom.Mask()[i]:
                Vharm, aharm, aharm_inv = vol_extensions[n]
        
                gfu_extension = GridFunction(Vharm)
                res = gfu_extension.vec.CreateVector()

                gfu_edge.data = gfu.vec
                Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)

                res = aharm.mat * gfu_extension.vec
                gfu_extension.vec.data = - aharm_inv * res
                Vharm.Embed(gfu_extension.vec, gfu_edge)
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

        for e in evec:
            gfu.vec[:]=0.0
            # bubble.vec.FV()[:] = e.real
            Vloc.Embed(e.real, gfu.vec)
            basis.Append(gfu.vec)


basis = MultiVector(gfu.vec, 0)

CalcHarmonicExtensions()

calc_vertex_basis(basis)
calc_edge_basis(basis)
calc_bubble_basis(basis)

# for i in range(len(basis)):
#     gfu.vec[:] = 0.0
#     gfu.vec.data = basis[i]
#     Draw(gfu)
#     input()

num = len(basis)



u, v = V.TnT()

dom_bnd = "bottom0|bottom1|right|top1|top0|left"

kappa = 1
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

asmall.Inverse(ainvsmall)
usmall = ainvsmall * f_small

gfu.vec[:] = 0.0

gfu.vec.data = basis * usmall


### big solution

# ainv = a.mat.Inverse()
# gfu_ex = GridFunction(V)
# gfu_ex.vec.data = ainv * f.vec
# Draw(gfu_ex, mesh, "gfu_ex")
Draw(gfu, mesh, "gfu")

print(Norm(gfu.vec))