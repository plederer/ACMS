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

order = 2

bubble_modes = 2
edge_modes = 5

V = H1(mesh, order = order, dirichlet = ".*")

vertex_dofs = V.GetDofs(mesh.BBoundaries(".*")) 

# for i in range(4):
#     vertex_dofs[i] = 1
# print(vertex_dofs)
# input()
all_edge_freedofs = BitArray(V.ndof)
all_edge_freedofs.Clear()
edge_freedofs = {}
for edge_name in mesh.GetBoundaries():
    free_dofs = V.GetDofs(mesh.Boundaries(edge_name))
    for i, b in enumerate(vertex_dofs):
        if b == 1:
            free_dofs[i] = 0
        edge_freedofs[edge_name] = free_dofs
    all_edge_freedofs = all_edge_freedofs | free_dofs

# print(all_edge_freedofs)
# input()
# all dofs on the skeleton without vertex dofs



u, v = V.TnT()

a = BilinearForm(V)
a += grad(u) * grad(v) * dx
a.Assemble()

m = BilinearForm(V)
m += u * v * dx
m.Assemble()


# in V.Freedofs we only have the interior dofs 
# since we set a dirichlet flag on the whole boundary 
gfu = GridFunction(V)
res = gfu.vec.CreateVector()

# a_inv = a.mat.Inverse(V.FreeDofs())


###############################################################
# edge basis

t = specialcf.tangential(2)

a_edge = BilinearForm(V)
a_edge += (grad(u)*t) * (grad(v)*t) * ds(skeleton = True) #, definedon=mesh.Boundaries("bottom"))

a_edge.Assemble()
a_edge_inv = a_edge.mat.Inverse(all_edge_freedofs)

# m_edge = BilinearForm(V)
# m_edge += u.Trace() * v.Trace() * ds()
# m_edge.Assemble()

# edge_ev_evec = {}
# edge_basis = {}

def CreateHarmonicExtension(fd_all):
    Vharm = H1(mesh, order = order, dirichlet = ".*")
    for i in range(Vharm.ndof):
        if (fd_all[i] == 0):
            Vharm.SetCouplingType(i, COUPLING_TYPE.UNUSED_DOF)
    Vharm = Compress(Vharm) 
    uharm, vharm = Vharm.TnT()
    aharm = BilinearForm(Vharm)
    aharm += grad(uharm)*grad(vharm)*dx()
    aharm.Assemble()
    aharm_inv = aharm.mat.Inverse(Vharm.FreeDofs(), inverse = "sparsecholesky")
    
    return Vharm, aharm, aharm_inv



def calc_edge_basis(basis):
    for edge_name in mesh.GetBoundaries():
        vertex_dofs = V.GetDofs(mesh.BBoundaries(".*")) 
        fd = V.GetDofs(mesh.Boundaries(edge_name)) & (~vertex_dofs) 

        ## Create harmonic extension for each edge
        # neighboring regions
        nb = mesh.Boundaries(edge_name).Neighbours(VOL)
        fd_harm = V.GetDofs(nb) & (~V.GetDofs(mesh.Boundaries(".*")))
        fd_all = fd | fd_harm
        # fd_all = V.GetDofs(nb)
        
        Vloc = H1(mesh, order = order, dirichlet = ".*")
        for i in range(Vloc.ndof):
            if (fd[i] == 0): # or (fd_harm == 1)):
                Vloc.SetCouplingType(i, COUPLING_TYPE.UNUSED_DOF)
        Vloc = Compress(Vloc)

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

        
        Vharm, aharm, aharm_inv = CreateHarmonicExtension(fd_all)

        gfu_extension = GridFunction(Vharm)
        res = gfu_extension.vec.CreateVector()

        for e in evec:
            # gfu_extension = GridFunction(V) #gfu.vec.CreateVector()
            # gfu_loc = GridFunction(Vloc)
            # for i in range(sum(fd)):
            #     gfu_loc.vec[dofs[i]] = e[i].real
            # gfu.vec[:] = 0
            Vloc.Embed(e.real, gfu.vec)
            Vharm.EmbedTranspose(gfu.vec, gfu_extension.vec)
            # Vloc.Embed(e.real, gfu.vec)
            # for i in range(nd):
            #     gfu.vec[dofs[i]] = e[i]
            # Draw(gfu_extension)
            # Draw(gfu, mesh, "gfu")
            # input()
            res = aharm.mat * gfu_extension.vec
            # gfu.vec.data += -ainv*res
            gfu_extension.vec.data = gfu_extension.vec - aharm_inv * res
            # Draw(gfu_extension, mesh, "extension")
            Vharm.Embed(gfu_extension.vec, gfu.vec)
            # Draw(gfu)
            # input()
            # edge_basis[edge_name].append(gfu_extension)
            basis.Append(gfu.vec)
###############################################################
# vertex basis
# vertex_basis = {}
def calc_vertex_basis(basis):
    for j,vertex_name in enumerate(mesh.GetBBoundaries()):
        # print(vertex_name)
        gfu_extension_edge = gfu.vec.CreateVector()
        fd = V.GetDofs(mesh.BBoundaries(vertex_name)) 
        # print(fd)
        # print("AAA")
        gfu.vec[:] = 0
        for i, b in enumerate(fd):
            # print("AAA")
            if b == 1:
                gfu.vec[i] = 1
                # print("AAA")

        # THIS IS JUST A LINEAR EXTENSION!!!!!!!
        # VERY EXPENSIVE AT THE MOMENT, FIND ALTERNATIVE!

        # nb = mesh.BBoundaries(vertex_name).Neighbours(BND)
        # fd = V.GetDofs(mesh.Boundaries(nb))




        res_edge = gfu_extension_edge.CreateVector()
        res_edge = a_edge.mat * gfu.vec
        gfu_extension_edge.data = gfu.vec - a_edge_inv * res_edge

        nb_edges = mesh.BBoundaries(vertex_name).Neighbours(BND)
        fd_harm_edges = V.GetDofs(nb_edges)
        nb = mesh.BBoundaries(vertex_name).Neighbours(VOL)
        # dofs in the interior
        fd_harm = V.GetDofs(nb) & (~V.GetDofs(mesh.Boundaries(".*")))
        fd_all = fd_harm_edges | fd_harm

        Vharm = H1(mesh, order = order, dirichlet = ".*")
        for i in range(Vharm.ndof):
            if (fd_all[i] == 0):
                Vharm.SetCouplingType(i, COUPLING_TYPE.UNUSED_DOF)
        Vharm = Compress(Vharm)

        uharm, vharm = Vharm.TnT()
        aharm = BilinearForm(Vharm)
        aharm += grad(uharm)*grad(vharm)*dx(nb)
        aharm.Assemble()
        aharm_inv = aharm.mat.Inverse(Vharm.FreeDofs(), inverse = "sparsecholesky")

        gfu_extension = GridFunction(Vharm)
        res = gfu_extension.vec.CreateVector()
        
        Vharm.EmbedTranspose(gfu_extension_edge, gfu_extension.vec)

        res = aharm.mat * gfu_extension.vec
        gfu_extension.vec.data = gfu_extension.vec - aharm_inv * res
        Vharm.Embed(gfu_extension.vec, gfu.vec)

        # res_edge = a.mat * gfu_extension_edge
        # gfu_extension_edge.data = gfu_extension_edge - a_inv * res_edge
        # gfu.vec.data = gfu_extension_edge

        basis.Append(gfu.vec)
        
        # Draw(gfu)
        # basis[j] = gfu_extension
        # Draw(gfu)
        # input()

###############################################################

def calc_bubble_basis(basis):
    for mat_name in mesh.GetMaterials():
        fd = V.GetDofs(mesh.Materials(mat_name)) & V.FreeDofs()
        Vloc = H1(mesh, order = order, dirichlet = ".*")
        for i in range(Vloc.ndof):
            if fd[i] == 0:
                Vloc.SetCouplingType(i, COUPLING_TYPE.UNUSED_DOF)
        Vloc = Compress(Vloc)

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