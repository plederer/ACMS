from ngsolve import *
from netgen.geom2d import SplineGeometry

import scipy.linalg
import scipy.sparse as sp


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
print(mesh.GetMaterials())
print(mesh.GetBoundaries())
print(mesh.GetBBoundaries())
#input()

edge_modes = 2

V = H1(mesh, order = 4, dirichlet = ".*")

vertex_dofs = V.GetDofs(mesh.BBoundaries(".*")) 

# for i in range(4):
#     vertex_dofs[i] = 1
# print(vertex_dofs)
# input()
edge_freedofs = {}
for edge_name in mesh.GetBoundaries():
    free_dofs = V.GetDofs(mesh.Boundaries(edge_name))
    for i, b in enumerate(vertex_dofs):
        if b == 1:
            free_dofs[i] = 0
        edge_freedofs[edge_name] = free_dofs

u, v = V.TnT()

t = specialcf.tangential(2)

a_edge = BilinearForm(V)
a_edge += (grad(u)*t) * (grad(v)*t) * ds(skeleton = True) #, definedon=mesh.Boundaries("bottom"))
a_edge.Assemble()

m_edge = BilinearForm(V)
m_edge += u.Trace() * v.Trace() * ds()
m_edge.Assemble()


a = BilinearForm(V)
a += grad(u) * grad(v) * dx
a.Assemble()

# in V.Freedofs we only have the interior dofs 
# since we set a dirichlet flag on the whole boundary 
gfu = GridFunction(V)
res = gfu.vec.CreateVector()

ainv = a.mat.Inverse(V.FreeDofs())

edge_ev_evec = {}
edge_basis = {}
for edge_name in mesh.GetBoundaries():
    edge_basis[edge_name] = []
    fd = edge_freedofs[edge_name]
    # print(fd)
    nd = sum(fd)

    A = Matrix(nd, nd)
    M = Matrix(nd, nd)

    dofs = []
    for i, b in enumerate(fd):
        if b == 1:
            dofs.append(i)

    for i in range(nd):
        for j in range(nd):
            A[i,j] = a_edge.mat[dofs[i], dofs[j]]
            M[i,j] = m_edge.mat[dofs[i], dofs[j]]

    # print(A)
    # print(M)

    ## ev numbering starts with zero!
    ev, evec = scipy.linalg.eigh(a=A, b=M, subset_by_index=[0, edge_modes-1])
    evec = evec.transpose()
    # edge_ev_evec[edge_name] = [ev, evec]


    for e in evec:
        gfu_extension = gfu.vec.CreateVector()
        gfu.vec[:] = 0
        for i in range(nd):
            gfu.vec[dofs[i]] = e[i]
        # Draw(gfu)
        res = a.mat * gfu.vec
        # gfu.vec.data += -ainv*res
        gfu_extension.data = gfu.vec - ainv * res
        # Draw(gfu_extension, mesh, "extension")
        # input()
        edge_basis[edge_name].append(gfu_extension)



for edge_name in mesh.GetBoundaries():
    for phi_e in edge_basis[edge_name]:
        gfu.vec.data = phi_e
        Draw(gfu)
        input()






# step 1:
# calc basis on every edge

# step 2:
# make harmonic extension
# 0 on all other edges
# -Delta phi_e = 0 in the int. (= other cells)

# step 3: construction of nodal basis functions
# on each vertex = 1, 
# harmonic extension onto the skeleton (for linear that's simple)

# step 4: construction of interior basis functions
