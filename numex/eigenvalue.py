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


bubble_modes = 10
edge_modes = 15

V = H1(mesh, order = 1, dirichlet = ".*")

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

a_inv = a.mat.Inverse(V.FreeDofs())


###############################################################
# edge basis

t = specialcf.tangential(2)

a_edge = BilinearForm(V)
a_edge += (grad(u)*t) * (grad(v)*t) * ds(skeleton = True) #, definedon=mesh.Boundaries("bottom"))

a_edge.Assemble()
a_edge_inv = a_edge.mat.Inverse(all_edge_freedofs)

m_edge = BilinearForm(V)
m_edge += u.Trace() * v.Trace() * ds()
m_edge.Assemble()

# edge_ev_evec = {}
# edge_basis = {}
def calc_edge_basis(basis):
    for edge_name in mesh.GetBoundaries():
        # edge_basis[edge_name] = []
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
            gfu_extension.data = gfu.vec - a_inv * res
            # Draw(gfu_extension, mesh, "extension")
            # input()
            # edge_basis[edge_name].append(gfu_extension)
            basis.Append(gfu_extension)
###############################################################
# vertex basis
# vertex_basis = {}
def calc_vertex_basis(basis):
    for j,vertex_name in enumerate(mesh.GetBBoundaries()):
        # print(vertex_name)
        gfu_extension = gfu.vec.CreateVector()
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
        res = a_edge.mat * gfu.vec
        gfu_extension.data = gfu.vec - a_edge_inv * res

        res = a.mat * gfu_extension
        gfu_extension.data = gfu_extension - a_inv * res
        # vertex_basis[vertex_name] = gfu_extension
        basis.Append(gfu_extension)
        # gfu.vec.data = gfu_extension
        # Draw(gfu)
        # basis[j] = gfu_extension
        # input()

###############################################################
# bubbles
# edge_ev_evec = {}
# edge_basis = {}

# bubble_basis = {}

def calc_bubble_basis(basis):
    for mat_name in mesh.GetMaterials():
        # bubble_basis[mat_name] = []
        fd = V.GetDofs(mesh.Materials(mat_name)) & V.FreeDofs()
        
        nd = sum(fd)
        # print(fd)
        A = Matrix(nd, nd)
        M = Matrix(nd, nd)

        dofs = []
        for i, b in enumerate(fd):
            if b == 1:
                dofs.append(i)

        for i in range(nd):
            for j in range(nd):
                A[i,j] = a.mat[dofs[i], dofs[j]]
                M[i,j] = m.mat[dofs[i], dofs[j]]

        # print(A)
        # print(M)

        ## ev numbering starts with zero!
        ev, evec = scipy.linalg.eigh(a=A, b=M, subset_by_index=[0, bubble_modes-1])
        # print(ev)
        evec = evec.transpose()
        # edge_ev_evec[edge_name] = [ev, evec]
        # print("AAAA")
        for e in evec:
            bubble = gfu.vec.CreateVector()
            bubble[:] = 0
            for i in range(nd):
                bubble[dofs[i]] = e[i]
            # bubble_basis[mat_name].append(bubble)
            # basis[j] = bubble
            basis.Append(bubble)


basis = MultiVector(gfu.vec, 0)

calc_vertex_basis(basis)
calc_edge_basis(basis)
calc_bubble_basis(basis)

num = len(basis)

u, v = V.TnT()

dom_bnd = "bottom0|bottom1|right|top1|top0|left"

a = BilinearForm(V)
a += grad(u) * grad(v) * dx
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

Draw(gfu, mesh, "gfu")
### big solution

ainv = a.mat.Inverse()
gfu_ex = GridFunction(V)
gfu_ex.vec.data = ainv * f.vec
Draw(gfu_ex, mesh, "gfu_ex")