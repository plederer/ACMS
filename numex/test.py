from ngsolve import *
from netgen.geom2d import unit_square, SplineGeometry

import scipy.linalg
# from netgen.occ import *

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
mesh = Mesh(geo.GenerateMesh(maxh = 1))
# Labeling subdomains in coarse mesh
mesh.ngmesh.SetMaterial(1,"omega0") 
mesh.ngmesh.SetMaterial(2,"omega1")
Draw(mesh)
# input()

V = H1(mesh, order = 1)
V2 = H1(mesh, order = 1)

free_dofs = V.GetDofs(mesh.Materials("omega1"))
# free_dofs[0] = 0
# free_dofs[1] = 0

print(free_dofs)
for d, b in enumerate(free_dofs):
    if b==0:
        V.SetCouplingType(d, COUPLING_TYPE.HIDDEN_DOF)

print(V.ndof)
V = Compress(V)
print(V.ndof)

u,v = V.TnT()
a = BilinearForm(V)
a += u * v * dx("omega2")
a.Assemble()
print(a.mat)

# # print(free_dofs)

# # u = V.TestFunction()
# # v = V.TrialFunction()
# u, v = V.TnT()

# t = specialcf.tangential(2)

# a = BilinearForm(V)
# # a += grad(u) * grad(v) * dx()
# a += (grad(u)*t) * (grad(v)*t) * ds(skeleton = True, definedon=mesh.Boundaries("bottom"))
# a.Assemble()

# m = BilinearForm(V)
# # m += u.Trace() * v.Trace() * ds("bottom")
# m += u * v * dx(element_boundary = True)
# m.Assemble()

# # f = LinearForm(V)
# # f += 1 * v.Trace() * ds("bottom")
# # f.Assemble()

# # print(a.mat)

# nd = V.ndof

# A = Matrix(nd, nd)
# M = Matrix(nd, nd)

# for i in range(nd):
#     for j in range(nd):
#         A[i,j] = a.mat[i, j]
#         M[i,j] = m.mat[i, j]

# ev, evec = scipy.linalg.eigh(a=A, b=M)
# ev = list(ev)
# # eigvalsa.sort(key = lambda x: abs(x))
# # print(ev)

# gfu = GridFunction(V)
# # print(gfu.vec)


# # for i in range(V.ndof):
# #     gfu.vec[:] = 0.0
# #     gfu.vec[i] = 1
# #     Draw(gfu)
# #     input()

# # print(evec)
# for e in evec:
#     # print(e)
#     for i in range(V.ndof):
#         gfu.vec[i] = e[i]
    
#     Draw(gfu)
#     input()

# # M = sp.csr_matrix(a.mat.CSR())


# # gfu = GridFunction(V)
# # gfu.vec.data = a.mat.Inverse(free_dofs) * f.vec

# # Draw(gfu)

# #  BitArray(V.ndof)



# # for d, b in enumerate(V.GetDofs(mesh.Boundaries(“bottom”))):
# #     if b==0:
# #         V.SetCouplingType(d, COUPLING_TYPE.HIDDEN_DOF)