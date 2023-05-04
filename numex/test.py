from ngsolve import *
from netgen.geom2d import unit_square

import scipy.linalg
# from netgen.occ import *

# wp = WorkPlane(Axes())
# wp.Rectangle(1,1)

# face1 = wp.Face().mat("m1")
# face2 = face1.Move((1, 0, 0)).mat("m2")

# all_faces = Glue([face1, face2])
# geo = OCCGeometry(face1)
ngmesh = unit_square.GenerateMesh(maxh=0.05)
mesh = Mesh(ngmesh)


V = H1(mesh, order = 3)


free_dofs = V.GetDofs(mesh.Boundaries("bottom"))
free_dofs[0] = 0
free_dofs[1] = 0

for d,b in enumerate(free_dofs):
    if b==0:
        V.SetCouplingType(d, COUPLING_TYPE.HIDDEN_DOF)

V = Compress(V)
print(V.ndof)

# print(free_dofs)

# u = V.TestFunction()
# v = V.TrialFunction()
u, v = V.TnT()

t = specialcf.tangential(2)

a = BilinearForm(V)
# a += grad(u) * grad(v) * dx()
a += (grad(u)*t) * (grad(v)*t) * ds(skeleton = True, definedon=mesh.Boundaries("bottom"))
a.Assemble()

m = BilinearForm(V)
# m += u.Trace() * v.Trace() * ds("bottom")
m += u * v * dx(element_boundary = True)
m.Assemble()

# f = LinearForm(V)
# f += 1 * v.Trace() * ds("bottom")
# f.Assemble()

# print(a.mat)

nd = V.ndof

A = Matrix(nd, nd)
M = Matrix(nd, nd)

for i in range(nd):
    for j in range(nd):
        A[i,j] = a.mat[i, j]
        M[i,j] = m.mat[i, j]

ev, evec = scipy.linalg.eigh(a=A, b=M)
ev = list(ev)
# eigvalsa.sort(key = lambda x: abs(x))
# print(ev)

gfu = GridFunction(V)
# print(gfu.vec)


# for i in range(V.ndof):
#     gfu.vec[:] = 0.0
#     gfu.vec[i] = 1
#     Draw(gfu)
#     input()

# print(evec)
for e in evec:
    # print(e)
    for i in range(V.ndof):
        gfu.vec[i] = e[i]
    
    Draw(gfu)
    input()

# M = sp.csr_matrix(a.mat.CSR())


# gfu = GridFunction(V)
# gfu.vec.data = a.mat.Inverse(free_dofs) * f.vec

# Draw(gfu)

#  BitArray(V.ndof)



# for d, b in enumerate(V.GetDofs(mesh.Boundaries(“bottom”))):
#     if b==0:
#         V.SetCouplingType(d, COUPLING_TYPE.HIDDEN_DOF)