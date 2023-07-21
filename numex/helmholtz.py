from ngsolve import *
from netgen.geom2d import *

import numpy 
import scipy.linalg
import scipy.sparse as sp

from netgen.occ import *

from helping_functions import *


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

dom_bnd = "c0|c1|c2|c3"

order = 2

V = H1(mesh, order = order, complex = True)

u, v = V.TnT()


kappa = 16
# CF = CoefficientFunction
# k = kappa * CF((0.6,0.8))
# f = exp(-200 * ( (x)**2 + (y)**2))
g = exp(-200 * ( (x+1/sqrt(2))**2 + (y-1/sqrt(2))**2))
# u_ex = exp(-1J * (k[0] * x + k[1] * y))
# Draw(u_ex, mesh, "u_ex")
# g = -1j * kappa * (k[0] * x + k[1] * y) * u_ex - 1j * u_ex
beta = 1
omega = 1


a = BilinearForm(V)
a += grad(u) * grad(v) * dx()
a += - kappa**2 * u * v * dx()
a += -1J * omega * beta * u * v * ds(dom_bnd)
a.Assemble()

l = LinearForm(V)
# l += f * v * dx()
l += g * v * ds(dom_bnd)
l.Assemble()

gfu_ex = GridFunction(V)

ainv = a.mat.Inverse(V.FreeDofs(), inverse = "sparsecholesky")

gfu_ex.vec.data = ainv * l.vec
# Draw(gfu_ex, mesh, "u_ex")
print("finished")

# V = H1(mesh, order = order, complex = False)


gfu = GridFunction(V)
Draw(gfu, mesh, "u_acms")
##
max_bm = 2
acms = ACMS(order = order, mesh = mesh, bm = max_bm, em = 10)
acms.CalcHarmonicExtensions(kappa = kappa)
acms.calc_basis()

basis = acms.complex_basis()

num = len(basis)
print(num)
# for be in acms.basis_v:
#     gfu.vec.FV()[:] = be
#     Redraw()
#     input()
# print("number of basis functions:",num)
# print("AAA")
asmall = InnerProduct (basis, a.mat * basis)

asmall_np = np.zeros((num, num), dtype=numpy.complex128)
asmall_np = asmall.NumPy()

# for i in range(num):
#     for j in range(num):
#         asmall_np[i,j] = complex(asmall[i,j])

# print(asmall_np)
SetNumThreads(1)
ainvs_small_np = numpy.linalg.inv(asmall_np)

ainvsmall = Matrix(num,num,complex=True)


for i in range(num):
    for j in range(num):
        ainvsmall[i,j] = ainvs_small_np[i,j]

f_small = InnerProduct(basis, l.vec)

usmall = ainvsmall * f_small
gfu.vec[:] = 0.0

gfu.vec.data = basis * usmall

Draw(gfu-gfu_ex, mesh, "error")

print("finished_acms")