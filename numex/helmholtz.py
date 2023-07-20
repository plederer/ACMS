from ngsolve import *
from netgen.geom2d import *

import numpy 
import scipy.linalg
import scipy.sparse as sp

from netgen.occ import *

# wp = WorkPlane()
# c = wp.Circle(0,0,1).Face()
# geo = OCCGeometry(c)
# ngmesh = geo.GenerateMesh(maxh=0.1)
# mesh = Mesh(ngmesh)

geo = SplineGeometry()
geo.AddCircle ( (0, 0), r=1, leftdomain=1, rightdomain=0)
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)

Draw(mesh)

V = H1(mesh, order = 4, complex = True)

u, v = V.TnT()


kappa = 16
# CF = CoefficientFunction
# k = kappa * CF((0.6,0.8))
# f = exp(-200 * ( (x)**2 + (y)**2))
g = exp(-200 * ( (x+1/sqrt(2))**2 + (y-1/sqrt(2))**2))
# u_ex = exp(-1J * (k[0] * x + k[1] * y))
# Draw(u_ex, mesh, "u_ex")
# g = -1j * kappa * (k[0] * x + k[1] * y) * u_ex - 1j * u_ex

Draw(g, mesh, "source")
beta = 1
omega = 1


a = BilinearForm(V)
a += grad(u) * grad(v) * dx()
a += - kappa**2 * u * v * dx()
a += -1J * omega * beta * u * v * ds()
a.Assemble()

l = LinearForm(V)
# l += f * v * dx()
l += g * v * ds()
l.Assemble()

gfu = GridFunction(V)

ainv = a.mat.Inverse(V.FreeDofs(), inverse = "sparsecholesky")

gfu.vec.data = ainv * l.vec
Draw(gfu, mesh, "u")
print("finished")