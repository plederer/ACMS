from ngsolve import *
from netgen.geom2d import SplineGeometry

import scipy.linalg
import scipy.sparse as sp
import numpy as np

SetNumThreads(4)

# Setting the geometry

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


# Basis functions per subdomains (constant for all subdomains)
bubble_modes = 60
edge_modes = 5

# Definition of Sobolev space, order of polynomial approximation can be increased
# The Dirichlet condition is imposed everywhere because it will be needed for the basis construction
# It does not effectively remove the boundary nodes
order = 2 # Polynomial degree of approximation
V = H1(mesh, order = order, dirichlet = ".*")

gfu = GridFunction(V)


###############################################################

"""
**Extension operators**

We need an extension from $\Gamma$ to $\Omega$, which is obtained by combining the extensions of functions from $\partial\Omega_j$ to $\Omega_j$.

For a given $\tau \in H^{1/2}(\partial \Omega_j)$, let $\tilde \tau \in H^1(\Omega_j)$ be any function satisfying $\tilde\tau_{\mid\partial\Omega_j}=\tau$.
Then, we indicate by $\tilde\tau_0 \in H_0^1(\Omega_j)$ the solution to

\begin{align} 
    \mathcal{A}_{j} (\tilde\tau_0, v)-(\kappa^2 \tilde\tau_0, v)_{\Omega_j} = -\left(\mathcal{A}_j(\tilde\tau,v)-(\kappa^2 \tilde\tau, v)_{\Omega_j}\right) \quad \forall  v\in  H_0^1(\Omega_j).
\end{align}

We characterize the $\mathcal{A}$-harmonic extension $E^{j}:H^{1/2}(\partial \Omega_j) \to H^1(\Omega_j)$ by setting 
    $E^{j}\tau := \tilde\tau+\tilde\tau_0$.

**Lemma** The extension operator $E^{j}:H^{1/2}(\partial \Omega_j) \to H^1(\Omega_j)$ is bounded, that is,
    \begin{equation}
        \|E^{j}\tau\|_{\mathcal{B}}\leq (1+\frac{1}{\beta^j})\|\tilde\tau\|_{\mathcal{B}},
    \end{equation}
    where $\tilde \tau\in H^1(\Omega_j)$ is any extension of $\tau\in \ H^{1/2}(\partial \Omega_j)$.
    

We note the following orthogonality relation, which is a crucial property for the construction of the ACMS  spaces: for all bubble functions $b_i^j \in H_0^1(\Omega_j)$, we have
\begin{align}
		\mathcal{A}_{j}(E^j \tau,b_i^j)-(\kappa^2 E^j \tau,b_i^j)_{\Omega_j}=0.
\end{align}

Assume that $e=\partial\Omega_j \cap \partial\Omega_i\in \mathcal{E}$ is a common edge of $\Omega_i$ and $\Omega_j$. Let $\tau \in H^{1/2}(\Gamma)$, which, by restriction, implies $\tau\in H^{1/2}(\partial \Omega_j)$ and $\tau\in H^{1/2}(\partial \Omega_i)$.


Extension on interface: $E^{\Gamma} : H^{1/2}(\Gamma) \to H^1_D(\Omega)$ by $(E^{\Gamma} \tau)_{\mid\Omega_j} = E^{j} \tau_{\mid \partial \Omega_j}$, for all $j = 1,\ldots,J$. 

Extension on edges: $E^{\Gamma} : H^{1/2}_{00}(e) \to H^1_D(\Omega)$ via $E^{\Gamma} \tau = E^{\Gamma} E_0^e\tau$, where $E_0^e: H^{1/2}_{00}(e)  \to H^{1/2}(\Gamma)$ denotes the extension by zero to the interface $\Gamma$.

"""




###############################################################
# EXTENSIONS

edge_extensions = {}
vol_extensions = {}

# Define harmonic extension on specific subdomain
# Returns the Sobolev space H^1_0(\Omega_j), the stiffness matrix and its inverse
def GetHarmonicExtensionDomain(dom_name):
    fd_all = V.GetDofs(mesh.Materials(dom_name)) # Dofs of specific domain
    base_space = H1(mesh, order = order, dirichlet = ".*") #Replicate H^1_0 on subdomain
    Vharm = Compress(base_space, fd_all)
    uharm, vharm = Vharm.TnT() # Trial and test functions
    aharm = BilinearForm(Vharm)
    #Setting bilinear form: - int (Grad u Grad v) d\Omega_j
    aharm += grad(uharm)*grad(vharm)*dx(dom_name)
    aharm.Assemble()
    aharm_inv = aharm.mat.Inverse(Vharm.FreeDofs(), inverse = "sparsecholesky")

    # Calc embedding
    # Is it local to global matrix? 
    ind = Vharm.ndof * [0]
    ii = 0
    for i, b in enumerate(fd_all):
        if b == True:
            ind[ii] = i
            ii += 1
    E = PermutationMatrix(base_space.ndof, ind)

    return Vharm, aharm.mat, aharm_inv, E


# Define harmonic extension on specific subdomain
# Returns the Sobolev space H^{1/2}_00(e??), the stiffness matrix and its inverse
def GetHarmonicExtensionEdge(edge_name):
    fd_all = V.GetDofs(mesh.Boundaries(edge_name)) # Dofs of specific edge
    bnd = "" # Initialize empty boundary 
    for b in mesh.GetBoundaries():
        if (b != edge_name): # If the edge is not our specified edge, then add it to bnd - why?
            bnd += b + "|"
    bnd = bnd[:-1] # Take every component exept the last one
    base_space = H1(mesh, order = order, dirichlet = bnd)

    #Setting bilinear form: - int (Grad u Grad v) d"e". 
    Vharm = Compress(base_space, fd_all) #Sobolev space H^{1/2}_00(e??)
    t = specialcf.tangential(2) # What is this specialcf?
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



# Function that computes the harmonic extensions on all subdomains and on all edges (of coarse mesh)
# Returns vol_extensions and edge_extensions
def CalcHarmonicExtensions():
    for dom_name in mesh.GetMaterials():
        Vharm, aharm, aharm_inv, E = GetHarmonicExtensionDomain(dom_name)
        vol_extensions[dom_name] = [Vharm, aharm, aharm_inv, E]

    for edge_name in mesh.GetBoundaries():
        Vharm, aharm, aharm_inv, E = GetHarmonicExtensionEdge(edge_name)
        edge_extensions[edge_name] = [Vharm, aharm, aharm_inv, E]



"""
**Edge basis**

Same for elliptic or Helmholtz case (changes the extension).

Let us consider $e\in\mathcal{E}$ and denote by $\partial_e$ the tangential derivative, i.e., differentiation along $e$.
We define the edge modes as solutions to the following weak formulation of the edge-Laplace eigenvalue problems: for each $e\in\mathcal{E}$, for $i \in \mathbb{N}$, find $(\tau^e_i,\lambda^e_i)\in H^{1/2}_{00}(e) \times\mathbb{R}$ such that
\begin{align}
(\partial_e \tau^e_i,\partial_e \eta)_e =\lambda^e_i ( \tau^e_i, \eta)_e \quad \text{for all } \eta\in H^{1/2}_{00}(e).
\end{align}
"""

###############################################################
# EDGE MODES

def calc_edge_basis(basis):
    for edge_name in mesh.GetBoundaries():
        vertex_dofs = V.GetDofs(mesh.BBoundaries(".*")) # Global vertices (coarse mesh)
        fd = V.GetDofs(mesh.Boundaries(edge_name)) & (~vertex_dofs) 
        # Vertices on a specific edge with boundaries removed (global vertices)
        base_space = H1(mesh, order = order, dirichlet = ".*") # Creating Sobolev space
        Vloc = Compress(base_space, fd) #Restricting Sobolev space on edge (with Dirichlet bc)

        uloc, vloc = Vloc.TnT() # Trial and test functions
        t = specialcf.tangential(2)
        #Setting bilinear form: - int (Grad u Grad v) de
        aloc = BilinearForm(Vloc)
        aloc += (grad(uloc)*t) * (grad(vloc)*t) * ds(skeleton=True, definedon=mesh.Boundaries(edge_name))
        aloc.Assemble()
        # What is the difference between the two differentials ds?
        #Setting bilinear form:  int u v de        
        mloc = BilinearForm(Vloc)
        mloc += uloc.Trace() * vloc.Trace() * ds(edge_name)
        mloc.Assemble()

        # Resolution of eigenvalue problem: AA x = ev MM x
        AA = sp.csr_matrix(aloc.mat.CSR())
        MM = sp.csr_matrix(mloc.mat.CSR())
        ev, evec =sp.linalg.eigs(A = AA, M = MM, k = edge_modes, which='SM')
        evec = evec.transpose()

        # Local to global mapping?
        ind = Vloc.ndof * [0]
        ii = 0
        for i, b in enumerate(fd):
            if b == True:
                ind[ii] = i
                ii += 1
        Eloc = PermutationMatrix(base_space.ndof, ind)

        for e in evec: # Going over eigenvectors
            # Vloc.Embed(e.real, gfu.vec)
            gfu.vec.data = Eloc.T * e.real # Grid funciton on full mesh
            #Mapping components?

            nb_dom = mesh.Boundaries(edge_name).Neighbours(VOL) #?
            gfu_edge = gfu.vec.CreateVector()
        
            for bi, bb in enumerate(mesh.GetMaterials()):
                if nb_dom.Mask()[bi]:
                    Vharm, aharm_mat, aharm_inv, E = vol_extensions[bb]
            
                    gfu_extension = GridFunction(Vharm) # Grid funciton on specific subdomain
                    res = gfu_extension.vec.CreateVector()

                    gfu_edge.data = gfu.vec  # Grid funciton on edge
                    # Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)
                    # Extension to subdomain * values on edge = function extended to subdomain
                    gfu_extension.vec.data = E * gfu_edge 
                    
                    # Harmonic extension on edge
                    res = aharm_mat * gfu_extension.vec 
                    gfu_extension.vec.data = - aharm_inv * res
                    # Vharm.Embed(gfu_extension.vec, gfu_edge)
                    gfu_edge.data = E.T * gfu_extension.vec
                    gfu.vec.data += gfu_edge

            basis.Append(gfu.vec)







"""
**Vertex basis**

**Helmholtz case:** For any $p\in\mathcal{V}$, let $\varphi_p: \Gamma\to \mathbb{R}$ denote a piecewise harmonic function, that is, $\Delta_e\varphi_{p\mid e}=0$ for all $e\in\mathcal{E}$, with $\Delta_e$ indicating the Laplace operator along the edge $e\in\mathcal{E}$, and $\varphi_p(q)=\delta_{p,q}$ for all $p,q\in\mathcal{V}$.
The vertex based space is then defined by linear combinations of corresponding  extensions:
\begin{align*}
V_{\mathcal{V}} = {\rm span}\{\, E^{\Gamma} \varphi_p \,:\, \ p\in\mathcal{V}\}.
\end{align*}
For our error analysis, we will employ the nodal interpolant
\begin{align}
I_{\mathcal{V}} v = \sum_{p\in \mathcal{V}} v(p) \varphi_p,
\end{align}
which is well-defined for functions $v:\overline{\Omega}\to\mathbb{C}$ that are continuous in all $p\in \mathcal{V}$. Moreover, note that the support of the vertex basis functions consists of all subdomains which share the vertex and is therefore local.
"""


###############################################################
# VERTEX BASIS

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





"""
**Bubble functions**


**Elliptic case:** Let us define the local bilinear form $\mathcal{A}{j}: H^1(\Omega_j) \times H^1(\Omega_j) \to \mathbb{R}$ with domain of integration $\Omega_j$ instead of $\Omega$.
Since $\mathcal{A}_{j}$ is symmetric, we can consider the eigenproblems: for $j=1,...,J$ and $i \in \mathbb{N}$, find $(b_i^j,\lambda_i^j)\in H_0^1(\Omega_j) \times \mathbb{R}$ such that
\begin{align}
	\mathcal{A}_{j}(b_i^j,v)= \lambda_i^j ( b_i^j,v)_{\Omega_j} \quad \forall v \in H_0^1(\Omega_j).
\end{align}

**Helmholtz case:** Let us define the local sesquilinear form $\mathcal{A}{j}: H^1(\Omega_j) \times H^1(\Omega_j) \to \mathbb{C}$ with domain of integration $\Omega_j$ instead of $\Omega$.
Since $\mathcal{A}_{j}$ is Hermitian, we can consider the eigenproblems: for $j=1,...,J$ and $i \in \mathbb{N}$, find $(b_i^j,\lambda_i^j)\in H)0^1(\Omega_j) \times \mathbb{R}$ such that
\begin{align}
	\mathcal{A}_{j}(b_i^j,v)= \lambda_i^j ( {\kappa^2} b_i^j,v)_{\Omega_j} \quad \forall v \in H_0^1(\Omega_j).
\end{align}
"""
###############################################################
# BUBBLE FUNCTIONS

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



#######################################################################
# SYSTEM ASSEMBLY


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
    