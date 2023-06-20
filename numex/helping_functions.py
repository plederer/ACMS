from ngsolve import *
from netgen.geom2d import SplineGeometry

import scipy.linalg
import scipy.sparse as sp
import numpy as np

###############################################################
# EXTENSIONS

class ACMS:
    def __init__(self, order, mesh, bm=0, em=0, dirichlet = ".*"):
        self.order = order # Polynomial degree of approximation
        self.dirichlet = dirichlet
        self.mesh = mesh
        self.V = H1(mesh, order = order, dirichlet = dirichlet)

        self.gfu = GridFunction(self.V)

        self.edge_extensions = {}
        self.vol_extensions = {}

        self.bubble_modes = bm
        self.edge_modes = em


    # Define harmonic extension on specific subdomain
    # Returns the Sobolev space H^1_0(\Omega_j), the stiffness matrix and its inverse
    def GetHarmonicExtensionDomain(self, dom_name):
        fd_all = self.V.GetDofs(self.mesh.Materials(dom_name)) # Dofs of specific domain
        base_space = H1(self.mesh, order = self.order, dirichlet = self.dirichlet) #Replicate H^1_0 on subdomain
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
    def GetHarmonicExtensionEdge(self, edge_name):
        fd_all = self.V.GetDofs(self.mesh.Boundaries(edge_name)) # Dofs of specific edge
        bnd = "" # Initialize empty boundary 
        for b in self.mesh.GetBoundaries():
            if (b != edge_name): # If the edge is not our specified edge, then add it to bnd - why?
                bnd += b + "|"
        bnd = bnd[:-1] # Take every component exept the last one
        base_space = H1(self.mesh, order = self.order, dirichlet = bnd)

        #Setting bilinear form: - int (Grad u Grad v) d"e". 
        Vharm = Compress(base_space, fd_all) #Sobolev space H^{1/2}_00(e??)
        t = specialcf.tangential(2) # What is this specialcf?
        uharm, vharm = Vharm.TnT()
        aharm = BilinearForm(Vharm)
        aharm += (grad(uharm)*t) * (grad(vharm)*t) * ds(skeleton = True, definedon=self.mesh.Boundaries(edge_name))
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
    def CalcHarmonicExtensions(self):
        for dom_name in self.mesh.GetMaterials():
            Vharm, aharm, aharm_inv, E = self.GetHarmonicExtensionDomain(dom_name)
            self.vol_extensions[dom_name] = [Vharm, aharm, aharm_inv, E]

        for edge_name in self.mesh.GetBoundaries():
            Vharm, aharm, aharm_inv, E = self.GetHarmonicExtensionEdge(edge_name)
            self.edge_extensions[edge_name] = [Vharm, aharm, aharm_inv, E]



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

    def calc_edge_basis(self, basis):
        for edge_name in self.mesh.GetBoundaries():
            vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(".*")) # Global vertices (coarse mesh)
            fd = self.V.GetDofs(self.mesh.Boundaries(edge_name)) & (~vertex_dofs) 
            # Vertices on a specific edge with boundaries removed (global vertices)
            base_space = H1(self.mesh, order = self.order, dirichlet = self.dirichlet) # Creating Sobolev space
            Vloc = Compress(base_space, fd) #Restricting Sobolev space on edge (with Dirichlet bc)

            uloc, vloc = Vloc.TnT() # Trial and test functions
            t = specialcf.tangential(2)
            #Setting bilinear form: - int (Grad u Grad v) de
            aloc = BilinearForm(Vloc)
            aloc += (grad(uloc)*t) * (grad(vloc)*t) * ds(skeleton=True, definedon=self.mesh.Boundaries(edge_name))
            aloc.Assemble()


            # What is the difference between the two differentials ds?
            #Setting bilinear form:  int u v de        
            mloc = BilinearForm(Vloc)
            mloc += uloc.Trace() * vloc.Trace() * ds(edge_name)
            #mloc += uloc * vloc * ds(skeleton = True, edge_name)
            mloc.Assemble()

            # Resolution of eigenvalue problem: AA x = ev MM x
            AA = sp.csr_matrix(aloc.mat.CSR())
            MM = sp.csr_matrix(mloc.mat.CSR())
            ev, evec =sp.linalg.eigs(A = AA, M = MM, k = self.edge_modes, which='SM')
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
                self.gfu.vec.data = Eloc.T * e.real # Grid funciton on full mesh
                #Mapping components?

                nb_dom = self.mesh.Boundaries(edge_name).Neighbours(VOL) #?
                gfu_edge = self.gfu.vec.CreateVector()
            
                for bi, bb in enumerate(self.mesh.GetMaterials()):
                    if nb_dom.Mask()[bi]:
                        Vharm, aharm_mat, aharm_inv, E = self.vol_extensions[bb]
                
                        gfu_extension = GridFunction(Vharm) # Grid funciton on specific subdomain
                        res = gfu_extension.vec.CreateVector()

                        gfu_edge.data = self.gfu.vec  # Grid funciton on edge
                        # Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)
                        # Extension to subdomain * values on edge = function extended to subdomain
                        gfu_extension.vec.data = E * gfu_edge 
                        # Harmonic extension on edge
                        res = aharm_mat * gfu_extension.vec 
                        gfu_extension.vec.data = - aharm_inv * res
                        # Vharm.Embed(gfu_extension.vec, gfu_edge)
                        gfu_edge.data = E.T * gfu_extension.vec
                        self.gfu.vec.data += gfu_edge

                basis.Append(self.gfu.vec)







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

    def calc_vertex_basis(self, basis):
        for j, vertex_name in enumerate(self.mesh.GetBBoundaries()):
            gfu_vertex = self.gfu.vec.CreateVector()
            fd = self.V.GetDofs(self.mesh.BBoundaries(vertex_name))

            nb_edges = self.mesh.BBoundaries(vertex_name).Neighbours(BND)
            nb_dom = self.mesh.BBoundaries(vertex_name).Neighbours(VOL)

            self.gfu.vec[:] = 0
            self.gfu.vec[np.nonzero(fd)[0]] = 1 

            for bi, bb in enumerate(self.mesh.GetBoundaries()):
                if nb_edges.Mask()[bi]:
                    # Vharm, aharm_mat, aharm_inv = GetHarmonicExtensionEdge(bb)
                    Vharm, aharm_mat, aharm_inv, E = self.edge_extensions[bb]
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
                    self.gfu.vec.data += gfu_vertex
            
            gfu_edge = self.gfu.vec.CreateVector()
            
            for bi, bb in enumerate(self.mesh.GetMaterials()):
                if nb_dom.Mask()[bi]:
                    Vharm, aharm_mat, aharm_inv, E = self.vol_extensions[bb]
                    gfu_extension = GridFunction(Vharm)
                    gfu_extension.vec[:] = 0.0
                    res = gfu_extension.vec.CreateVector()
                    gfu_edge[:]=0.0
                    gfu_edge.data = self.gfu.vec
                    # Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)
                    gfu_extension.vec.data = E * gfu_edge
                    
                    res.data = aharm_mat * gfu_extension.vec
                    gfu_extension.vec.data = - aharm_inv * res
                    # Vharm.Embed(gfu_extension.vec, gfu_edge)
                    gfu_edge.data = E.T * gfu_extension.vec
                    self.gfu.vec.data += gfu_edge

            basis.Append(self.gfu.vec)





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

    def calc_bubble_basis(self, basis):
        for mat_name in self.mesh.GetMaterials():
            fd = self.V.GetDofs(self.mesh.Materials(mat_name)) & self.V.FreeDofs()
            Vloc = Compress(H1(self.mesh, order = self.order, dirichlet = self.dirichlet), fd)

            uloc, vloc = Vloc.TnT()
            aloc = BilinearForm(Vloc)
            aloc += grad(uloc) * grad(vloc) * dx()
            aloc.Assemble()

            mloc = BilinearForm(Vloc)
            mloc += uloc * vloc * dx()
            mloc.Assemble()

            AA = sp.csr_matrix(aloc.mat.CSR())
            MM = sp.csr_matrix(mloc.mat.CSR())
            ev, evec =scipy.sparse.linalg.eigs(A = AA, M = MM, k = self.bubble_modes, which='SM')
            evec = evec.transpose()

            ind = Vloc.ndof * [0]
            ii = 0
            for i, b in enumerate(fd):
                if b == True:
                    ind[ii] = i
                    ii += 1
            E = PermutationMatrix(self.V.ndof, ind)
            
            for e in evec:
                self.gfu.vec[:]=0.0
                # Vloc.Embed(e.real, gfu.vec)
                self.gfu.vec.data = E.T * e.real
                basis.Append(self.gfu.vec)

