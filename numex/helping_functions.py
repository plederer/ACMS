from ngsolve import *
# from netgen.geom2d import SplineGeometry

import scipy.linalg
import scipy.sparse as sp
import numpy as np
# from ngsolve.webgui import Draw
# from netgen.webgui import Draw as DrawGeo

import time


###############################################################

"""
**Extension operators**

Extension on interface: $E^{\Gamma} : H^{1/2}(\Gamma) \to H^1_D(\Omega)$ by $(E^{\Gamma} \tau)_{\mid\Omega_j} = E^{j} \tau_{\mid \partial \Omega_j}$, for all $j = 1,\ldots,J$. 

Extension on edges: $E^{\Gamma} : H^{1/2}_{00}(e) \to H^1_D(\Omega)$ via $E^{\Gamma} \tau = E^{\Gamma} E_0^e\tau$, where $E_0^e: H^{1/2}_{00}(e)  \to H^{1/2}(\Gamma)$ denotes the extension by zero to the interface $\Gamma$.

"""


def GetVertexNeighbours(vname, mesh):
    m = mesh.BBoundaries(vname).Neighbours(VOL).Mask()
    nb = BitArray(len(m))
    nb.Clear()
    for r in range(len(m)):
        if m[r] == 1:
            nb |= mesh.Materials(mesh.GetMaterials()[r]).Mask()
    return nb




###############################################################
# EXTENSIONS

class ACMS:
    def __init__(self, order, mesh, bm = 0, em = 0, mesh_info = None, bi = 0, alpha = 1):
        self.order = order # Polynomial degree of approximation
        self.dirichlet = mesh_info["dir_edges"]
        self.mesh = mesh
        self.V = H1(mesh, order = order, dirichlet = self.dirichlet)
        self.gfu = GridFunction(self.V)

        self.edge_extensions = {}
        self.vol_extensions = {}

        self.bubble_modes = bm
        self.edge_modes = em

        self.basis_v = MultiVector(self.gfu.vec, 0)
        self.basis_e = MultiVector(self.gfu.vec, 0)
        self.basis_b = MultiVector(self.gfu.vec, 0)
        
        self.alpha = alpha
        self.verts = mesh_info["verts"]
        self.edges = mesh_info["edges"]
        self.doms = list( dict.fromkeys(mesh.GetMaterials()) )
        

        self.bi = bi 

    # Define harmonic extension on specific subdomain
    # Returns the Sobolev space H^1_0(\Omega_j), the stiffness matrix and its inverse
    def GetHarmonicExtensionDomain(self, dom_name, kappa = 0):
        fd_all = self.V.GetDofs(self.mesh.Materials(dom_name)) # Dofs of specific domain
        base_space = H1(self.mesh, order = self.order, dirichlet = self.dirichlet) #Replicate H^1_0 on subdomain
        Vharm = Compress(base_space, fd_all)
        uharm, vharm = Vharm.TnT() # Trial and test functions
        aharm = BilinearForm(Vharm)
        #Setting bilinear form: - int (Grad u Grad v) d\Omega_j
        aharm += self.alpha * grad(uharm)*grad(vharm)*dx(definedon = self.mesh.Materials(dom_name), bonus_intorder = self.bi) #Why no alpha here works?
        if (kappa!= 0):
            aharm += -kappa**2 * uharm*vharm*dx(definedon = self.mesh.Materials(dom_name), bonus_intorder = self.bi)
        aharm.Assemble()
        aharm_inv = aharm.mat.Inverse(Vharm.FreeDofs(), inverse = "sparsecholesky")

        # Calc embedding - Local to global mapping
        # Computes global indices of local dofs 
        ind = Vharm.ndof * [0]
        ii = 0 # ii = index of local dofs
        for i, b in enumerate(fd_all): # i = index of global dofs
            if b == True: # If I am on a local dof -> save it and increase counter
                ind[ii] = i
                ii += 1
        E = PermutationMatrix(base_space.ndof, ind) # NGSolve for contructing mapping

        return Vharm, aharm.mat, aharm_inv, E


    # Define harmonic extension on specific subdomain
    # Returns the Sobolev space H^{1/2}_00(e), the stiffness matrix and its inverse
    def GetHarmonicExtensionEdge(self, edge_name, kappa = 0):
        fd_all = self.V.GetDofs(self.mesh.Boundaries(edge_name)) # Dofs of specific edge
        bnd = "" # Initialize empty boundary 
         # The space construction requires bc specified on the full domain 
        # so first we set Dirichlet everywhere and then remove internal vertices on our edge
        # This gives the edge vertices with Dirichlet bc
        for b in self.mesh.GetBoundaries():
            if (b != edge_name): # If the edge is not our specified edge, then add it to bnd 
                bnd += b + "|"
        bnd = bnd[:-1] # Remove the last added "|" - unnecessary
        base_space = H1(self.mesh, order = self.order, dirichlet = bnd)

        #Setting bilinear form: - int (Grad u Grad v) d"e". 
        Vharm = Compress(base_space, fd_all) #Sobolev space H^{1/2}_00(e)
        t = specialcf.tangential(2)  # Object to be evaluated - Tangential vector along edge (2=dimension)
        uharm, vharm = Vharm.TnT()
        aharm = BilinearForm(Vharm)
        aharm += (grad(uharm)*t) * (grad(vharm)*t) * ds(skeleton = True, definedon=self.mesh.Boundaries(edge_name), bonus_intorder = self.bi)
        aharm.Assemble()
        # Matrix in inverted only on internal dofs (FreeDofs) so it can be used for all edges
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
    def CalcHarmonicExtensions(self, kappa = 0, edge_names = None):
        if edge_names == None:
            edge_names = self.mesh.GetBoundaries()
        
        # remove double entries
        mats = tuple( dict.fromkeys(self.mesh.GetMaterials()) )

        for dom_name in mats:
            Vharm, aharm, aharm_inv, E = self.GetHarmonicExtensionDomain(dom_name, kappa = kappa)
            self.vol_extensions[dom_name] = [Vharm, aharm, aharm_inv, E]

        for edge_name in edge_names:
            Vharm, aharm, aharm_inv, E = self.GetHarmonicExtensionEdge(edge_name, kappa = kappa)
            self.edge_extensions[edge_name] = [Vharm, aharm, aharm_inv, E]



    """
    **Edge basis**

    For each $e\in\mathcal{E}$, for $i \in \mathbb{N}$, find $(\tau^e_i,\lambda^e_i)\in H^{1/2}_{00}(e) \times\mathbb{R}$ such that
    \begin{align}
    (\partial_e \tau^e_i,\partial_e \eta)_e =\lambda^e_i ( \tau^e_i, \eta)_e \quad \text{for all } \eta\in H^{1/2}_{00}(e).
    \end{align}
    """

    ###############################################################
    # EDGE MODES

    def CalcMaxEdgeModes(self):
        for edge_name in self.edges:
            dirverts = ""
            for v in self.verts:
                dirverts += v + "|"

            dirverts = dirverts[:-1]
            # vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(".*")) # Global vertices (coarse mesh)
            vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(dirverts)) # Global vertices (coarse mesh)

            fd = self.V.GetDofs(self.mesh.Boundaries(edge_name)) & (~vertex_dofs) 
            base_space = H1(self.mesh, order = self.order, dirichlet = self.dirichlet) # Creating Sobolev space
            Vloc = Compress(base_space, fd)

            if Vloc.ndof - 1 <= self.edge_modes:
                print("Maximum number of edge modes exceeded - All edge modes are used")
                self.edge_modes = Vloc.ndof - 2
            
    def calc_edge_basis(self, basis=None):
        if (basis == None):
            basis = self.basis_e
        self.CalcMaxEdgeModes()
        
        for edge_name in self.edges:
            vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(".*")) # Global vertices (coarse mesh)
            fd = self.V.GetDofs(self.mesh.Boundaries(edge_name)) & (~vertex_dofs) 
            # Vertices on a specific edge with boundaries removed (global vertices)
            base_space = H1(self.mesh, order = self.order, dirichlet = self.dirichlet) # Creating Sobolev space
            Vloc = Compress(base_space, fd) #Restricting Sobolev space on edge (with Dirichlet bc)

            uloc, vloc = Vloc.TnT() # Trial and test functions
            t = specialcf.tangential(2)
            #Setting bilinear form:  int (Grad u Grad v) de
            aloc = BilinearForm(Vloc)
            # This allows us to take the normal derivative of a function that is in H1 and computing the integral only on edges
            # Otherwise NGSolve does not allow to take the trace of a function in H^{1/2}(e) - uloc is defined on edge
            aloc += (grad(uloc)*t) * (grad(vloc)*t) * ds(skeleton=True, definedon = self.mesh.Boundaries(edge_name), bonus_intorder = self.bi)
            aloc.Assemble()
            #Setting bilinear form:  int u v de        
            mloc = BilinearForm(Vloc)
            mloc += uloc.Trace() * vloc.Trace() * ds(definedon = self.mesh.Boundaries(edge_name), bonus_intorder = self.bi)
            #mloc += uloc * vloc * ds(skeleton = True, edge_name)
            mloc.Assemble()

            # Solving eigenvalue problem: AA x = ev MM x
            AA = sp.csr_matrix(aloc.mat.CSR())
            MM = sp.csr_matrix(mloc.mat.CSR())
                
            ev, evec =sp.linalg.eigs(A = AA, M = MM, k = self.edge_modes, which='SM')
            idx = ev.argsort()[::]   
            ev = ev[idx]
            evec = evec[:,idx]
            evec = evec.transpose()

            # Local to global mapping
            ind = Vloc.ndof * [0]
            ii = 0
            for i, b in enumerate(fd):
                if b == True:
                    ind[ii] = i
                    ii += 1
            Eloc = PermutationMatrix(base_space.ndof, ind)

            for e in evec: # Going over eigenvectors
                # Vloc.Embed(e.real, gfu.vec)
                self.gfu.vec[:]=0.0
                self.gfu.vec.data = Eloc.T * e.real # Grid funciton on full mesh                

                nb_dom = self.mesh.Boundaries(edge_name).Neighbours(VOL) # It gives volumes that are neighbours of my edge
                gfu_edge = self.gfu.vec.CreateVector()
                gfu_edge[:] = 0.0
            
                for bi, bb in enumerate(self.mesh.GetMaterials()):
                    if nb_dom.Mask()[bi]:
                        Vharm, aharm_mat, aharm_inv, E = self.vol_extensions[bb]
                        # gfu_extension gfu_edge are auxiliary functions
                        gfu_extension = GridFunction(Vharm) # Grid funciton on specific subdomain
                        res = gfu_extension.vec.CreateVector()
                        gfu_extension.vec[:] = 0.0

                        gfu_edge.data = self.gfu.vec  # Grid funciton on edge
                        # Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)
                        # Extension to subdomain * values on edge = function extended to subdomain
                        gfu_extension.vec.data = E * gfu_edge 
                        # Restricting globally defined edge function to the subdomain I want
                        # Harmonic extension on edge
                        res[:] = 0.0
                        res = aharm_mat * gfu_extension.vec 
                        gfu_extension.vec.data = - aharm_inv * res
                        #Include Dirichlet bc because we loop over all subdomains to which we want to extend
                        # Vharm.Embed(gfu_extension.vec, gfu_edge)
                        gfu_edge.data = E.T * gfu_extension.vec
                        self.gfu.vec.data += gfu_edge # Boundary value stored
                
                
                # Draw(self.gfu, self.mesh, "basis")
                # input()
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

    def calc_vertex_basis(self, basis=None):
        if (basis == None):
            basis = self.basis_v
        for j, vertex_name in enumerate(self.verts):
            gfu_vertex = self.gfu.vec.CreateVector() # Initialise grid function for vertices
            fd = self.V.GetDofs(self.mesh.BBoundaries(vertex_name)) # Gets coarse vertex representation on full mesh

            nb_edges = self.mesh.BBoundaries(vertex_name).Neighbours(BND) # Neighbouring edges (geometric object - region)
            nb_dom = self.mesh.BBoundaries(vertex_name).Neighbours(VOL) # Neighbouring subdomains (geometric object - region)
            # nb_dom = GetVertexNeighbours(vertex_name, self.mesh)
            # print(nb_dom)

            self.gfu.vec[:] = 0
            self.gfu.vec[np.nonzero(fd)[0]] = 1  # Set the grid function to one in the current vertex

            # First extend to edges
            for bi, bb in enumerate(self.mesh.GetBoundaries()):
                if nb_edges.Mask()[bi]:  # If the edge is in the neighbourhood of the vertex ... extend
                    # Vharm, aharm_mat, aharm_inv = GetHarmonicExtensionEdge(bb)
                    Vharm, aharm_mat, aharm_inv, E = self.edge_extensions[bb] # Extension to the edge(s)
                    gfu_extension = GridFunction(Vharm) # Auxiliary function on harmonic space
                    gfu_extension.vec[:] = 0.0 # Initializing to 0
                    res = gfu_extension.vec.CreateVector() #
                    res[:]=0.0
                    # Set the grid function to one in the current vertex AGAIN. The extension sets it to 0 again.
                    gfu_vertex[:] = 0
                    gfu_vertex[np.nonzero(fd)[0]] = 1 

                    # Extend to current edge
                    # Q: Why are we using * product which is component-wise
                    gfu_extension.vec.data = E * gfu_vertex # Extend to current edge
                    # Vharm.EmbedTranspose(gfu_vertex, gfu_extension.vec)
                    res.data = aharm_mat * gfu_extension.vec
                    # # # only harmonic extension to one edge
                    # # # has zero vertex value! 
                    # Which is why we need to set it again to 1 in every loop
                    gfu_extension.vec.data = - aharm_inv * res
                    # Vharm.Embed(gfu_extension.vec, gfu_vertex)
                    gfu_vertex.data = E.T * gfu_extension.vec
                    self.gfu.vec.data += gfu_vertex # Storing the current extension
            
            
            gfu_edge = self.gfu.vec.CreateVector()
            
            # Then extend to subdomains
            for bi, bb in enumerate(self.mesh.GetMaterials()):
                if nb_dom.Mask()[bi]: # If the subdomain is on extended edges.. extend in subdomain
                    Vharm, aharm_mat, aharm_inv, E = self.vol_extensions[bb] # Extension to subdomain
                    gfu_extension = GridFunction(Vharm) # Auxiliary function on harmonic space
                    gfu_extension.vec[:] = 0.0 # Initializing to 0
                    res = gfu_extension.vec.CreateVector() #
                    gfu_edge[:]=0.0 # Initializing to 0
                    gfu_edge.data = self.gfu.vec # Storing the edge extensions
                    # Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)

                    # Extend on subdomain
                    gfu_extension.vec.data = E * gfu_edge
                    res.data = aharm_mat * gfu_extension.vec
                    gfu_extension.vec.data = - aharm_inv * res
                    # Vharm.Embed(gfu_extension.vec, gfu_edge)
                    gfu_edge.data = E.T * gfu_extension.vec
                    self.gfu.vec.data += gfu_edge
            
            if (Norm(self.gfu.vec) > 1):
                basis.Append(self.gfu.vec)





    """
    **Bubble functions**

    **Helmholtz case:** Let us define the local sesquilinear form $\mathcal{A}{j}: H^1(\Omega_j) \times H^1(\Omega_j) \to \mathbb{C}$ with domain of integration $\Omega_j$ instead of $\Omega$.
    Since $\mathcal{A}_{j}$ is Hermitian, we can consider the eigenproblems: for $j=1,...,J$ and $i \in \mathbb{N}$, find $(b_i^j,\lambda_i^j)\in H)0^1(\Omega_j) \times \mathbb{R}$ such that
    \begin{align}
        \mathcal{A}_{j}(b_i^j,v)= \lambda_i^j ( {\kappa^2} b_i^j,v)_{\Omega_j} \quad \forall v \in H_0^1(\Omega_j).
    \end{align}
    """
    ###############################################################
    # BUBBLE FUNCTIONS

    def CalcMaxBubbleModes(self):
        for mat_name in self.doms: #self.mesh.GetMaterials(): # Subdomains labels
            # DOFS that are in the interior of the subdomain (excludes edges)
            fd = self.V.GetDofs(self.mesh.Materials(mat_name)) & self.V.FreeDofs()
            Vloc = Compress(H1(self.mesh, order = self.order, dirichlet = self.dirichlet), fd)
            
            #Control on the maximum number of used edges, so it does not crash
            if Vloc.ndof - 1 <= self.bubble_modes:
                print("Maximum number of bubble modes exeeded - All bubble modes are used")
                self.bubble_modes = Vloc.ndof - 2

    
    def calc_bubble_basis(self, basis=None):
        if (basis == None):
            basis = self.basis_b
        self.CalcMaxBubbleModes()
            
        for mat_name in self.doms: # Subdomains labels
            # DOFS that are in the interior of the subdomain (excludes edges)
            fd = self.V.GetDofs(self.mesh.Materials(mat_name)) & self.V.FreeDofs()
            Vloc = Compress(H1(self.mesh, order = self.order, dirichlet = self.dirichlet), fd)

            #Setting bilinear form: int (Grad u Grad v) d\Omega_j
            uloc, vloc = Vloc.TnT()
            aloc = BilinearForm(Vloc)
            aloc += self.alpha * grad(uloc) * grad(vloc) * dx(bonus_intorder = self.bi)
            aloc.Assemble()

            #Setting bilinear form: int  u v d\Omega_j
            mloc = BilinearForm(Vloc)
            mloc += uloc * vloc * dx(bonus_intorder = self.bi)
            mloc.Assemble()

            # Solving eigenvalue problem: AA x = ev MM x
            AA = sp.csr_matrix(aloc.mat.CSR())
            MM = sp.csr_matrix(mloc.mat.CSR())
            
                
            ev, evec =scipy.sparse.linalg.eigs(A = AA, M = MM, k = self.bubble_modes, which='SM')
            idx = ev.argsort()[::]   
            ev = ev[idx]
            evec = evec[:,idx]
            evec = evec.transpose()


            # Local to global mapping
            ind = Vloc.ndof * [0]
            ii = 0
            for i, b in enumerate(fd):
                if b == True:
                    ind[ii] = i
                    ii += 1
            E = PermutationMatrix(self.V.ndof, ind)
            
            for e in evec: # Going over eigenvectors
                self.gfu.vec[:]=0.0
                # Vloc.Embed(e.real, gfu.vec)
                self.gfu.vec.data = E.T * e.real # Grid funciton on full mesh
                basis.Append(self.gfu.vec)
            
            

    def calc_basis(self):
        start_time = time.time()
        self.calc_vertex_basis() 
        vertex_time = time.time() 
        # print("Vertex basis functions computation in --- %s seconds ---" % (vertex_time - start_time))
        self.calc_edge_basis()
        edges_time = time.time() 
        # print("Edge basis functions computation in --- %s seconds ---" % (edges_time - vertex_time))
        self.calc_bubble_basis()
        bubbles_time = time.time() 
        # print("Bubble basis functions computation in --- %s seconds ---" % (bubbles_time - edges_time))
        return self.basis_v, self.basis_e, self.basis_b
    
    def complex_basis(self):
        Vc = H1(self. mesh, order = self.order, complex = True)
        gfu = GridFunction(Vc)
        basis = MultiVector(gfu.vec, 0)

        for bv in self.basis_v:
            gfu.vec.FV()[:] = bv
            basis.Append(gfu.vec)

        for be in self.basis_e:
            gfu.vec.FV()[:] = be
            basis.Append(gfu.vec)
                
        for bb in self.basis_b:
            gfu.vec.FV()[:] = bb
            basis.Append(gfu.vec)
        
        return basis
        