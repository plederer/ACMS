from ngsolve import *
# from netgen.geom2d import SplineGeometry

from ngsolve.la import Real2ComplexMatrix
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
    def __init__(self, order, mesh, bm = 0, em = 0, mesh_info = None, bi = 0, alpha = 1, kappa = 1, omega = 1, f = 1, g = 1):
        self.order = order # Polynomial degree of approximation
        self.dirichlet = mesh_info["dir_edges"]
        self.dom_bnd = mesh_info["dom_bnd"] 
        self.mesh = mesh
        self.V = H1(mesh, order = order, dirichlet = self.dirichlet)
        self.Vc = H1(mesh, order = order, dirichlet = self.dirichlet, complex = True)
        self.gfu = GridFunction(self.V)
        self.gfuc = GridFunction(self.Vc)

        self.f = f 
        self.g = g 

        self.edge_extensions = {}
        self.vol_extensions = {}

        self.bubble_modes = bm
        self.edge_modes = em
        self.edgeversions = {}
        # self.cellversions = {}

        # self.basis_all = MultiVector(self.gfuc.vec, 0) 
        
        self.alpha = alpha
        self.kappa = kappa
        self.omega = omega
        self.verts = mesh_info["verts"]
        self.edges = mesh_info["edges"]
        self.doms = list( dict.fromkeys(mesh.GetMaterials()) )
        
        self.FreeVertices = BitArray(len(mesh.GetBBoundaries()))
        for v in range(len(mesh.GetBBoundaries())):
            if not "inner_vertex" in mesh.ngmesh.GetCD2Name(v):
                self.FreeVertices[v] = 1
            else:
                self.FreeVertices[v] = 0


        self.FreeEdges = BitArray(len(mesh.GetBoundaries()))
        for e in range(len(mesh.GetBoundaries())):
            if not "inner_edge" in mesh.ngmesh.GetBCName(e):
                self.FreeEdges[e] = 1
            else:
                self.FreeEdges[e] = 0
        
        self.nverts = len(self.verts)
        self.nedges = len(self.edges)
        self.ncells = len(self.doms)

        self.acmsdofs = self.nverts + self.nedges * self.edge_modes + self.ncells  * self.bubble_modes

        self.basis_v = MultiVector(self.gfuc.vec, self.nverts)
        self.basis_e = MultiVector(self.gfuc.vec, self.nedges * self.edge_modes)
        self.basis_b = MultiVector(self.gfuc.vec, self.ncells * self.bubble_modes)
        
        self.asmall = Matrix(self.acmsdofs, self.acmsdofs, complex = True)
        self.asmall[:,:] = 0 + 0*1J
        self.fsmall = Vector(self.acmsdofs, complex = True)
        self.fsmall[:] = 0 + 0*1J

        self.localbasis = {}

        self.bi = bi 

    # Define harmonic extension on specific subdomain
    # Returns the Sobolev space H^1_0(\Omega_j), the stiffness matrix and its inverse
    def GetHarmonicExtensionDomain(self, dom_name):
        # start = time.time()
        fd_all = self.V.GetDofs(self.mesh.Materials(dom_name)) # Dofs of specific domain
        # print(fd_all)
        # print("dom_name = ", dom_name)
        # input()
        base_space = H1(self.mesh, order = self.order, complex = True) #, dirichlet = self.dirichlet) #Replicate H^1_0 on subdomain
        # print(self.dirichlet)
        Vharm = Compress(base_space, fd_all)

        fd = Vharm.FreeDofs()
        edges = self.mesh.Materials(dom_name).Neighbours(BND).Split()
        for bnds in edges[:-1]:
            fd = fd & ~Vharm.GetDofs(bnds)
        

        uharm, vharm = Vharm.TnT() # Trial and test functions
        aharm = BilinearForm(Vharm)
        #Setting bilinear form: - int (Grad u Grad v) d\Omega_j
        aharm += self.alpha * grad(uharm)*grad(vharm)*dx(definedon = self.mesh.Materials(dom_name), bonus_intorder = self.bi) #Why no alpha here works?
        # if (kappa!= 0):
        aharm += -self.kappa**2 * uharm*vharm*dx(definedon = self.mesh.Materials(dom_name), bonus_intorder = self.bi)
        aharm.Assemble()

        # print(Vharm.FreeDofs())
        # aharm_inv = aharm.mat.Inverse(Vharm.FreeDofs() & ~fd, inverse = "sparsecholesky")
        aharm_inv = aharm.mat.Inverse(fd, inverse = "sparsecholesky")
        # print("setp = ", time.time() - start)
        # start = time.time()
    
        # t = specialcf.tangential(2) 
        # aharm_edge = BilinearForm(Vharm, check_unused=False)
        # aharm_edge += (grad(uharm)*t) * (grad(vharm)*t) * ds(skeleton = True, definedon = self.mesh.Boundaries(local_dom_bnd), bonus_intorder = self.bi)
        # # aharm_edge += (Grad(uharm)*t) * (Grad(vharm)*t) * dx(element_boundary = True, 
        #                                                     #  bonus_intorder = self.bi)
        # aharm_edge.Assemble()
        
        
        # for i in range(size)
        
        # fd[0:sum(Vharm.GetDofs(self.mesh.Materials(dom_name).Neighbours(BBND)))] = 0
        # fd[0:5] = 0
        # aharm_edge_inv = aharm_edge.mat.Inverse(fd, inverse = "sparsecholesky")
        # print("setp = ", time.time() - start)
        # input()
        # self.cellversions[voltype] = [aharm.mat, aharm_inv, aharm_edge.mat, aharm_edge_inv]
        # E = 0
        # return Vharm, aharm.mat, aharm_inv, E, aharm_edge.mat, aharm_edge_inv
        return Vharm, aharm.mat, aharm_inv #, aharm_edge.mat, aharm_edge_inv


    # Define harmonic extension on specific subdomain
    # Returns the Sobolev space H^{1/2}_00(e), the stiffness matrix and its inverse
    def GetHarmonicExtensionEdge(self, edge_name, kappa = 0):
        # start = time.time()
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
        # end = time.time()
        # print(edge_name + " takes: {} ".format( end - start))
        return Vharm, aharm.mat, aharm_inv, E



    # Function that computes the harmonic extensions on all subdomains and on all edges (of coarse mesh)
    # Returns vol_extensions and edge_extensions
    def CalcHarmonicExtensions(self, kappa = 0, edge_names = None):
        if edge_names == None:
            edge_names = self.mesh.GetBoundaries()
        
        # remove double entries
        mats = tuple( dict.fromkeys(self.mesh.GetMaterials()) )
        
        for dom_name in mats:
            # Vharm, aharm, aharm_inv, E, aharm_edge, aharm_edge_inv = self.GetHarmonicExtensionDomain(dom_name, kappa = kappa)
            self.vol_extensions[dom_name] = list(self.GetHarmonicExtensionDomain(dom_name))
        
        # for edge_name in edge_names:
        #     Vharm, aharm, aharm_inv, E = self.GetHarmonicExtensionEdge(edge_name, kappa = kappa)
        #     self.edge_extensions[edge_name] = [Vharm, aharm, aharm_inv, E]
        
    def Assemble_localA(self, acms_cell):
        nbnd = self.mesh.Materials(acms_cell).Neighbours(BND)
        nbbnd = self.mesh.Materials(acms_cell).Neighbours(BBND)

        vertices = nbbnd.Mask() & self.FreeVertices
        edges = nbnd.Mask() & self.FreeEdges
        
        Vharm, aharm_mat, aharm_inv = self.vol_extensions[acms_cell]
        
        local_vertex_dofs = Vharm.GetDofs(nbbnd)
        
        gfu = GridFunction(Vharm)
        localbasis = MultiVector(gfu.vec, 4 + 4 * self.edge_modes)

        dofs = []
        lii = 0
        for i, b in enumerate(vertices):
            if b == 1:
                # derive acms dof numbering 
                for j in range(self.nverts):
                    if self.verts[j][0] == i:
                        dofs.append(j)
                # add corresponding vertex basis function
                #vertex name
                vname = self.mesh.GetBBoundaries()[i]
                # vertex_nr = np.nonzero(self.mesh.BBoundaries(vname).Mask())[0][0]
                # print("vertex_nr =", vertex_nr)
                # print("i =", i)
                vnbnd = nbnd * self.mesh.BBoundaries(vname).Neighbours(BND)
                # print(vnbnd)
                # input()

                # ddofs2 = Vharm.GetDofs(vnbnd)
                # # print(ddofs2)
                # # input()
                
                Vxcoord = self.mesh.vertices[i].point[0]
                Vycoord = self.mesh.vertices[i].point[1]
                
                # slength = Integrate(1, self.mesh, definedon=vnbnd, order = 0)/2

                
                for bnds in vnbnd.Split():
                    
                    # ddofs = Vharm.GetDofs(bnds)
                    # print(sum(ddofs))
                    # fac = [1-f/(sum(ddofs)-1) for f in range(sum(ddofs))]
                    # print(fac)
                    
                    # ii = 0
                    # for d, bb in enumerate(ddofs):
                    #     if ii ==1:
                    #         ii +=1
                    #     if bb == 1 and ii < sum(ddofs)-1:
                    #         gfu.vec[d] = fac[ii]
                    #         print(fac[ii])
                    #         ii +=1

                    slength = Integrate(1, self.mesh, definedon=bnds, order = 0)
                    for e in bnds.Elements():
                        edofs = Vharm.GetDofNrs(e)
                        # print(edofs)
                        for vi, v in enumerate(e.vertices):
                            xcoord = self.mesh.vertices[v.nr].point[0]
                            ycoord = self.mesh.vertices[v.nr].point[1] 
                            vlen = sqrt((xcoord - Vxcoord)**2 + (ycoord - Vycoord)**2)
                            # print(1-vlen/0.484)
                            gfu.vec[edofs[vi]] = 1-vlen/slength
                    # Draw(gfu)
                    # input()

                
                # ddofs = Vharm.GetDofs(self.mesh.BBoundaries(vname))
                # for d, bb in enumerate(ddofs):
                #     if bb == 1:
                #         gfu.vec[d] = 1
                
                # gfu.vec.data += -(aharm_edge_inv @ aharm_edge_mat) * gfu.vec 
                gfu.vec.data += -(aharm_inv @ aharm_mat) * gfu.vec  
                # print(Norm(gfu.vec))
                # Draw(gfu)
                # input()
                localbasis[lii][:] = gfu.vec
                lii +=1
                gfu.vec[:] = 0
        
        local_dom_bnd = ""
        for i, b in enumerate(edges):
            if b == 1:
                for j in range(self.nedges):
                    if self.edges[j][0] == i:
                        for l in range(self.edge_modes):
                            dofs.append(self.nverts + j*self.edge_modes + l)

                bndname = self.mesh.GetBoundaries()[i]
                if bndname in self.dom_bnd:
                    local_dom_bnd += bndname + "|"
                ddofs = Vharm.GetDofs(self.mesh.Boundaries(bndname)) & (~local_vertex_dofs)
                edgetype = ""
                if "V" in bndname:
                    edgetype = "V"
                elif "H" in bndname:
                    edgetype = "H"

                for l in range(self.edge_modes):
                    ii = 0
                    for d, bb in enumerate(ddofs):
                        if bb == 1:
                            gfu.vec[d] = self.edgeversions[edgetype][1][l,ii]#.real
                            ii+=1
                        
                    gfu.vec.data += -(aharm_inv @ aharm_mat) * gfu.vec
                    localbasis[lii][:] = gfu.vec
                    lii+=1
                    gfu.vec[:] = 0
        
        self.localbasis[acms_cell] = (localbasis, dofs)
        uharm, vharm = Vharm.TnT() 
        local_a = BilinearForm(Vharm, check_unused=False)
        
        # local_a += self.alpha * grad(uharm)*grad(vharm)*dx(definedon = self.mesh.Materials(acms_cell), bonus_intorder = self.bi) 
        # local_a += -self.kappa**2 * uharm*vharm*dx(definedon = self.mesh.Materials(acms_cell), bonus_intorder = self.bi)
        beta = -1
        

        local_dom_bnd = local_dom_bnd[:-1]
        if local_dom_bnd != "":
            local_a += -1J * self.omega * beta * uharm * vharm * ds(local_dom_bnd) #, bonus_intorder = 10)
        local_a.Assemble()

        local_a.mat.AsVector().data += aharm_mat.AsVector()
        

        localmat = InnerProduct(localbasis, (local_a.mat * localbasis).Evaluate(), conjugate = False)
        

        local_f = LinearForm(Vharm)
        # local_f += f * vharm * dx(definedon = self.mesh.Materials(acms_cell)) #bonus_intorder=10)
        local_f += self.g * vharm * ds(local_dom_bnd)
        local_f.Assemble()
        

        localvec = InnerProduct(localbasis, local_f.vec, conjugate = False)
        # print(localvec)
        # print("cell = {}, localvec = {}, local_f = {}".format(acms_cell, Norm(localvec), Norm(local_f.vec)))
        # input()
        for i in range(len(dofs)):
            for j in range(len(dofs)): 
                self.asmall[dofs[i],dofs[j]] += localmat[i,j]
            
            self.fsmall[dofs[i]] += localvec[i]
                
        
    def SetGlobalFunction(self, gfu, coeffs):
        for acms_cell in self.doms:
            Vharm, aharm_mat, aharm_inv, E, aharm_edge_mat, aharm_edge_inv = self.vol_extensions[acms_cell]

            edges = self.mesh.Materials(acms_cell).Neighbours(BND).Mask() & self.FreeEdges
            edge_names = ""
            for i, b in enumerate(edges):
                if b == 1:
                    bname = self.mesh.GetBoundaries()[i]
                    if bname not in self.dom_bnd:
                        edge_names += self.mesh.GetBoundaries()[i] + "|"
            edge_names = edge_names[:-1]
            # print(edge_names)

            vertices = self.mesh.Materials(acms_cell).Neighbours(BBND).Mask() & self.FreeVertices
            # v_names_2 = ""
            v_names = ""
            for i, b in enumerate(vertices):
                if b == 1:
                    vname = self.mesh.GetBBoundaries()[i]
                    # print(sum(self.mesh.BBoundaries(vname).Neighbours(BND).Mask()))
                    if sum(self.mesh.BBoundaries(vname).Neighbours(BND).Mask()) == 4:
                        v_names += vname + "|"
                    # else:
                        # v_names_4 += vname + "|"
            v_names = v_names[:-1]
            # v_names_4 = v_names_4[:-1]
            # print(v_names)

            dofs = self.localbasis[acms_cell][1]
            localcoeffs = Vector(len(dofs),complex = True)
            # print(coeffs)

            for d in range(len(dofs)):
                localcoeffs[d] = coeffs[dofs[d]]
            # print(localcoeffs)
        
            localvec = (self.localbasis[acms_cell][0] * localcoeffs).Evaluate()
            # print(localvec)

            for i in range(Vharm.ndof):
                if (Vharm.GetDofs(self.mesh.Boundaries(edge_names))[i] == 1):
                    if (Vharm.GetDofs(self.mesh.BBoundaries(v_names))[i] == 0):
                        # print(localvec[i])
                        localvec[i] = localvec[i]* 0.5
                        # print(localvec[i])
                        # input()
                    else:
                        localvec[i] = localvec[i] * 0.25

            # fd_all = self.Vc.GetDofs(self.mesh.Materials(acms_cell)) # Dofs of specific domain
            # base_space = H1(self.mesh, order = self.order, dirichlet = self.dirichlet, complex = True) #Replicate H^1_0 on subdomain
            # Vharmc = Compress(base_space, fd_all)  
            # localgfu = GridFunction(Vharmc)
            # localgfu.vec.data = localvec
            # Draw(localgfu)
            # input()
            # print(localvec)
            # print(scale_mat)
            # localvec[:] = scale_mat * localvec
            # print(localvec)
            # input()
            fd_all = self.Vc.GetDofs(self.mesh.Materials(acms_cell))
            ii = 0
            for i, b in enumerate(fd_all):
                if b == True:
                    gfu.vec.FV()[i] += localvec[ii]
                    ii+=1
            # input()
            # helpvec = (E * localvec).Evaluate()
            # for i in range(len(gfu.vec)):
            #     gfu.vec.FV()[i] += helpvec[i]


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
        
        check_version = True
        edgeversions = {}

        ee = 0
        
        for edge_name in self.edges:
            ss = time.time()
            dirverts = ""
            for v in self.verts:
                dirverts += v[1] + "|"

            dirverts = dirverts[:-1]
            # vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(".*")) # Global vertices (coarse mesh)
            vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(dirverts)) # Global vertices (coarse mesh)

           
            fd = self.V.GetDofs(self.mesh.Boundaries(edge_name[1])) & (~vertex_dofs) & (~self.V.FreeDofs())
            
            # base_space = H1(self.mesh, order = self.order, dirichlet = self.dirichlet) # Creating Sobolev space
            
            # Vloc = Compress(base_space, fd)
            
            
            # print(Vloc.ndof - sum(fd))
            

            # if sum(fd) - 1 <= self.edge_modes:
            #     print("Maximum number of edge modes exceeded - All edge modes are used")
                # self.edge_modes = Vloc.ndof - 2
            ee += time.time() - ss
        # print("compressions = ", ee)
            
    def calc_edge_basis(self, basis = None):
        if (basis == None):
            basis = self.basis_e
        # if calc_all == True:
        #     basis = self.basis_all
        # else:
        #     basis = self.basis_e
        # # sstart = time.time()
        # self.CalcMaxEdgeModes()
        # print("max", time.time() - sstart )
        # quit()
        # print("number of edges = ",len(self.edges))

        check_version = True

        # stores ndof and evecs
        # selfedgeversions = {}
        
        ee = 0
        eeig = 0

        iie = 0

        if self.edge_modes > 0:
            for edge_name in self.edges:
                edgetype = ""
                # edgestart = time.time()
                if "V" in edge_name[1]:
                    edgetype = "V"
                elif "H" in edge_name[1]:
                    edgetype = "H"
                else:
                    raise Exception("wrong edge type")
                

                # vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(".*")) # Global vertices (coarse mesh)
                # fd = self.V.GetDofs(self.mesh.Boundaries(edge_name)) & (~vertex_dofs) 

                
                vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(".*")) # Global vertices (coarse mesh)
                fd = self.V.GetDofs(self.mesh.Boundaries(edge_name[1])) & (~vertex_dofs) 

                # edgebasis = MultiVector(self.gfu.vec, self.edge_modes)

                if edgetype not in self.edgeversions:                           
                    ndofs = sum(fd)
                    # edgeversions[edgetype] = [ndofs]

                    base_space = H1(self.mesh, order = self.order, dirichlet = self.dirichlet) # Creating Sobolev space
                    Vloc = Compress(base_space, fd) #Restricting Sobolev space on edge (with Dirichlet bc)
                    uloc, vloc = Vloc.TnT() # Trial and test functions
                    t = specialcf.tangential(2)
                    
                    #Setting bilinear form:  int (Grad u Grad v) de
                    aloc = BilinearForm(Vloc, symmetric = True)
                    # This allows us to take the normal derivative of a function that is in H1 and computing the integral only on edges
                    # Otherwise NGSolve does not allow to take the trace of a function in H^{1/2}(e) - uloc is defined on edge
                    aloc += (grad(uloc)*t) * (grad(vloc)*t) * ds(skeleton=True, definedon = self.mesh.Boundaries(edge_name[1]), bonus_intorder = self.bi)
                    aloc.Assemble()
                    
                    #Setting bilinear form:  int u v de        
                    mloc = BilinearForm(Vloc, symmetric = True)
                    mloc += uloc.Trace() * vloc.Trace() * ds(skeleton = True, definedon = self.mesh.Boundaries(edge_name[1]), bonus_intorder = self.bi)
                    mloc.Assemble()
                    
                    
                    # Solving eigenvalue problem: AA x = ev MM x
                    AA = sp.csr_matrix(aloc.mat.CSR())
                    MM = sp.csr_matrix(mloc.mat.CSR())
                    
                    ev, evec =sp.linalg.eigs(A = AA, M = MM, k = self.edge_modes, which='SM')
                    idx = ev.argsort()[::]   
                    ev = ev[idx]
                    evec = evec[:,idx]
                    evec = evec.transpose()
                    self.edgeversions[edgetype] = [ndofs, evec]
                
                # eigend = time.time()
                # eeig += eigend - eigstart
                
                # edgeversions[edgetype][0] * [0]
                # permstart = time.time()

                # ind = []
                # start = time.time()
                # for i, b in enumerate(fd): #edgeversions[edgetype][0]):
                #     if b == True:
                #         ind = [i + j for j in range(edgeversions[edgetype][0])]
                        # ind.append(i)
                        # ind[ii] = i
                        # ii += 1
                # print(ind)
                # quit()
                
                
                # ind = list(np.nonzero(fd)[0])
                # # ind = list(np.argmax(fd == 1))
                # # print(ind)
                # # ee += time.time() - permstart
                # ee += time.time() - start
                # Eloc = PermutationMatrix(self.V.ndof, ind)
                
                # for i,e in enumerate(self.edgeversions[edgetype][1]):
                #         edgebasis[i] = Eloc.T * e.real
                
                
                # if (str(ndofs) in edgeversions) and check_version:
                #     evec = edgeversions[str(ndofs)]
                #     for i,e in enumerate(evec):
                #         edgebasis[i] = Eloc.T * e.real
                # else:
                #     # print("DO CALC")
                #     base_space = H1(self.mesh, order = self.order, dirichlet = self.dirichlet) # Creating Sobolev space
                #     Vloc = Compress(base_space, fd) #Restricting Sobolev space on edge (with Dirichlet bc)
                #     uloc, vloc = Vloc.TnT() # Trial and test functions
                #     t = specialcf.tangential(2)
                    
                #     #Setting bilinear form:  int (Grad u Grad v) de
                #     aloc = BilinearForm(Vloc, symmetric = True)
                #     # This allows us to take the normal derivative of a function that is in H1 and computing the integral only on edges
                #     # Otherwise NGSolve does not allow to take the trace of a function in H^{1/2}(e) - uloc is defined on edge
                #     aloc += (grad(uloc)*t) * (grad(vloc)*t) * ds(skeleton=True, definedon = self.mesh.Boundaries(edge_name), bonus_intorder = self.bi)
                #     aloc.Assemble()
                    
                #     #Setting bilinear form:  int u v de        
                #     mloc = BilinearForm(Vloc, symmetric = True)
                #     mloc += uloc.Trace() * vloc.Trace() * ds(skeleton = True, definedon = self.mesh.Boundaries(edge_name), bonus_intorder = self.bi)
                #     mloc.Assemble()
                    
                    
                #     # Solving eigenvalue problem: AA x = ev MM x
                #     AA = sp.csr_matrix(aloc.mat.CSR())
                #     MM = sp.csr_matrix(mloc.mat.CSR())
                    
                #     ev, evec =sp.linalg.eigs(A = AA, M = MM, k = self.edge_modes, which='SM')
                #     idx = ev.argsort()[::]   
                #     ev = ev[idx]
                #     evec = evec[:,idx]
                #     evec = evec.transpose()
                    
                #     if check_version:
                #         edgeversions[str(Vloc.ndof)] = evec
                #     for i,e in enumerate(evec):
                #         edgebasis[i] = Eloc.T * e.real
                
                # nb_dom = self.mesh.Boundaries(edge_name[1]).Neighbours(VOL) # It gives volumes that are neighbours of my edge
                
                # for bi, bb in enumerate(self.mesh.GetMaterials()):
                #     if nb_dom.Mask()[bi]:
                #         Vharm, aharm_mat, aharm_inv, E, aharm_edge, aharm_edge_inv = self.vol_extensions[bb]
                #         edgebasis.data += -(E.T @ aharm_inv @ aharm_mat @ E) * edgebasis
                
                
                
                # for i in range(len(evec)):
                
                #     # self.gfuc.vec.FV()[:] = edgebasis[i]
                #     # end = time.time()
                    
                #     basis[iie] = edgebasis[i] #self.gfuc.vec
                #     iie += 1
                    # basis.Append(self.gfuc.vec)
                
                    
        # print("time eigenvalues = ", eeig)
        # print("time perm = ", ee)
                # old version
    
                # for e in evec: # Going over eigenvectors
                #     # Vloc.Embed(e.real, gfu.vec)
                #     self.gfu.vec[:]=0.0
                #     self.gfu.vec.data = Eloc.T * e.real # Grid funciton on full mesh                

                #     nb_dom = self.mesh.Boundaries(edge_name).Neighbours(VOL) # It gives volumes that are neighbours of my edge
                #     gfu_edge = self.gfu.vec.CreateVector()
                #     gfu_edge[:] = 0.0
                    
                #     for bi, bb in enumerate(self.mesh.GetMaterials()):
                #         if nb_dom.Mask()[bi]:
                #             Vharm, aharm_mat, aharm_inv, E = self.vol_extensions[bb]
                #             # gfu_extension gfu_edge are auxiliary functions
                #             gfu_extension = GridFunction(Vharm) # Grid funciton on specific subdomain
                #             res = gfu_extension.vec.CreateVector()
                #             # gfu_extension.vec[:] = 0.0

                #             # gfu_edge.data = self.gfu.vec  # Grid funciton on edge
                #             # Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)
                #             # Extension to subdomain * values on edge = function extended to subdomain
                #             gfu_extension.vec.data = E * self.gfu.vec 
                #             # Restricting globally defined edge function to the subdomain I want
                #             # Harmonic extension on edge
                #             # res[:] = 0.0
                #             res = aharm_mat * gfu_extension.vec 
                #             gfu_extension.vec.data = - aharm_inv * res
                #             #Include Dirichlet bc because we loop over all subdomains to which we want to extend
                #             # Vharm.Embed(gfu_extension.vec, gfu_edge)
                #             # gfu_edge.data = E.T * gfu_extension.vec
                #             self.gfu.vec.data += -(E.T @ aharm_inv @ aharm_mat @ E) * self.gfu.vec #gfu_extension.vec # Boundary value stored
                    # Draw(self.gfu, self.mesh, "basis")
                    # input()
                    # basis.Append(self.gfu.vec)  

        # quit()





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

    def calc_vertex_basis(self, basis = None):
        if basis == None:
            basis = self.basis_v

        ee = 0
        iiv = 0
        for j, vertex_name in enumerate(self.verts):
            # gfu_vertex = self.gfu.vec.CreateVector() # Initialise grid function for vertices
            # fd = self.V.GetDofs(self.mesh.BBoundaries(vertex_name)) # Gets coarse vertex representation on full mesh
            
            reg = self.mesh.BBoundaries(vertex_name[1])

            nb_edges = reg.Neighbours(BND) # Neighbouring edges (geometric object - region)
            nb_dom = reg.Neighbours(VOL) # Neighbouring subdomains (geometric object - region)
            vertex_nr = np.nonzero(reg.Mask())[0]

            self.gfu.vec[:] = 0
            self.gfu.vec[vertex_nr] = 1  # Set the grid function to one in the current vertex
        
            # First extend to edges
            
            for bi, bb in enumerate(self.mesh.GetBoundaries()):
                if nb_edges.Mask()[bi]:  # If the edge is in the neighbourhood of the vertex ... extend
                    #old version
                    if False:
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
                    else:
                        # this does not work for curved boundaries!
                        
                        Vxcoord = self.mesh.vertices[vertex_nr].point[0]
                        Vycoord = self.mesh.vertices[vertex_nr].point[1]
                        
                        length = Integrate(1, self.mesh, definedon=self.mesh.Boundaries(bb), order = 0)
                        for e in self.mesh.Boundaries(bb).Elements():
                            for v in e.vertices:
                                if v.nr != vertex_nr: 
                                    xcoord = self.mesh.vertices[v.nr].point[0]
                                    ycoord = self.mesh.vertices[v.nr].point[1] 
                                    vlen = sqrt((xcoord - Vxcoord)**2 + (ycoord - Vycoord)**2)
                                    self.gfu.vec[v.nr] = 1-vlen/length
                    
             
                        
            # gfu_edge = self.gfu.vec.CreateVector()
            
            # Then extend to subdomains
            
            for bi, bb in enumerate(self.mesh.GetMaterials()):
                if nb_dom.Mask()[bi]: # If the subdomain is on extended edges.. extend in subdomain
                    Vharm, aharm_mat, aharm_inv, E, aharm_edge, aharm_edge_inv = self.vol_extensions[bb] # Extension to subdomain
                    # gfu_extension = GridFunction(Vharm) # Auxiliary function on harmonic space
                    # gfu_extension.vec[:] = 0.0 # Initializing to 0
                    # res = gfu_extension.vec.CreateVector() #
                    # gfu_edge[:]=0.0 # Initializing to 0
                    # gfu_edge.data = self.gfu.vec # Storing the edge extensions
                    # # Vharm.EmbedTranspose(gfu_edge, gfu_extension.vec)

                    # # Extend on subdomain
                    # gfu_extension.vec.data = E * gfu_edge
                    # res.data = aharm_mat * gfu_extension.vec
                    # gfu_extension.vec.data = - aharm_inv * res
                    # # Vharm.Embed(gfu_extension.vec, gfu_edge)
                    # gfu_edge.data = E.T * gfu_extension.vec

                    # self.gfu.vec.data += gfu_edge
                    # gfu_edge.data = -(E.T @ aharm_inv @ aharm_mat @ E) * self.gfuc.vec
                    # print(Norm(gfu_edge.vec))
                    self.gfu.vec.data += -(E.T @ aharm_inv @ aharm_mat @ E) * self.gfu.vec
                    # self.gfuc.vec.data += (aharm_mat @ E) * self.gfuc.vec
            
            
            # if (Norm(self.gfu.vec) > 1):
            #     # self.gfuc.vec.FV()[:] = self.gfu.vec
            #     # basis.Append(self.gfuc.vec)
            # start = time.time()
            # ee += time.time() - start
            basis[iiv] = self.gfu.vec
            
            iiv += 1
                # basis.Append(self.gfu.vec)
        # print("time for copy = ", ee)




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
        if basis == None:
            basis = self.basis_b
        
        iib = 0

        if self.bubble_modes > 0:
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
                    # self.gfu.vec.data = E.T * e.real # Grid funciton on full mesh
                    self.gfuc.vec.FV()[:] = E.T * e.real

                    basis[iib] = self.gfuc.vec
                    iib += 1

                    # basis.Append(self.gfuc.vec)
                    # basis.Append(self.gfu.vec)
            
            

    def calc_basis(self): 
        start_time = time.time()
        self.calc_vertex_basis() 
        vertex_time = time.time() 
        print("Vertex basis functions computation in --- %s seconds ---" % (vertex_time - start_time))
        self.calc_edge_basis()
        edges_time = time.time() 
        print("Edge basis functions computation in --- %s seconds ---" % (edges_time - vertex_time))
        self.calc_bubble_basis()
        bubbles_time = time.time() 
        print("Bubble basis functions computation in --- %s seconds ---" % (bubbles_time - edges_time))
        # quit()
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
        