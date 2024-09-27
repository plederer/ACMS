from ngsolve import *
# from netgen.geom2d import SplineGeometry

from ngsolve.eigenvalues import *
from ngsolve.la import Real2ComplexMatrix
import scipy.linalg
import scipy.sparse as sp
import numpy as np
# from ngsolve.webgui import Draw
# from netgen.webgui import Draw as DrawGeo

import time



###############################################################

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
    def __init__(self, order, mesh, bm = 0, em = 0, mesh_info = None, bi = 0, alpha = 1, kappa = 1, omega = 1, f = 1, g = 1, beta = 1, gamma = 1, save_doms=None, calc_all = False):
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

        self.calc_all = calc_all
        
        self.edge_extensions = {}
        self.vol_extensions = {}

        self.bubble_modes = bm
        self.edge_modes = em
        self.edgeversions = {}

        # self.basis_all = MultiVector(self.gfuc.vec, 0) 
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa
        self.omega = omega
        self.verts = mesh_info["verts"]
        self.edges = mesh_info["edges"]

        self.Nx = 0
        if "Nx" in mesh_info:
            self.Nx = mesh_info["Nx"]
        self.Ny = 0
        if "Ny" in mesh_info:
            self.Ny = mesh_info["Ny"]
        self.Ncell = 0
        if "Ncell" in mesh_info:
            self.Ncell = mesh_info["Ncell"]    
        
        self.is_square_shape_crystal = False
        
        if self.Nx > 0:
            self.is_square_shape_crystal = True
        

        self.doms = list( dict.fromkeys(mesh.GetMaterials()) )
        self.FreeVertices = BitArray(len(mesh.GetBBoundaries()))
        self.FreeEdges = BitArray(len(mesh.GetBoundaries()))

        self.timings = {}
        self.timings["calc_harmonic_ext_assemble_and_inv"] = 0
        self.timings["calc_harmonic_ext_remaining"] = 0
        self.timings["total_calc_harmonic_ext"] = 0
        self.timings["apply_harmonic_ext"] = 0
        self.timings["assemble_vertices"] = 0
        self.timings["assemble_edges"] = 0
        self.timings["assemble_bubbles"] = 0
        self.timings["assemble_basis"] = 0
        self.timings["assemble_extensions"] = 0
        self.timings["total_assemble"] = 0
        


        ss = time.time()
        for v in range(len(mesh.GetBBoundaries())):
            if not "inner_vertex" in mesh.ngmesh.GetCD2Name(v):
                self.FreeVertices[v] = 1
            else:
                self.FreeVertices[v] = 0
        
        for e in range(len(mesh.GetBoundaries())):
            if not "inner_edge" in mesh.ngmesh.GetBCName(e):
                self.FreeEdges[e] = 1
            else:
                self.FreeEdges[e] = 0
        
        self.timings["total_initialize_class"] = time.time() - ss

        self.nverts = len(self.verts)
        self.nedges = len(self.edges)
        self.ncells = len(self.doms)

        self.acmsdofs = self.nverts + self.nedges * self.edge_modes + self.ncells  * self.bubble_modes
        self.ndofemax = 0
        

        #self.basis_v = MultiVector(self.gfuc.vec, self.nverts)
        #self.basis_e = MultiVector(self.gfuc.vec,0) # self.nedges * self.edge_modes)
        #self.basis_b = MultiVector(self.gfuc.vec, self.ncells * self.bubble_modes)
        
        self.asmall = Matrix(self.acmsdofs, self.acmsdofs, complex = True)
        self.asmall[:,:] = 0 + 0*1J
        self.fsmall = Vector(self.acmsdofs, complex = True)
        self.fsmall[:] = 0 + 0*1J
        
        self.ainvsmall = Matrix(self.acmsdofs, self.acmsdofs, complex = True)

        self.localbasis = {}
        # if save_localbasis == None:
        #     self.save_localbasis = self.doms
        # else:
        #     self.save_localbasis = save_localbasis
            
        if save_doms == None:
            self.save_doms = self.doms
        else:
            self.save_doms = save_doms
            

        

        self.bi = bi 

###############################################################
###############################################################

    # Define harmonic extension on specific subdomain
    # Returns the Sobolev space H^1_0(\Omega_j), the stiffness matrix and its inverse
    def GetHarmonicExtensionDomain(self, dom_name):
        # start = time.time()
        
        # fd_all = self.V.GetDofs(self.mesh.Materials(dom_name)) # Dofs of specific domain
        # base_space = H1(self.mesh, order = self.order, complex = True) #, dirichlet = self.dirichlet) #Replicate H^1_0 on subdomain
        # # print(self.dirichlet)
        # Vharm = Compress(base_space, fd_all)
        sss = time.time()
        ss_extension = time.time()
        is_c = True #bool(sum((self.mesh.Materials(dom_name).Neighbours(BND) * self.mesh.Boundaries(self.dom_bnd)).Mask()))
        
        base_space = H1(self.mesh, order = self.order, complex = is_c, definedon = self.mesh.Materials(dom_name))

        Vharm = Compress(base_space)

        # base_space_C = H1(self.mesh, order = self.order, complex = False, definedon = self.mesh.Materials(dom_name))
        # Vharm_C = Compress(base_space_C)
        
        fd = Vharm.FreeDofs()
        edges = self.mesh.Materials(dom_name).Neighbours(BND).Split()
        for bnds in edges[0:4]:
            fd &= ~Vharm.GetDofs(bnds)
        self.timings["calc_harmonic_ext_remaining"] += time.time() - sss

        uharm, vharm = Vharm.TnT() # Trial and test functions
        aharm = BilinearForm(Vharm, symmetric = True)
        #Setting bilinear form: - int (Grad u Grad v) d\Omega_j
        aharm += self.alpha * grad(uharm)*grad(vharm)*dx(definedon = self.mesh.Materials(dom_name), bonus_intorder = self.bi) 
        aharm += -self.gamma * self.kappa**2 * uharm*vharm*dx(definedon = self.mesh.Materials(dom_name), bonus_intorder = self.bi)
        
        
        
        sss = time.time()
        with TaskManager():
            aharm.Assemble()
            aharm_inv = aharm.mat.Inverse(fd, inverse = "sparsecholesky")
        self.timings["calc_harmonic_ext_assemble_and_inv"] += time.time() - sss
        self.timings["total_calc_harmonic_ext"] += time.time() - ss_extension
    
        # aharm_aharm_inv = Matrix(Vharm.ndof,Vharm.ndof)
        # aharminv_aharm = aharm_inv @ aharm.mat
        # aharminv_aharm = ProductMatrix(aharm_inv, aharm.mat)
        # aharminv_aharm = np.matmul(aharm_inv.ToDense(), aharm.mat.ToDense())
        return Vharm, aharm.mat, aharm_inv


###############################################################
###############################################################

    # Define harmonic extension on specific subdomain
    # Returns the Sobolev space H^{1/2}_00(e), the stiffness matrix and its inverse
    def GetHarmonicExtensionEdge(self, edge_name):
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


    ###############################################################
    ###############################################################

    # Function that computes the harmonic extensions on all subdomains and on all edges (of coarse mesh)
    # Returns vol_extensions and edge_extensions
    def CalcHarmonicExtensions(self, kappa = 0, edge_names = None):
        if edge_names == None:
            edge_names = self.mesh.GetBoundaries()
        
        # remove double entries
        # mats = tuple( dict.fromkeys(self.mesh.GetMaterials()) )
        
        # ss = time.time()
        for dom_name in self.doms:
            # Vharm, aharm, aharm_inv, E, aharm_edge, aharm_edge_inv = self.GetHarmonicExtensionDomain(dom_name, kappa = kappa)
            self.vol_extensions[dom_name] = list(self.GetHarmonicExtensionDomain(dom_name))
        # self.timings["total_calc_harmonic_ext"] = time.time() - ss
       
    
    ###############################################################
    ###############################################################

    def Solve(self, condense = False):

        # ss = time.time()

        # usmall = sp.linalg.solve(self.asmall, self.fsmall, assume_a='sym')
       
        # self.ainvsmall = sp.linalg.inv(asparse)
        # # self.ainvsmall = Matrix(np.linalg.inv(self.asmall))
        # self.timings["total_calc_inverse"] = time.time() - ss

        if not condense:
            ss = time.time()
            if False:
                usmall = scipy.linalg.solve(self.asmall, self.fsmall, assume_a='sym')
            else:
                asparse = sp.csc_matrix(self.asmall)
                # self.ainvsmall = sp.linalg.inv(asparse)
                sol = sp.linalg.spsolve(asparse, self.fsmall)
                usmall = Vector(sol)
            self.timings["total_solve"] = time.time() - ss
        else:
            ss = time.time()
            
            nv = self.nverts
            nd = self.acmsdofs

            # A = sp.coo_matrix(self.asmall.NumPy()
            
            A = self.asmall.NumPy()
            f = self.fsmall.NumPy()
            usmall = np.zeros(nd, dtype = 'complex128')
            if False:
                Aee = A[nv:nd, nv:nd]
                Avv = A[0:nv, 0:nv]
                Ave = A[0:nv, nv:nd]
                Aev = A[nv:nd, 0:nv]
                
                fv = f[0:nv]
                fe = f[nv:nd]
                
                uv = usmall[0:nv]
                ue = usmall[nv:nd]
            else:
                Avv = A[nv:nd, nv:nd]
                Aee = np.diag(np.diag(A[0:nv, 0:nv]))
                Aee = A[0:nv, 0:nv]
                Aev = A[0:nv, nv:nd]
                Ave = A[nv:nd, 0:nv]
                
                fe = f[0:nv]
                fv = f[nv:nd]
                
                ue = usmall[0:nv]
                uv = usmall[nv:nd]
                
            self.timings["solve_create_matrices"] = time.time() - ss
            sss = time.time()

            Aeeinv = np.linalg.inv(Aee)
            self.timings["solve_local_inverse"] = time.time() - sss
            sss = time.time()

            S1 = np.matmul(Aeeinv, Aev) #Aee.dot(Aev)
            S2 = np.matmul(Ave,S1) # np.Ave.dot(S1)
            # S = sp.csr_matrix(Avv - S2) #np.matmul(Ave, np.matmul(Aeeinv,Aev))
            S = Avv - S2

            self.timings["solve_create_S"] = time.time() - sss
            
            sss = time.time()
            # Sinv = np.linalg.inv(S)
            
            self.timings["solve_solve_S"] = time.time() - sss
            
            sss = time.time()
            Hext = np.matmul(Ave,Aeeinv) #Ave.dot(Aeeinv)
            HextT = Hext.transpose()
            self.timings["solve_create_Hext"] = time.time() - sss

            sss = time.time()
            fv -= Hext.dot(fe)   # np.dot(np.matmul(Ave,Aeeinv),fe)
            # uv += Sinv.dot(fv)
            uv += scipy.linalg.solve(S, fv, assume_a='sym')
            ue += Aeeinv.dot(fe)
            ue -= HextT.dot(uv) #np.matmul(Aeeinv,Aev).dot(uv)
            self.timings["total_solve_condensed_system"] = time.time() - sss
            self.timings["total_solve"] = time.time() - ss

        return Vector(usmall)
    

    def GetEdges(self, acms_cell):
        nbnd = self.mesh.Materials(acms_cell).Neighbours(BND)
        edge_list = []
        edge_center_of_mass = []

        edges = nbnd.Mask() & self.FreeEdges
        ee = 0
        eemax = sum(edges)
        for i, b in enumerate(edges):
            if ee == eemax:
                break
            if b == 1:
                ee+=1
                bndname = self.mesh.GetBoundaries()[i]
                verts = self.mesh.Boundaries(bndname).Neighbours(BBND).Split()
                Vxcoord = 0.0
                Vycoord = 0.0
                for vv in verts:
                    m = vv.Mask()
                    for ii in range(self.mesh.nv):
                        if m[ii] == 1: # find vertex number
                            
                            Vxcoord += self.mesh.vertices[ii].point[0]
                            Vycoord += self.mesh.vertices[ii].point[1]
                            break
                # np.lexsoort has problems with machine prec zero numbers
                if abs(Vxcoord) < 1e-12:
                    Vxcoord = 0.0
                if abs(Vycoord) < 1e-12:
                    Vycoord = 0.0 
                edge_center_of_mass.append([Vxcoord/2, Vycoord/2])
                edge_list.append([bndname, i])
                

        edge_center_of_mass = np.array(edge_center_of_mass[:], dtype='f')
        p1 = np.array(edge_center_of_mass[:,1] , dtype='f')
        p0 = np.array(edge_center_of_mass[:,0], dtype='f')
        ind = np.lexsort((p1, p0))

        if self.is_square_shape_crystal:
            edge_list = np.array(edge_list)[ind]
        else:
            ind = [i for i in range(len(ind))]

        return edge_list, ind
    
    
    def GetVertices(self, acms_cell):
        nbbnd = self.mesh.Materials(acms_cell).Neighbours(BBND)

        vertices = nbbnd.Mask() & self.FreeVertices

        vert_list = []
        loc_coordinates = []
        for i, b in enumerate(vertices):
            if b == 1:
                vname = self.mesh.GetBBoundaries()[i]
                vert_list.append(vname)
                Vxcoord = self.mesh.vertices[i].point[0]
                Vycoord = self.mesh.vertices[i].point[1]
                loc_coordinates.append([Vxcoord, Vycoord])

        loc_coordinates = np.array(loc_coordinates[:])
        p1 = np.array(loc_coordinates[:,1] , dtype='f')
        p0 = np.array(loc_coordinates[:,0], dtype='f')
        ind = np.lexsort((p1, p0))
        # loc_coordinates = np.array(loc_coordinates)[ind]

        if self.is_square_shape_crystal:
            vert_list = np.array(vert_list)[ind]
        else:
            ind = [i for i in range(len(ind))]
        
        return vert_list, ind
        
    def GetDofs(self, nr):
        Ny = self.Ny
        em = self.edge_modes

        dofs = []

        if self.is_square_shape_crystal:
            # position of vertex basis function in local basis
            vi = []
            # position of first edge basis function per edge
            ei = []
            ii = 0
            
            ##################
            # vertex ordering (0,0), (0,1), (1,0), (1,1)
            # ordering of edges is: left, bottom, top, right
            ##################

            # first vertex dofs (0,0)
            # nr//Ny: counts the additional vertex dof when switching to the next column since there are Ny + 1 vertices
            # nr * em: edge modes per vertival edge
            # (nr//Ny) * (Ny + 1) * em: offset for horizonal em when going to next coumn
            dd = nr + nr//Ny + nr * em + (nr//Ny) * (Ny + 1) * em
            dofs.append(dd)
            vi.append(ii)
            ii+=1

            # first edge (left)
            for l in range(em):
                dofs.append(dd+l+1)
            ei.append(ii)
            ii+=em
            

            # vertex dof (0,1)
            dofs.append(dd + em + 1)
            vi.append(ii)
            ii+=1

            # next edge dofs (bottom)
            # (nr//Ny+1) * (Ny+1): offset vertical vertices 
            # (nr//Ny + 1) * Ny * em: offset vertical edges
            # nr//Ny * (Ny + 1) * em: offset horizontal edges
            # nr%Ny * em: offset in current column of horicontal edges
            dd = (nr//Ny+1) * (Ny+1) + (nr//Ny + 1) * Ny * em + nr//Ny * (Ny + 1) * em + nr%Ny * em
            for l in range(em):
                dofs.append(dd+l)
            ei.append(ii)
            ii+=em
            
            # next edge dofs (top)
            for l in range(em):
                dofs.append(dd + em +l)
            ei.append(ii)
            ii+=em

            # vertex dof (1,0)
            # (nr//Ny+1) * (Ny+1): offset vertical vertices 
            # (nr//Ny + 1) * Ny * em: offset vertical edges
            # (nr//Ny+1) * (Ny+1) * em: offset horicontal edges
            # nr%Ny * (em+1) : offset of dofs of vertical edges + the previous vertex dofs on that vertical line
            dd = (nr//Ny+1) * (Ny+1) + (nr//Ny + 1) * Ny * em + (nr//Ny+1) * (Ny+1) * em + nr%Ny * (em+1)

            vi.append(ii)
            ii+=1 
            dofs.append(dd)

            # last edge (right)
            for l in range(em):
                dofs.append(dd+l+1)
            ei.append(ii)
            ii+=em

            # vertex dof (1,1)
            dofs.append(dd + em + 1)
            vi.append(ii)
            ii+=1
        
        else:
            vd, vi = self.GetVDofs(nr) 
            dofs += vd
            ed, eii = self.GetEDofs(nr) 
            dofs += ed

            # need offset regarding to number of vertices
            ei = [e + len(vi) for e in eii]
            
        return dofs, vi, ei
        
    def GetVDofs(self, nr):
        ## old style dof numbering for vertices
        acms_cell = self.doms[nr]
        nbbnd = self.mesh.Materials(acms_cell).Neighbours(BBND)
        dofs = []
        vertices = nbbnd.Mask() & self.FreeVertices
        vi = []
        ii = 0
        for i, b in enumerate(vertices):
            if b == 1:
                for j in range(self.nverts):
                        if self.verts[j][0] == i:
                            dofs.append(j)
                            vi.append(ii)
                            ii+=1
                            break
                    
        return dofs, vi
    
    def GetEDofs(self, nr):
        ## old style dof numbering for edges
        acms_cell = self.doms[nr]
        nbnd = self.mesh.Materials(acms_cell).Neighbours(BND)

        edges = nbnd.Mask() & self.FreeEdges
        ee = 0
        eemax = sum(edges)
        dofs = []
        ii = 0
        ei = []
        for i, b in enumerate(edges):
            if ee == eemax:
                break
            if b == 1:
                for j in range(self.nedges):
                    if self.edges[j][0] == int(i):
                        for l in range(self.edge_modes):
                            dofs.append(self.nverts + j*self.edge_modes + l)
                        ei.append(ii)
                        ii += self.edge_modes
                        break
                    
        return dofs, ei
    

    def Assemble(self):
        if len(self.localbasis) > 0:
            self.localbasis = {}
        for m in range(len(self.doms)):
            self.Assemble_localA_and_f(m)
    
    def Assemble_localA_and_f(self, acms_cell_nr):
        acms_cell = self.doms[acms_cell_nr]
        Vharm, aharm_mat, aharm_inv = self.GetHarmonicExtensionDomain(acms_cell)
        if acms_cell in self.save_doms: 
           self.vol_extensions[acms_cell] = [Vharm, aharm_mat, aharm_inv]
        ss_assemble = time.time()

        nbnd = self.mesh.Materials(acms_cell).Neighbours(BND)
        nbbnd = self.mesh.Materials(acms_cell).Neighbours(BBND)

        ### due to inner vertices
        offset = sum(nbbnd.Mask())
        
        vertices = nbbnd.Mask() & self.FreeVertices
        edges = nbnd.Mask() & self.FreeEdges
        
        local_vertex_dofs = Vharm.GetDofs(nbbnd)
        
        gfu = GridFunction(Vharm)

        localbasis = MultiVector(gfu.vec, sum(vertices) + sum(edges) * self.edge_modes + self.bubble_modes)
        localbasis_edges = MultiVector(gfu.vec, self.edge_modes)

        dofs = []
        lii = 0
        
        vii = 0
        ss_extension = time.time()
        ttt = time.time()
        sss = time.time()

        # print(acms_cell)
        # print(self.GetVertices(acms_cell))

        # for i, b in enumerate(vertices):
        # dofs += self.GetVDofs(acms_cell_nr)
        vert_list, ind  = self.GetVertices(acms_cell)
        
        dofs, vbi, ebi = self.GetDofs(acms_cell_nr)
        # print(dofs)
        # print(vbi)
        # input(ebi)
        for vname in vert_list:
            
            # if b == 1:
            if True:
                
                # derive acms dof numbering 
                # for j in range(self.nverts):
                #     if self.verts[j][0] == i:
                #         dofs.append(j)
                #         break

                
                # add corresponding vertex basis function
                #vertex name
                # vname = self.mesh.GetBBoundaries()[i]
                
                vnbnd = nbnd * self.mesh.BBoundaries(vname).Neighbours(BND)
                
                if True:
                    # if i < len(gfu.vec):
                    
                    
                    gfu.vec[ind[vii]] = 1 #set active vertex dof to 1
                    vii += 1
                    # orient = 0
                    
                    for bnds in vnbnd.Split():
                        els = [e for e in bnds.Elements()]
                        nels = len(els)
                        # orient = sum(Integrate(specialcf.tangential(2), self.mesh, definedon=bnds, order = 0))
                        # print(orient)
                        vals = [i/(nels) for i in range(1,nels) ]
                        
                        # if els[0].vertices[0].nr == i: # or els[0].vertices[1].nr == 1:
                            # vals.reverse()

                        #vname = Vxx -> vname[1:] = xx
                        if els[0].vertices[0].nr == int(vname[1:]): # or 
                            vals.reverse()
                        # print(Vharm.GetDofs(bnds))

                        # offset = 5 # 4 vertices + 1 vertex in the middle that was created for the circle domain
                        bdofs = Vharm.GetDofs(bnds)
                        
                  
                        
                        ii = offset
                        while bdofs[ii] == 0:
                            ii +=1
                        
                        dd = nels-1 #just linear dofs
                        for iii in range(0,dd):
                            gfu.vec[ii] = vals[iii]
                            ii+=1
                    
                else:   
                    # print(vv)         
                    # Vxcoord = self.mesh.vertices[int(vv)].point[0]
                    # Vycoord = self.mesh.vertices[int(vv)].point[1]

                    for bnds in vnbnd.Split():
                        # slength = Integrate(1, self.mesh, definedon=bnds, order = 0)
                        slength = 0

                        Xcoords = []
                        Ycoords = []
                        for e in bnds.Elements():
                            
                            
                            for vi, v in enumerate(e.vertices):
                                xcoord = self.mesh.vertices[v.nr].point[0]
                                Xcoords.append(xcoord)
                                ycoord = self.mesh.vertices[v.nr].point[1] 
                                Ycoords.append(ycoord)
                            
                            el_len = sqrt((Xcoords[0] - Xcoords[1])**2 + (Ycoords[0] - Ycoords[1])**2)
                            slength += el_len
                            # vlen = sqrt((xcoord - Vxcoord)**2 + (ycoord - Vycoord)**2)
                        for e in bnds.Elements():
                            edofs = Vharm.GetDofNrs(e)
                            for vi, v in enumerate(e.vertices):
                                gfu.vec[edofs[vi]] = 1-vlen/slength
                            
                # sss = time.time()
                # aharminv_aharm = ProductMatrix(aharm_inv, aharm.mat)
                # aharminv_aharm = Matrix(Vharm.ndof, complex = True)
                # aharminv_aharm = (aharm_inv @ aharm_mat).ToDense()

                with TaskManager():
                    gfu.vec.data += -(aharm_inv @ aharm_mat) * gfu.vec  
                
                # gfu.vec.data += -(aharm_mat) * gfu.vec  
                # gfu.vec[:] = 1
                # Draw(gfu, self.mesh, "test")
                # input()
                localbasis[vbi[lii]][:] = gfu.vec
                lii +=1
                gfu.vec[:] = 0
        self.timings["assemble_vertices"] += time.time()-sss    

        sss = time.time()
        local_dom_bnd = ""
        edge_list, ind = self.GetEdges(acms_cell)
        ee = 0
        eemax = sum(edges)
        # for i, b in enumerate(edges):
        # dofs += self.GetEDofs(acms_cell_nr)
        # input()
        lii = 0
        for bndname, i in edge_list:
            # if ee == eemax:
            #     break
            # if b == 1:
            #     ee+=1
            if True:
                # for j in range(self.nedges):
                #     if self.edges[j][0] == int(i):
                #         for l in range(self.edge_modes):
                #             dofs.append(self.nverts + j*self.edge_modes + l)
                #         break
                       
                # bndname = self.mesh.GetBoundaries()[i]
                if bndname in self.dom_bnd:
                    local_dom_bnd += bndname + "|"

                ddofs = Vharm.GetDofs(self.mesh.Boundaries(bndname)) & (~local_vertex_dofs)
                
                edgetype = ""
                if "V" in bndname:
                    edgetype = "V"
                elif "H" in bndname:
                    edgetype = "H"
                elif "D" in bndname:
                    edgetype = "D"
                elif "C" in bndname:
                    edgetype = "C"
                    

                els = [e for e in self.mesh.Boundaries(bndname).Elements()]
                efirst = els[0].elementnode.nr
                elast = els[-1].elementnode.nr

                switch = False
                if self.is_square_shape_crystal:
                    if elast < efirst and "H" in bndname:
                        switch = True
                    if elast > efirst and "V" in bndname:
                        switch = True
                else:
                    # if elast < efirst and "H" in bndname:
                    #     switch = True
                    if elast < efirst and "V" in bndname:
                        switch = True
                    if elast < efirst and "D" in bndname:
                        switch = True
                    if elast > efirst and "C" in bndname:
                        switch = True
                # orient = []
                # for e in els:
                    
                #     v1 = e.vertices[0].nr
                #     v2 = e.vertices[1].nr
                #     if v1 < v2:
                #         orient.append(1)
                #     else:
                #         orient.append(-1)
                #     if "H" in bndname:
                #         print("e.nr = ", e.elementnode.nr)
                #         # print(self.mesh.vertices[e.vertices[0].nr].point[0])
                #         # print(self.mesh.vertices[e.vertices[1].nr].point[0])

                nels = len(els)   
                edgetype += "_" + str(nels)
                
                # if "C" in bndname or "D" in bndname:
                # print(bndname)
                if self.calc_all:
                    edgetype = bndname #edge_name
                
                dde = nels * (self.order -1) #inner dofs on edges
                ddn = nels -1 #vertex dofs
                dd = dde + ddn #nels * (self.order -1) + nels -1
                sss = time.time()
                
                # for l in range(self.edge_modes):
                #     ii = 0
                #     for d, bb in enumerate(ddofs):
                #         if bb == 1:
                #             gfu.vec[d] = self.edgeversions[edgetype][0][l][ii]#.real
                #             # gfu.vec[d] = self.edgeversions[edgetype][1][l,ii]#.real
                #             ii+=1
                #         if ii == dd-1:
                #             break
                    
                #     gfu.vec.data += -(aharm_inv @ aharm_mat) * gfu.vec
                #     localbasis[lii][:] = gfu.vec
                #     lii+=1
                #     gfu.vec[:] = 0
                # if "dom_bnd_top_H" in bndname:
                #     print(bndname)
                #     print(self.edgeversions[edgetype][0][0][1])
                #     print(np.linalg.norm(self.edgeversions[edgetype][0][0][1]))
                #     # orient = sum(Integrate(specialcf.tangential(2), self.mesh, definedon=self.mesh.Boundaries(bndname), order = 0))
                #     print("switch:", switch)
                    # print("orientation = ", orient)
                    # print("original orientation = ", self.edgeversions[edgetype][1])
                    
                    # print("dd = ", dd)
                gfu.vec[:] = 0
                # Draw(gfu, self.mesh, "test")
                ind = list(range(0,dd))
                if switch and not self.calc_all:
                    inner_dofs = self.order - 1
                    for l in range(inner_dofs):
                        tmp = ind[ddn + l]
                        ind[ddn + l] = ind[ddn + l + inner_dofs]
                        ind[ddn + l + inner_dofs] = tmp
                
                for l in range(self.edge_modes):
                    ii = 0
                    for d, bb in enumerate(ddofs):
                        if bb == 1:
                            gfu.vec[d] = self.edgeversions[edgetype][0][l][ind[ii]]#.real
                            # gfu.vec[d] = self.edgeversions[edgetype][1][l,ii]#.real
                            ii+=1
                        if ii == dd:
                            break
                    # print("ii = ", ii)
                    # gfu.vec.data += -(aharm_inv @ aharm_mat) * gfu.vec
                    localbasis_edges[l][:] = gfu.vec
                    gfu.vec[:] = 0
                with TaskManager():
                    localbasis_edges[:] += -(aharm_inv @ aharm_mat) * localbasis_edges
                    
                
                # for l in range(self.edge_modes):
                #     gfu.vec.data = localbasis_edges[l]
                #     Draw(gfu, self.mesh,"test" )
                #     input()
                # print(lii)
                # print(ebi[lii])

                localbasis[ebi[lii]:ebi[lii]+self.edge_modes] = localbasis_edges

                # lii += self.edge_modes
                lii +=1
                self.timings["assemble_edges"] += time.time()-sss
        # print(dofs)
        sss = time.time()
        if self.bubble_modes > 0:
            voli = int(np.nonzero(self.mesh.Materials(acms_cell).Mask())[0][0])
            # print(voli)
            for l in range(self.bubble_modes):
                dofs.append(self.nverts + self.nedges*self.edge_modes + voli * self.bubble_modes + l)
                
            uloc, vloc = Vharm.TnT()
            aloc = BilinearForm(Vharm)
            aloc += self.alpha * grad(uloc) * grad(vloc) * dx(bonus_intorder = self.bi)
            aloc.Assemble()

            #Setting bilinear form: int  u v d\Omega_j
            mloc = BilinearForm(Vharm)
            mloc += uloc * vloc * dx(bonus_intorder = self.bi)
            mloc.Assemble()

            # minv = mloc.mat.Inverse(Vharm.FreeDofs()) #, inverse = "sparsecholesky")

            # Solving eigenvalue problem: AA x = ev MM x
            
            ddd = []
            
            for i in range(Vharm.ndof):
                if Vharm.FreeDofs()[i]:
                    ddd.append(i)   
            # AA = sp.csr_matrix(aloc.mat.CSR())
            # MM = sp.csr_matrix(mloc.mat.CSR())
            AA = aloc.mat.ToDense().NumPy()
            MM = mloc.mat.ToDense().NumPy()
            
            MI = np.zeros((Vharm.ndof,Vharm.ndof), dtype=complex)
            submat = MM[np.ix_(ddd,ddd)]
            # print(submat)
            submat_inv = np.linalg.inv(submat)
            MI[np.ix_(ddd,ddd)] = submat_inv
            

                            
            ev, evec =scipy.sparse.linalg.eigs(A = AA, M = MM, Minv = MI, k = self.bubble_modes, which='SM')
            idx = ev.argsort()[::]   
            ev = ev[idx]
            evec = evec[:,idx]
            evec = evec.transpose()

            for e in evec:
                gfu.vec[:]=0.0
                gfu.vec[:] = e
                localbasis[lii][:] = gfu.vec
                lii+=1
        
        self.timings["assemble_bubbles"] += time.time() - sss
        self.timings["assemble_basis"] += time.time() - ttt

        # print(dofs)
        # input()
        if acms_cell in self.save_doms:
           self.localbasis[acms_cell] = (localbasis, dofs)
        
        uharm, vharm = Vharm.TnT() 
        
        ttt = time.time()
        sss = time.time()
        if local_dom_bnd != "":
            
            local_a = BilinearForm(Vharm, symmetric = True, check_unused=False)
            
            beta = self.beta # ATTENTION: This should be given in input
            

            local_dom_bnd = local_dom_bnd[:-1]
            
            local_a += -1J * self.omega * beta * uharm * vharm * ds(local_dom_bnd, bonus_intorder = self.bi)
            
            with TaskManager():
                local_a.Assemble()
                local_a.mat.AsVector().data += aharm_mat.AsVector()
                localmat = InnerProduct(localbasis, (local_a.mat * localbasis).Evaluate(), conjugate = False)
        else:
            with TaskManager():
                localmat = InnerProduct(localbasis, (aharm_mat * localbasis).Evaluate(), conjugate = False)
        self.timings["assemble_extensions"] += time.time() - ttt

        
        local_f = LinearForm(Vharm)
        local_f += self.f * vharm * dx(definedon = self.mesh.Materials(acms_cell))
        local_f += self.g * vharm * ds(local_dom_bnd)
        with TaskManager():
            local_f.Assemble()
        
        ttt = time.time()
        localvec = InnerProduct(localbasis, local_f.vec, conjugate = False)
        self.timings["assemble_extensions"] += time.time() - ttt
        for i in range(len(dofs)):
            for j in range(len(dofs)): 
                self.asmall[dofs[i],dofs[j]] += localmat[i,j]
            
            self.fsmall[dofs[i]] += localvec[i]
        self.timings["total_assemble"] += time.time() - ss_assemble
                
###############################################################
###############################################################
    def IntegrateACMS(self, bndname, coeffs):

        # print(self.mesh.GetBoundaries())
        # print(self.mesh.GetBoundaries())

        integral = 0
        for edgename in self.mesh.GetBoundaries():
            if bndname in edgename:
                # print(edgename)
                cells = self.mesh.Boundaries(edgename).Neighbours(VOL).Split()[0].Mask()

                
                # print(cells)
                for i,b in enumerate(cells):
                    if b == 1:
                        cellname = self.mesh.GetMaterials()[i]
                        # print(cellname)
                        localbasis, dofs = self.localbasis[cellname]

                        Vharm, aharm_mat, aharm_inv = self.vol_extensions[cellname]
                        gfu = GridFunction(Vharm)
                        
                        localcoeffs = Vector(len(dofs), complex = True)
                        
                        for d in range(len(dofs)):
                            localcoeffs[d] = coeffs[dofs[d]]
                    
                        gfu.vec.data = (localbasis * localcoeffs).Evaluate()
                        Draw(gfu, self.mesh, "cell")
                        
                        # rr = sqrt(gfu.real**2 + gfu.imag**2)
                        rr = gfu.Norm()
                        integral+= Integrate(rr, self.mesh, definedon = self.mesh.Boundaries(edgename))
                        # print(integral)
        
        return integral
    
###############################################################
###############################################################

        
    def SetGlobalFunction(self, gfu, coeffs, doms = None):
        ss = time.time()
        self.timings["set_setup"] = 0.0
        self.timings["set_calc_localbasis"] = 0.0
        self.timings["set_average"] = 0.0
        self.timings["set_local_to_global"] = 0.0
        
        if doms == None:
            doms = self.doms
        
        for acms_cell in doms:
            Vharm, aharm_mat, aharm_inv = self.vol_extensions[acms_cell]
            
            sss = time.time()

            dofs = self.localbasis[acms_cell][1]
            localcoeffs = Vector(len(dofs),complex = True)

            for d in range(len(dofs)):
                localcoeffs[d] = coeffs[dofs[d]]
            self.timings["set_setup"] += time.time() - sss

            sss = time.time()
            localvec = (self.localbasis[acms_cell][0] * localcoeffs).Evaluate()
            self.timings["set_calc_localbasis"] += time.time() - sss
            
            sss = time.time()
            gfu_local = GridFunction(Vharm)
            gfu_local.vec.data = localvec
            gfu_test = GridFunction(self.Vc)
            gfu_test.Set(gfu_local, definedon = self.mesh.Materials(acms_cell)) #, dual = True)
            gfu.vec.data += gfu_test.vec
            
            # fd_all = self.Vc.GetDofs(self.mesh.Materials(acms_cell))
           
            # ii = 0
            # for i, b in enumerate(fd_all):
            #     if b == True:
            #         gfu.vec.FV()[i] += localvec[ii]
            #         ii+=1
            self.timings["set_local_to_global"] += time.time() - sss
        
        #####
        sss = time.time()
        Vcc = self.Vc
        gfu_average = gfu.vec.CreateVector() 
        gfu_average[:]=1

        bnds = ""
        for ee in self.edges:
            bnds += ee[1] + "|"
        bnds = bnds[:-1]
        
        edofs = Vcc.GetDofs(self.mesh.Boundaries(bnds) - self.mesh.Boundaries(self.dom_bnd))
        
        for ee in range(Vcc.ndof):
            if edofs[ee] == 1:
                gfu_average[ee] = 1/2
        

        for vi, vs in enumerate(self.mesh.GetBBoundaries()):
            neig = len(self.mesh.BBoundaries(vs).Neighbours(VOL).Split())
            if not "inner_vertex" in vs and neig > 2:
                gfu_average[vi] = 1/neig        

        for i in range(Vcc.ndof):
            gfu.vec[i] *= gfu_average[i]
        self.timings["set_average"] += time.time() - sss
        ######

        self.timings["total_set_global_functions"] = time.time() - ss
    

   
    ###############################################################
    ###############################################################
    ###############################################################
    # EDGE MODES

    def CalcMaxEdgeModes(self):
        ee = 0
        
        for edge_name in self.edges:
            ss = time.time()
            dirverts = ""
            for v in self.verts:
                dirverts += v[1] + "|"

            dirverts = dirverts[:-1]
            vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(dirverts)) # Global vertices (coarse mesh)

            fd = self.V.GetDofs(self.mesh.Boundaries(edge_name[1])) & (~vertex_dofs) & (~self.V.FreeDofs())
            
            ee += time.time() - ss
     
     
    ###############################################################
    ###############################################################
       
    def calc_edge_basis(self, basis = None):
        #if (basis == None):
        #    basis = self.basis_e
        
        self.timings["calc_edgebasis_eigenvalues"] = 0
        self.timings["calc_edgebasis_assemble_and_inv"] = 0
        self.timings["calc_edgebasis_remaining"] = 0

        ss = time.time()
        vertex_dofs = self.V.GetDofs(self.mesh.BBoundaries(".*")) # Global vertices (coarse mesh)
        if self.edge_modes > 0:
            for edge_name in self.edges:
                edgetype = ""
                # edgestart = time.time()
                if "V" in edge_name[1]:
                    edgetype = "V"
                elif "H" in edge_name[1]:
                    edgetype = "H"
                elif "D" in edge_name[1]:
                    edgetype = "D"
                elif "C" in edge_name[1]:
                    edgetype = "C"
                else:
                    print("edge_name = ", edge_name)
                    raise Exception("wrong edge type")

                
                # ndofs = sum(list(fd))

                ssss = time.time()
                els = [e for e in self.mesh.Boundaries(edge_name[1]).Elements()]
                nels = len(els)
                # print(ndofs)
                # ndofs = sum(H1(self.mesh, order = self.order, definedon = self.mesh.Boundaries(edge_name[1])).FreeDofs()) - 2
                # print(tt - 2)
                
                self.timings["calc_edgebasis_remaining"] += time.time() - ssss
                # print(vertex_dofs)
                # print(fd)
                # quit()
                edgetype += "_" + str(nels)

                # if "C" in edge_name[1] or "D" in edge_name[1]:
                if self.calc_all:
                    edgetype = edge_name[1]
                

                if edgetype not in self.edgeversions:    
                    fd = self.V.GetDofs(self.mesh.Boundaries(edge_name[1])) & (~vertex_dofs)                           
                    # print("AAA")
                    base_space = H1(self.mesh, order = self.order)#, dirichlet = self.dirichlet) # Creating Sobolev space
                    Vloc = Compress(base_space, fd) #Restricting Sobolev space on edge (with Dirichlet bc)

                    if Vloc.ndof >= self.ndofemax:
                        self.ndofemax = Vloc.ndof

                    uloc, vloc = Vloc.TnT() # Trial and test functions
                    t = specialcf.tangential(2)
                    
                    #Setting bilinear form:  int (Grad u Grad v) de
                    aloc = BilinearForm(Vloc, symmetric = True)
                    # This allows us to take the normal derivative of a function that is in H1 and computing the integral only on edges
                    # Otherwise NGSolve does not allow to take the trace of a function in H^{1/2}(e) - uloc is defined on edge
                    aloc += (grad(uloc)*t) * (grad(vloc)*t) * ds(skeleton=True, definedon = self.mesh.Boundaries(edge_name[1]), bonus_intorder = self.bi)

                    # print(Vloc.FreeDofs())
                    # gfu = GridFunction(Vloc)
                    # Draw(gfu, self.mesh, "test")
                    # for i in range(Vloc.ndof):
                    #     gfu.vec[:] = 0.0
                    #     gfu.vec[i] = 1
                    #     Redraw()
                    #     input()
                    
                    #Setting bilinear form:  int u v de        
                    mloc = BilinearForm(Vloc, symmetric = True)
                    mloc += uloc.Trace() * vloc.Trace() * ds(skeleton = True, definedon = self.mesh.Boundaries(edge_name[1]), bonus_intorder = self.bi)

                    sss= time.time()
                    aloc.Assemble()
                    mloc.Assemble()
                    
                    self.timings["calc_edgebasis_assemble_and_inv"] += time.time() - sss
                    
                    # Solving eigenvalue problem: AA x = ev MM x
                    
                    # print("edge type = ", edgetype)
            
            
                    sss = time.time()
                    try:
                        # print(self.edge_modes)
                        # print(sum(fd))
                        if False:
                            minv = aloc.mat.Inverse(Vloc.FreeDofs(), inverse = "sparsecholesky") #IdentityMatrix(Vloc.ndof)        
                            lams, uvecs = PINVIT(aloc.mat, mloc.mat, pre = minv, num = self.edge_modes, printrates = False, maxit = 20)
                        else:
                            AA = sp.csr_matrix(aloc.mat.CSR())
                            MM = sp.csr_matrix(mloc.mat.CSR())
                            lams, uvecs =sp.linalg.eigs(A = AA, M = MM, k = self.edge_modes, which='SM', tol = 1e-8)
                            idx = lams.argsort()[::]   
                            lams = lams[idx]
                            uvecs = uvecs[:,idx]
                            # for i in range(self.edge_modes):
                            #     if uvecs[:,i][0].real < 0 and sqrt(uvecs[:,i][0].real**2) > 1e-10:
                            #         uvecs[:,i] = -1 * uvecs[:,i]
                                # else...
                                # if uvecs[:,i][0] uvecs[:,i][1] < 0:
                                #     uvecs[:,i] = -1 * uvecs[:,i]
                            uvecs = uvecs.transpose()
                        self.edgeversions[edgetype] = [uvecs]
                        self.timings["calc_edgebasis_eigenvalues"] += time.time() - sss
                    except:
                        self.edge_modes = 0
                        break
                    # print("self.edge_modes = ", self.edge_modes)
                    # print("self.bubble_modes = ", self.bubble_modes)
        
        self.timings["total_calc_edgebasis"] = time.time() - ss
        return (self.edge_modes != 0) #or (self.bubble_modes != 0)
    
                


   
    ###############################################################
    ###############################################################
    ###############################################################
    # VERTEX BASIS

    def calc_vertex_basis(self, basis = None):
        if basis == None:
            basis = self.basis_v

        iiv = 0
        for j, vertex_name in enumerate(self.verts):
            
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
                    
             
            # Then extend to subdomains
            for bi, bb in enumerate(self.mesh.GetMaterials()):
                if nb_dom.Mask()[bi]: # If the subdomain is on extended edges.. extend in subdomain
                    Vharm, aharm_mat, aharm_inv, E, aharm_edge, aharm_edge_inv = self.vol_extensions[bb] # Extension to subdomain
                    self.gfu.vec.data += -(E.T @ aharm_inv @ aharm_mat @ E) * self.gfu.vec
            
            basis[iiv] = self.gfu.vec
            
            iiv += 1
         
         
         
             
             
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

    ###############################################################
    ###############################################################


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
                    self.gfuc.vec.FV()[:] = E.T * e.real

                    basis[iib] = self.gfuc.vec
                    iib += 1

    def PrintTiminigs(self, all = False):
        print( 60 * "#")
        total = 0
        if all == True:
            for key in self.timings.keys():
                # print("time for " + key + ": " + str(self.timings[key]))
                print(f"{'time for ' + key + ': ':<45}" +  str(self.timings[key]))
            print( 60 * "#")
        for key in self.timings.keys():
            if "total" in key:
                print(f"{'time for ' + key + ': ':<35}" +  str(self.timings[key]))
                total += self.timings[key]
        print("Total time: ", total)
        print( 60 * "#")
            
            
###############################################################
###############################################################


    # def calc_basis(self): 
    #     start_time = time.time()
    #     self.calc_vertex_basis() 
    #     vertex_time = time.time() 
    #     print("Vertex basis functions computation in --- %s seconds ---" % (vertex_time - start_time))
    #     self.calc_edge_basis()
    #     edges_time = time.time() 
    #     print("Edge basis functions computation in --- %s seconds ---" % (edges_time - vertex_time))
    #     self.calc_bubble_basis()
    #     bubbles_time = time.time() 
    #     print("Bubble basis functions computation in --- %s seconds ---" % (bubbles_time - edges_time))
    #     # quit()
    #     return self.basis_v, self.basis_e, self.basis_b
    
    
  ###############################################################
###############################################################

  
    # def complex_basis(self):
    #     Vc = H1(self. mesh, order = self.order, complex = True)
    #     gfu = GridFunction(Vc)
    #     basis = MultiVector(gfu.vec, 0)

    #     for bv in self.basis_v:
    #         gfu.vec.FV()[:] = bv
    #         basis.Append(gfu.vec)

    #     for be in self.basis_e:
    #         gfu.vec.FV()[:] = be
    #         basis.Append(gfu.vec)
                
    #     for bb in self.basis_b:
    #         gfu.vec.FV()[:] = bb
    #         basis.Append(gfu.vec)
        
    #     return basis
        
