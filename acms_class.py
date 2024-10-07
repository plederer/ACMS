from ngsolve import *
import scipy.linalg
import scipy.sparse as sp
import numpy as np

from ngsolve.eigenvalues import PINVIT

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
                
        self.asmall = Matrix(self.acmsdofs, self.acmsdofs, complex = True)
        self.asmall[:,:] = 0 + 0*1J
        self.fsmall = Vector(self.acmsdofs, complex = True)
        self.fsmall[:] = 0 + 0*1J
        
        self.ainvsmall = Matrix(self.acmsdofs, self.acmsdofs, complex = True)

        self.localbasis = {}
            
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
        sss = time.time()
        ss_extension = time.time()
        base_space = H1(self.mesh, order = self.order, complex = True, definedon = self.mesh.Materials(dom_name))

        Vharm = Compress(base_space)
        
        fd = Vharm.FreeDofs()
        edges = self.mesh.Materials(dom_name).Neighbours(BND).Split()
        for bnds in edges[0:4]:
            fd &= ~Vharm.GetDofs(bnds)
        self.timings["calc_harmonic_ext_remaining"] += time.time() - sss

        uharm, vharm = Vharm.TnT() 
        aharm = BilinearForm(Vharm, symmetric = True)
        aharm += self.alpha * grad(uharm)*grad(vharm)*dx(definedon = self.mesh.Materials(dom_name), bonus_intorder = self.bi) 
        aharm += -self.gamma * self.kappa**2 * uharm*vharm*dx(definedon = self.mesh.Materials(dom_name), bonus_intorder = self.bi)
        
        sss = time.time()
        with TaskManager():
            aharm.Assemble()
            aharm_inv = aharm.mat.Inverse(fd, inverse = "sparsecholesky")
        self.timings["calc_harmonic_ext_assemble_and_inv"] += time.time() - sss
        self.timings["total_calc_harmonic_ext"] += time.time() - ss_extension
    
        return Vharm, aharm.mat, aharm_inv, fd


###############################################################
###############################################################

    def GetHarmonicExtensionEdge(self, edge_name):
        fd_all = self.V.GetDofs(self.mesh.Boundaries(edge_name)) 
        bnd = "" 
        for b in self.mesh.GetBoundaries():
            if (b != edge_name): # If the edge is not our specified edge, then add it to bnd 
                bnd += b + "|"
        bnd = bnd[:-1] # Remove the last added "|" - unnecessary
        base_space = H1(self.mesh, order = self.order, dirichlet = bnd)

        
        Vharm = Compress(base_space, fd_all) 
        t = specialcf.tangential(2)
        uharm, vharm = Vharm.TnT()
        aharm = BilinearForm(Vharm)
        aharm += (grad(uharm)*t) * (grad(vharm)*t) * ds(skeleton = True, definedon=self.mesh.Boundaries(edge_name), bonus_intorder = self.bi)
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


    ###############################################################
    ###############################################################

    # Function that computes the harmonic extensions on all subdomains and on all edges (of coarse mesh)
    # Returns vol_extensions and edge_extensions
    def CalcHarmonicExtensions(self, kappa = 0, edge_names = None):
        if edge_names == None:
            edge_names = self.mesh.GetBoundaries()
        
        for dom_name in self.doms:
            self.vol_extensions[dom_name] = list(self.GetHarmonicExtensionDomain(dom_name))     
    
    ###############################################################
    ###############################################################

    def Solve(self, condense = False):

        if not condense:
            ss = time.time()
            asparse = sp.csc_matrix(self.asmall)
            sol = sp.linalg.spsolve(asparse, self.fsmall)
            usmall = Vector(sol)
            self.timings["total_solve"] = time.time() - ss
        else:
            ss = time.time()
            
            nv = self.nverts + self.nedges * self.edge_modes 
            nd = self.acmsdofs
            
            A = self.asmall.NumPy()
            f = self.fsmall.NumPy()
            usmall = np.zeros(nd, dtype = 'complex128')
            
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
            S = Avv - S2
            
            self.timings["solve_create_S"] = time.time() - sss
            
            sss = time.time()
            Hext = np.matmul(Ave,Aeeinv) 
            HextT = Hext.transpose()
            self.timings["solve_create_Hext"] = time.time() - sss

            sss = time.time()
            fv -= Hext.dot(fe)  
            
            sss = time.time()
            asparse = sp.csc_matrix(S)
            sol = sp.linalg.spsolve(asparse, fv)
            self.timings["solve_solve_S"] = time.time() - sss

            uv += sol
            ue += Aeeinv.dot(fe)
            ue -= HextT.dot(uv)
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
        bm = self.bubble_modes

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
            
        bbi = self.nverts + self.nedges * self.edge_modes
        for i in range(bm):
            dofs.append(bbi + i + nr*bm)

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
        Vharm, aharm_mat, aharm_inv, fd = self.GetHarmonicExtensionDomain(acms_cell)
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

        vert_list, ind  = self.GetVertices(acms_cell)
        
        dofs, vbi, ebi = self.GetDofs(acms_cell_nr)
       
        for vname in vert_list:
            vnbnd = nbnd * self.mesh.BBoundaries(vname).Neighbours(BND)
            if True:
                gfu.vec[ind[vii]] = 1 #set active vertex dof to 1
                vii += 1
                # orient = 0
                
                for bnds in vnbnd.Split():
                    els = [e for e in bnds.Elements()]
                    nels = len(els)
                    vals = [i/(nels) for i in range(1,nels) ]
                    
                    #vname = Vxx -> vname[1:] = xx
                    if els[0].vertices[0].nr == int(vname[1:]): # or 
                        vals.reverse()
                    
                    bdofs = Vharm.GetDofs(bnds)
        
                    ii = offset
                    while bdofs[ii] == 0:
                        ii +=1
                    
                    dd = nels-1 #just linear dofs
                    for iii in range(0,dd):
                        gfu.vec[ii] = vals[iii]
                        ii+=1
                
            else: 
                # old version  
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
                        
            with TaskManager():
                gfu.vec.data += -(aharm_inv @ aharm_mat) * gfu.vec  

            localbasis[vbi[lii]][:] = gfu.vec
            lii +=1
            gfu.vec[:] = 0
        self.timings["assemble_vertices"] += time.time()-sss    

        sss = time.time()
        local_dom_bnd = ""
        edge_list, ind = self.GetEdges(acms_cell)
        ee = 0
        eemax = sum(edges)
       
        lii = 0
        for bndname, i in edge_list:
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
                if elast < efirst and "V" in bndname:
                    switch = True
                if elast < efirst and "D" in bndname:
                    switch = True
                if elast > efirst and "C" in bndname:
                    switch = True

            nels = len(els)   
            edgetype += "_" + str(nels)
            
            if self.calc_all:
                edgetype = bndname 
            
            dde = nels * (self.order -1) #inner dofs on edges
            ddn = nels -1 #vertex dofs
            dd = dde + ddn 
            sss = time.time()
            
            gfu.vec[:] = 0
            
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
                        ii+=1
                    if ii == dd:
                        break
                localbasis_edges[l][:] = gfu.vec
                gfu.vec[:] = 0
            with TaskManager():
                localbasis_edges[:] += -(aharm_inv @ aharm_mat) * localbasis_edges
                
            localbasis[ebi[lii]:ebi[lii]+self.edge_modes] = localbasis_edges

            lii +=1
            self.timings["assemble_edges"] += time.time()-sss
       
        sss = time.time()
        lii = 0
        if self.bubble_modes > 0:
            uloc, vloc = Vharm.TnT()
            aloc = BilinearForm(Vharm)
            aloc += self.alpha * grad(uloc) * grad(vloc) * dx(bonus_intorder = self.bi)
            aloc.Assemble()

            mloc = BilinearForm(Vharm)
            mloc += uloc * vloc * dx(bonus_intorder = self.bi)
            mloc.Assemble()

            # AA = aloc.mat.ToDense().NumPy()
            # MM = mloc.mat.ToDense().NumPy()
             
            # MI = np.zeros((Vharm.ndof,Vharm.ndof), dtype=complex)
            # submat = MM[np.ix_(fd,fd)]
            # submat_inv = np.linalg.inv(submat)
            # MI[np.ix_(fd,fd)] = submat_inv
            
            # ev, evec =scipy.sparse.linalg.eigs(A = AA, M = MM, Minv = MI, k = self.bubble_modes, which="SM")
            # idx = ev.argsort()[::]   
            # ev = ev[idx]
            # evec = evec[:,idx]
            # evec = evec.transpose()

            minv = aloc.mat.Inverse(fd, inverse = "sparsecholesky")       
            ev, evec = PINVIT(aloc.mat, mloc.mat, pre = minv, num = self.bubble_modes, printrates = False, maxit = 20)
            
            bbi = len(vert_list) + self.edge_modes* len(edge_list)

            for e in evec:
                gfu.vec[:]=0.0
                gfu.vec[:] = e
                localbasis[bbi + lii][:] = gfu.vec
                lii+=1
            

        self.timings["assemble_bubbles"] += time.time() - sss
        self.timings["assemble_basis"] += time.time() - ttt

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
        integral = 0
        for edgename in self.mesh.GetBoundaries():
            if bndname in edgename:

                cells = self.mesh.Boundaries(edgename).Neighbours(VOL).Split()[0].Mask()

                for i,b in enumerate(cells):
                    if b == 1:
                        cellname = self.mesh.GetMaterials()[i]
                        localbasis, dofs = self.localbasis[cellname]

                        Vharm, aharm_mat, aharm_inv = self.vol_extensions[cellname]
                        gfu = GridFunction(Vharm)
                        
                        localcoeffs = Vector(len(dofs), complex = True)
                        
                        for d in range(len(dofs)):
                            localcoeffs[d] = coeffs[dofs[d]]
                    
                        gfu.vec.data = (localbasis * localcoeffs).Evaluate()
                        Draw(gfu, self.mesh, "cell")
                        
                        rr = gfu.Norm()
                        integral+= Integrate(rr, self.mesh, definedon = self.mesh.Boundaries(edgename))
                      
        
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

                ssss = time.time()
                els = [e for e in self.mesh.Boundaries(edge_name[1]).Elements()]
                nels = len(els)

                self.timings["calc_edgebasis_remaining"] += time.time() - ssss
                edgetype += "_" + str(nels)

                if self.calc_all:
                    edgetype = edge_name[1]
                

                if edgetype not in self.edgeversions:    
                    fd = self.V.GetDofs(self.mesh.Boundaries(edge_name[1])) & (~vertex_dofs)                           
                    base_space = H1(self.mesh, order = self.order)
                    Vloc = Compress(base_space, fd) 
                    if Vloc.ndof >= self.ndofemax:
                        self.ndofemax = Vloc.ndof

                    uloc, vloc = Vloc.TnT()
                    t = specialcf.tangential(2)
                    
                    
                    aloc = BilinearForm(Vloc, symmetric = True)
                    aloc += (grad(uloc)*t) * (grad(vloc)*t) * ds(skeleton=True, definedon = self.mesh.Boundaries(edge_name[1]), bonus_intorder = self.bi)

                    mloc = BilinearForm(Vloc, symmetric = True)
                    mloc += uloc.Trace() * vloc.Trace() * ds(skeleton = True, definedon = self.mesh.Boundaries(edge_name[1]), bonus_intorder = self.bi)

                    sss= time.time()
                    aloc.Assemble()
                    mloc.Assemble()
                    
                    self.timings["calc_edgebasis_assemble_and_inv"] += time.time() - sss
                    
                    sss = time.time()
                    try:
                        AA = sp.csr_matrix(aloc.mat.CSR())
                        MM = sp.csr_matrix(mloc.mat.CSR())
                        lams, uvecs =sp.linalg.eigs(A = AA, M = MM, k = self.edge_modes, which='SM', tol = 1e-8)
                        idx = lams.argsort()[::]   
                        lams = lams[idx]
                        uvecs = uvecs[:,idx]
                       
                        uvecs = uvecs.transpose()
                        self.edgeversions[edgetype] = [uvecs]
                        self.timings["calc_edgebasis_eigenvalues"] += time.time() - sss
                    except:
                        self.edge_modes = 0
                        break
                   
        self.timings["total_calc_edgebasis"] = time.time() - ss
        return (self.edge_modes != 0)
   
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
            
  