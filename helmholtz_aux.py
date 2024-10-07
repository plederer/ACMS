from ngsolve import *
from netgen.geom2d import *

import pickle

import numpy 
import scipy.linalg
import scipy.sparse as sp

import os.path

from netgen.occ import *
from helping_functions import *
import time


def GetMeshinfo(mesh):
    dir_edges = ""
    edge_basis = []

    for e in range(len(mesh.GetBoundaries())):
        if not "inner_edge" in mesh.ngmesh.GetBCName(e):
            dir_edges += mesh.ngmesh.GetBCName(e) + "|"
            edge_basis.append((e, mesh.ngmesh.GetBCName(e)))

    dir_edges = dir_edges[:-1]
    
    
    vertex_basis = []
    vertex_basis_names = []
    vertex_basis_ps = []
    for v in range(len(mesh.GetBBoundaries())):
        if not "inner_vertex" in mesh.ngmesh.GetCD2Name(v):
            # print(mesh.ngmesh.Get)
            # print(mesh.ngmesh.Points()[v+1])
            # vertex_basis.append((v, mesh.ngmesh.GetCD2Name(v)))
            vertex_basis.append(v)
            vertex_basis_ps.append(list(mesh.ngmesh.Points()[v+1])[0:-1])
            vertex_basis_names.append(mesh.ngmesh.GetCD2Name(v))

    vertex_basis_ps = np.array(vertex_basis_ps[:])
    p1 = np.array(vertex_basis_ps[:,1] , dtype='f')
    p0 = np.array(vertex_basis_ps[:,0], dtype='f')
    ind = np.lexsort((p1, p0))
    vertex_basis = np.array(vertex_basis)[ind]
    vertex_basis_names = np.array(vertex_basis_names)[ind]
    vertices = list(zip(vertex_basis, vertex_basis_names))
    return {"dir_edges": dir_edges, "verts": vertices, "edges": edge_basis}


##################################################################
##################################################################


def unit_disc(maxH):
    
    l = sqrt(2) # Edge of square
    r = 1 # Circle radius

    circ = WorkPlane().Circle(0, 0, r).Face()
    rect = WorkPlane().Rotate(45).RectangleC(l, l).Face()
    quadUR = MoveTo(r/2,r/2).RectangleC(r, r).Face()
    # quadUR.edges[0] = 
    quadUL = MoveTo(-r/2,r/2).RectangleC(r, r).Face()
    quadLR = MoveTo(r/2,-r/2).RectangleC(r, r).Face()
    quadLL = MoveTo(-r/2,-r/2).RectangleC(r, r).Face()

    triangleUR = rect - quadUR 
    triangleUR.faces.edges[0].name = "test_D"
    triangleUR.faces.edges[2].name = "test_V"
    triangleUR.faces.edges[3].name = "test_H"
    triangleUR.faces.edges[1].name = "test_D"
    triangleUR.faces.edges[4].name = "test_D"

    
    triangleUL = rect - quadUL
    triangleUL.faces.edges[1].name = "test_H"
    triangleUL.faces.edges[3].name = "test_D"
    triangleLR = rect - quadLR
    triangleLR.faces.edges[4].name = "test_V"
    triangleLL = rect - quadLL

    circ.edges.name = "dom_bnd_C"
    outer = circ - rect

    shape = Glue([triangleUR, triangleUL, triangleLR, triangleLL, outer])
    # shape = triangleUR
    # DrawGeo(shape)
    nels_r = int(r/maxH)
    nels_d = int(l/maxH)
    length_arc = 1/4 * 2 * r * math.pi
    nels_c = int(length_arc/maxH)

    partition_r = [i*1/nels_r for i in range(1,nels_r)]
    partition_d = [i*1/nels_d for i in range(1,nels_d)]
    partition_c = [i*1/nels_c for i in range(1,nels_c)]
    for ed in shape.edges:
        if "H" in ed.name or "V" in ed.name:
            ed.partition = partition_r
        if "D" in ed.name:
            ed.partition = partition_d
        if "C" in ed.name:
            ed.partition = partition_c
    
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh = maxH))
    mesh.Curve(10)

    nmat = len(mesh.GetMaterials())
    nbnd = len(mesh.GetBoundaries())
    nvert = len(mesh.GetBBoundaries())

    for i in range(nmat):
        mesh.ngmesh.SetMaterial(i+1,"D" + str(i))

    dom_bnd = ""

    bi = 0
    for i in range(nbnd):
        if not "dom_bnd" in mesh.ngmesh.GetBCName(i): # != "dom_bnd":
            if "V" in mesh.ngmesh.GetBCName(i):
                mesh.ngmesh.SetBCName(i,"E_V" + str(i))
            elif "H" in mesh.ngmesh.GetBCName(i):
                mesh.ngmesh.SetBCName(i,"E_H" + str(i))
            else:
                mesh.ngmesh.SetBCName(i,"E_D" + str(i))
        else:
            mesh.ngmesh.SetBCName(i,"dom_bnd_C_" + str(bi))
            dom_bnd += "dom_bnd_C_" + str(bi) + "|"
            bi+=1
    dom_bnd = dom_bnd[:-1]
    
    for i in range(nvert):
        mesh.ngmesh.SetCD2Name(i+1,"V" + str(i))

        
    # print(mesh.nv)
    # print(mesh.GetMaterials()) # 8 subdomains
    # print(mesh.GetBoundaries()) # 12 edges
    # print(mesh.GetBBoundaries()) # 5 vertices

    alpha = 1 #mesh.MaterialCF(1, default=0)

    mesh_info = GetMeshinfo(mesh)
    mesh_info["dom_bnd"] = dom_bnd

    return mesh, dom_bnd, alpha, mesh_info


def crystal_geometry(maxH, Nx, Ny, incl, r, Lx, Ly, alpha_outer = 1, alpha_inner = 1, defects = np.ones((0,0)), layers = 0, load_mesh = False, save_mesh = True):
    pickle_name =   "maxH:" +     str(maxH) + "_" + \
                    "Nx:" +       str(Nx) + "_" + \
                    "Ny:" +       str(Ny) + "_" + \
                    "incl:" +     str(incl) + "_" + \
                    "r:" +        str(r) + "_" + \
                    "Lx:" +       str(Lx) + "_" + \
                    "Ly:" +       str(Ly) + "_" + \
                    "ai:" +       str(alpha_outer) + "_" + \
                    "ao:" +       str(alpha_inner) + "_" + \
                    "layers:" +   str(layers) + "_" + \
                    ".dat"
    dirname = os.path.dirname(__file__)
    
    if not os.path.exists("meshes"):
        os.mkdir("meshes")

    load_file = os.path.join(dirname, "meshes/" + pickle_name)
    
    gm = 0
    if load_mesh == True:
        try:
            with TaskManager():
                picklefile = open(load_file, "rb")
                data = pickle.load(picklefile)
                geo = data["geo"]
                mesh = Mesh("mesh_" + pickle_name + ".vol.gz")
                mesh.ngmesh.SetGeometry(geo)
                mesh.Curve(10)
                dom_bnd = data["dom_bnd"]
                # alpha = data["alpha"]
                # mesh_info = data["mesh_info"]
                picklefile.close()
                print(60 * "#")
                print("Loaded mesh!!!")
                print(60 * "#")
        except:
            gm = 1
    else:
        gm = 1

    if gm == 1:
        print(60 * "#")
        print("Generating mesh!!!")
        print(60 * "#")
        # if len(defects) == 0:
        #     defects = np.ones((Nx,Ny))

        #Crystal type: 0 = full crystal # 1 = air defect in crystal  # 2 = air
        crystaltype = [["outer","outer"], 
                    ["outer","inner"],
                    ["inner", "inner"]]

        domain = [MoveTo(Lx*i,Ly*j).RectangleC(Lx,Ly).Face() for i in range(Nx) for j in range(Ny)]
        
        if incl == 1: #Circular inclusion
            inclusion = [MoveTo(0,0).Circle(Lx*i,Ly*j, r).Face() for i in range(Nx) for j in range(Ny)]
                    
        elif incl >= 2: #Circular inclusions (incl on one side)
            inclusion = []
            for i in range(Nx):
                for j in range(Ny):   
                    Mx = Lx*i
                    My = Ly*j
                    exponent = 2
                    circular_incl = [MoveTo(0,0).Circle(Mx + (-1)**ee * Lx/incl * (kk + 1/2), My + (-1)**ff * Ly/incl * (ll + 1/2), r).Face()  for ee in range(exponent)  for ff in range(exponent) for kk in range(incl//2) for ll in range(incl//2)]
                    inclusion.append(Glue([circ for circ in circular_incl]))

        elif incl == 0: #Square inclusion (incl = 0)
            inclusion = [MoveTo(Lx*i,Ly*j).RectangleC(r, r).Face() for i in range(Nx) for j in range(Ny)]
        else:
            inclusion = []
        # outer = [domain[i*Ny+j]-inclusion[i*Ny+j] for i in range(Nx) for j in range(Ny)]
        # inner = [domain[i*Ny+j]*inclusion[i*Ny+j] for i in range(Nx) for j in range(Ny)]
        outer = []
        inner = []

        def GetCell(i,j):
            outerdom = domain[i*Ny+j]
            # outerdom.faces.name = "outer"+str(i*Ny+j)
            if incl != -1:
                # outerdom.faces.name = crystaltype[int(defects[i,j])][0]+str(i*Ny+j)
                outerdom.faces.name = "outer"+str(i*Ny+j)
                outerdom = outerdom - inclusion[i*Ny+j]
            else:
                outerdom.faces.name = "outer" +str(i*Ny+j)
            outerdom.faces.edges.Min(Y).name = "E_H"
            outerdom.faces.edges.Max(Y).name = "E_H"
            outerdom.faces.edges.Min(X).name = "E_V"
            outerdom.faces.edges.Max(X).name = "E_V"
            
            if incl != -1:
                innerdom = domain[i*Ny+j]*inclusion[i*Ny+j]
                innerdom.faces.edges.name="inner_edge"+str(i*Ny+j)
                innerdom.faces.vertices.name="inner_vertex"+str(i*Ny+j)
                # innerdom.faces.name=crystaltype[int(defects[i,j])][1]+str(i*Ny+j)
                innerdom.faces.name="inner"+str(i*Ny+j)
            else:
                innerdom = []

            if (j == 0) :
                outerdom.faces.edges.Min(Y).name = "dom_bnd_bottom_H"
            if (j == (Ny-1)) :
                outerdom.faces.edges.Max(Y).name = "dom_bnd_top_H"
            if (i == 0):
                outerdom.faces.edges.Min(X).name = "dom_bnd_left_V"
            if (i == (Nx-1)) :
                outerdom.faces.edges.Max(X).name = "dom_bnd_right_V"
            
            if layers > 0:
                if (j == layers) and (i >= layers) and (i <= Nx-1-layers):
                    outerdom.faces.edges.Min(Y).name = "crystal_bnd_bottom_H"
                if (j == (Ny-1-layers)) and (i >= layers) and (i <= Nx-1-layers) :
                    outerdom.faces.edges.Max(Y).name = "crystal_bnd_top_H"
                if (i == layers) and (j >= layers) and (j <= Ny-1-layers):
                    outerdom.faces.edges.Min(X).name = "crystal_bnd_left_V"
                if (i == (Nx-1-layers)) and (j >= layers) and (j <= Ny-1-layers) :
                    outerdom.faces.edges.Max(X).name = "crystal_bnd_right_V"

            return innerdom, outerdom
        
        
        for i in range(layers, Nx-layers):
            for j in range(layers, Ny-layers):
                innerdom, outerdom = GetCell(i,j)
                outer.append(outerdom)
                if incl != -1:
                    inner.append(innerdom)
        
        if layers > 0:

            for i in [ii for ii in range(0, layers)] + [ii for ii in range(Nx-layers,Nx)]:
                for j in range(0,Ny):
                    innerdom, outerdom = GetCell(i,j)
                    outer.append(outerdom)
                    inner.append(innerdom)
            
            for j in [jj for jj in range(0, layers)] + [jj for jj in range(Ny-layers,Ny)]:
                for i in range(layers,Nx-layers):
                    innerdom, outerdom = GetCell(i,j)
                    outer.append(outerdom)
                    inner.append(innerdom)
            
        
        outershapes = [out_dom for out_dom in outer]
        innershapes = [in_dom for in_dom in inner]

        
        crystalshape = Glue(outershapes + innershapes)
        
        nels_x = int(Lx/maxH)
        nels_y = int(Ly/maxH)
        partition_x = [i*1/nels_x for i in range(1,nels_x)]
        partition_y = [i*1/nels_y for i in range(1,nels_y)]
        for ed in crystalshape.edges:
            if "H" in ed.name:
                ed.partition = partition_x
            if "V" in ed.name:
                ed.partition = partition_y
            

        with TaskManager():
            geo = OCCGeometry(crystalshape, dim=2)
            mesh = Mesh(geo.GenerateMesh(maxh = maxH))
            mesh.Curve(10)
        # Draw(mesh)
        # input()
        
        nmat = len(mesh.GetMaterials())
        nbnd = len(mesh.GetBoundaries())
        nvert = len(mesh.GetBBoundaries())
        
        dom_bnd = ""
        bi_V = 0
        bi_H = 0
        for i in range(nbnd): #
            if not "dom_bnd" in mesh.ngmesh.GetBCName(i): # != "dom_bnd":
                if not "inner_edge" in mesh.ngmesh.GetBCName(i): # != "dom_bnd":
                    if not "crystal_bnd" in mesh.ngmesh.GetBCName(i):
                        if "V" in mesh.ngmesh.GetBCName(i):
                            mesh.ngmesh.SetBCName(i,"E_V" + str(i))
                        else:
                            mesh.ngmesh.SetBCName(i,"E_H" + str(i))
                    else:
                        name = mesh.ngmesh.GetBCName(i)
                        # print(name)
                        mesh.ngmesh.SetBCName(i,name + str(i))
            else:
                if "V" in mesh.ngmesh.GetBCName(i):
                    name = mesh.ngmesh.GetBCName(i)
                    mesh.ngmesh.SetBCName(i,name + str(bi_V))
                    dom_bnd += name + str(bi_V) + "|"
                    bi_V += 1
                else:
                    name = mesh.ngmesh.GetBCName(i)
                    mesh.ngmesh.SetBCName(i,name + str(bi_H))
                    dom_bnd += name + str(bi_H) + "|"
                    bi_H += 1
                # bi+=1
        dom_bnd = dom_bnd[:-1]
        
        for i in range(nvert): #Removing vertices on circles
            if not "inner_vertex" in mesh.ngmesh.GetCD2Name(i):
                mesh.ngmesh.SetCD2Name(i+1,"V" + str(i))

        # Draw(mesh)

        if save_mesh:
            with TaskManager():
                picklefile = open(load_file, "wb")
                data = {}
                # data["ngmesh"] = mesh.ngmesh
                # data[""]
                data["geo"] = geo
                data["dom_bnd"] = dom_bnd
                # data["mesh_info"] = mesh_info
                # data["alpha"] = alpha
                pickle.dump(data, picklefile)
                picklefile.close()
                mesh.ngmesh.Save("mesh_" + pickle_name + ".vol.gz")
    
    coeffs = {}
    nmat = len(mesh.GetMaterials())
    for d in range(len(mesh.GetMaterials())):
        dom_name = mesh.ngmesh.GetMaterial(d+1) 
        if "outer" in dom_name:
            coeffs[dom_name] = alpha_outer
        else:
            dd = int(dom_name[5:]) 
            ii = dd// Ny
            jj = dd - ii * Ny
            if defects[ii,jj] == 0:
                coeffs[dom_name] = alpha_outer
            else:
                coeffs[dom_name] = alpha_inner

    alpha_cf = mesh.MaterialCF(coeffs, default=0)
    
    alpha = GridFunction(L2(mesh, order = 0))
    alpha.Set(alpha_cf)
                
    # ########################
    # rename inner domains give them the same name as the outer one has inner name just used 
    for d in range(nmat):
        if "inner" in mesh.ngmesh.GetMaterial(d+1): # the +1 comes from the asking the negten mesh instead of the ngsolve mesh!
            offset = int(nmat/(1+incl**2))
            ii = int((d-offset)/incl**2) 
            mesh.ngmesh.SetMaterial(d+1, mesh.ngmesh.GetMaterial(ii+1))
    
    mesh_info = GetMeshinfo(mesh)
    mesh_info["dom_bnd"] = dom_bnd
    mesh_info["Nx"] = Nx
    mesh_info["Ny"] = Ny
    mesh_info["Ncell"] = Ny
        
    # ########################
    # definition of diffusion coefficient: alpha_outer = 1/12.1 #SILICON  # alpha_inner = 1 #AIR
    
    Draw(alpha, mesh, "alpha")

    return mesh, dom_bnd, alpha, mesh_info
