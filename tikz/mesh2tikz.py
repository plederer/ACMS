import sys
# from netgen.meshing import *
from ngsolve import *
from netgen.occ import *
from math import *

def unit_disc(maxH):
    
    r = 1 # Circle radius
    l = sqrt(2) * r # Edge of square
    

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

    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh = maxH))
    return mesh

def ExportTikz(mesh,tikzfile, **kwargs):
    w = kwargs.get('width','0.05pt')
    x = kwargs.get('xrange',None)
    y = kwargs.get('yrange',None)
    
    segm = []
    for el in mesh.Elements2D():
        n = len(el.vertices)
        for i in range(0,n-1):
            if el.vertices[i].nr < el.vertices[i+1].nr:
                segm.append([el.vertices[i],el.vertices[i+1]])
            else:
                segm.append([el.vertices[i+1],el.vertices[i]])
        if el.vertices[-1].nr < el.vertices[0].nr:
            segm.append([el.vertices[-1],el.vertices[0]])
        else:
            segm.append([el.vertices[0],el.vertices[-1]])
            
    lines = []
    for seg in segm:
        if seg not in lines:
            lines.append(seg)
            
    points = []
    if(x==None or y==None):
        for pnt in mesh.Points():
            points.append('('+str(pnt.p[0])+','+str(pnt.p[1])+')')
    else:
        eps = 1e-12
        for pnt in mesh.Points():
            if( x[0]-eps < pnt.p[0] < x[1]+eps and y[0]-eps < pnt.p[1] < y[1]+eps ):
                points.append('('+str(pnt.p[0])+','+str(pnt.p[1])+')')
            else:
                points.append('out')
        
    f = open(tikzfile,'w')
    #f.write(r'\documentclass[tikz]{standalone}')
    #f.write(r'\begin{document}')
    # f.write('\\begin{tikzpicture}[scale=5]\n')
    i = 0
    for point in points:
        print("point = ", point)
        i = i+1
        f.write('\\coordinate (' + str(i) + ') at ' + point+';\n')

    for line in lines:
        if(points[line[0].nr-1]!='out' and points[line[1].nr-1]!='out'):
            f.write('\\draw[line width='+w+'] '+points[line[0].nr-1]+' -- '+points[line[1].nr-1]+';\n')

    # f.write('\\end{tikzpicture}\n')
    # f.write(r'\end{document}')
    f.close()

if __name__ == "__main__":
    mesh = unit_disc(0.1)
    # Draw(mesh)
    # input()
    
    tikzfile = "circle_mesh.tex"
    # if(len(sys.argv)>=8):
    #     ExportTikz(meshfile,tikzfile,width=sys.argv[3],
    #                xrange=[int(sys.argv[4]),int(sys.argv[5])],
    #                yrange=[int(sys.argv[6]),int(sys.argv[7])])
    # elif(len(sys.argv)>=4):
    #     ExportTikz(meshfile,tikzfile,width=sys.argv[3])
    # else:
    ExportTikz(mesh.ngmesh,tikzfile)
