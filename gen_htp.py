from OCC.Core.BRep import BRep_Builder, BRepTools
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BOPAlgo import BOPAlgo_Operation
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Vec
from OCC.Core.Geom import Geom_Plane
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Edge, TopoDS_Shape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GeomLib import GeomLib_Tool
import os
class Point:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def is_equal(self, other, tolerance=1e-3):
        return self.distance(other) < tolerance

    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z})"


class Edge:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def reversed(self):
        return Edge(self.p2, self.p1)

    def __repr__(self):
        return f"Edge({self.p1}, {self.p2})"
def edges_to_wire():
    data = request.json
    vInputEdges = data["input_edges"]
    vFacesInputOrder = data["face_order"]

    vEdgesOfWire = [vInputEdges[0]]
    vSortedFacesOrder = [vFacesInputOrder[0]]
    nEdges = len(vInputEdges)
    bEdgeReversed = False

    for i in range(nEdges):
        iEdge = vInputEdges[i]
        iPnt2 = iEdge.p2 if not bEdgeReversed else iEdge.p1

        for j in range(i + 1, nEdges):
            jEdge = vInputEdges[j]
            jPnt1 = jEdge.p1
            jPnt2 = jEdge.p2

            if iPnt2.is_equal(jPnt1):
                vInputEdges[i + 1], vInputEdges[j] = vInputEdges[j], vInputEdges[i + 1]
                vEdgesOfWire.append(vInputEdges[i + 1])
                vFacesInputOrder[i + 1], vFacesInputOrder[j] = vFacesInputOrder[j], vFacesInputOrder[i + 1]
                vSortedFacesOrder.append(vFacesInputOrder[i + 1])
                bEdgeReversed = False
                break
            elif iPnt2.is_equal(jPnt2):
                vInputEdges[i + 1], vInputEdges[j] = vInputEdges[j], vInputEdges[i + 1]
                vEdgesOfWire.append(vInputEdges[i + 1])
                vFacesInputOrder[i + 1], vFacesInputOrder[j] = vFacesInputOrder[j], vFacesInputOrder[i + 1]
                vSortedFacesOrder.append(vFacesInputOrder[i + 1])
                bEdgeReversed = True
                break

    return jsonify({
        "ordered_edges": vEdgesOfWire,
        "sorted_faces_order": vSortedFacesOrder
    })

def GeneratePoints(aEdge, vContPnts, startPnt, length):
    # Get the vertices of the edge
    first_vertex = TopExp.FirstVertex(aEdge)
    last_vertex = TopExp.LastVertex(aEdge)

    # Get the geometric points of the vertices
    pnt_f_vertex = BRep_Tool.Pnt(first_vertex)
    pnt_l_vertex = BRep_Tool.Pnt(last_vertex)

    # Create a BRepAdaptor_Curve from the edge
    brep_curve = BRepAdaptor_Curve(aEdge)
    brep_first_param = brep_curve.FirstParameter()
    brep_last_param = brep_curve.LastParameter()

    # Calculate curve length
    c_length = GCPnts_AbscissaPoint.Length(brep_curve)
    length.append(c_length)

    # Number of points to generate along the curve
    n_points = 300
    dp = (brep_last_param - brep_first_param) / n_points

    # Store points based on distance from the start point
    if startPnt.Distance(pnt_f_vertex) < startPnt.Distance(pnt_l_vertex):
        for p in [brep_first_param + dp * i for i in range(1, n_points)]:
            pnt = brep_curve.Value(p)
            vContPnts.append(pnt)
    else:
        for p in [brep_last_param - dp * i for i in range(1, n_points)]:
            pnt = brep_curve.Value(p)
            vContPnts.append(pnt)

    return
def tool_normals(g_surface, v_pnts, v_normals, pfn):
    z_dir = gp_Dir(0, 0, 1)
    
    for point in v_pnts:
        # Get U and V parameters for the surface at the given point
        u_param, v_param = 0.0, 0.0
        if GeomLib_Tool.Parameters(g_surface, point, 1.0, u_param, v_param):
            # Get surface properties like normal at the (U, V) point
            surface_props = GeomLProp_SLProps(g_surface, u_param, v_param, 2, 1e-5)
            normal_dir = surface_props.Normal()
            
            # Invert the normal if it's pointing the wrong way
            if normal_dir.XYZ().Dot(z_dir.XYZ()) < 0:
                normal_dir.Reverse()
            
            normal_vec = gp_Vec(normal_dir)
            v_normals.append(normal_vec.XYZ())  # Store normal vector coordinates
            
            # Write the normal components to file
            pfn.write(f"{normal_vec.X():10.6f} \t {normal_vec.Y():10.6f} \t {normal_vec.Z():10.6f}\n")
def PrintPointsToFile(points, file):
    for point in points:
        file.write(f"{point.X():10.6f} {point.Y():10.6f} {point.Z():10.6f}\n")
def PrintHeightsToFile(points_count, slice_height, file):
    file.write(f"{points_count:5.0f} {slice_height:10.6f}\n")
def frange(start, stop, step):
    while start < stop:
        yield start
        start += step
def FindStart(edges, ordered_edges, start_point):
    closest_edge = None
    closest_distance = float('inf')
    start_point = gp_Pnt(0, 0, start_point.Z())

    # Find the edge closest to the start point
    for edge in edges:
        # Get the vertices of the edge
        first_vertex = BRep_Tool.FirstVertex(edge)
        last_vertex = BRep_Tool.LastVertex(edge)

        # Compute distances to the start point
        distance_to_first = first_vertex.Distance(start_point)
        distance_to_last = last_vertex.Distance(start_point)

        if distance_to_first < closest_distance:
            closest_distance = distance_to_first
            closest_edge = edge
            ordered_edges.clear()
            ordered_edges.append(edge)  # Start with the closest edge
        elif distance_to_last < closest_distance:
            closest_distance = distance_to_last
            closest_edge = edge
            ordered_edges.clear()
            ordered_edges.append(edge)

    # If a closest edge is found, add it to the ordered_edges list
    if closest_edge:
        ordered_edges.append(closest_edge)
def EdgesOrdering(slice_edges, ordered_edges):
    ordered_edges.clear()

    if not slice_edges:
        return

    # Start with the first edge and add it to ordered_edges
    ordered_edges.append(slice_edges.pop(0))

    bEdgeReversed = False  # This tracks if the current edge needs to be reversed

    while slice_edges:
        current_edge = ordered_edges[-1]
        
        # Get the last vertex of the current edge
        last_vertex = BRep_Tool.LastVertex(current_edge) if not bEdgeReversed else BRep_Tool.FirstVertex(current_edge)

        next_edge = None
        for i, edge in enumerate(slice_edges):
            first_vertex = BRep_Tool.FirstVertex(edge)
            last_edge_vertex = BRep_Tool.LastVertex(edge)

            # Check if the next edge can be connected directly or reversed
            if last_vertex.IsEqual(first_vertex, 1e-6):
                next_edge = edge
                bEdgeReversed = False  # Edge is connected as is
                slice_edges.pop(i)  # Remove the edge from the list
                break
            elif last_vertex.IsEqual(last_edge_vertex, 1e-6):
                next_edge = edge
                bEdgeReversed = True  # Edge needs to be reversed
                slice_edges.pop(i)
                break

        if next_edge:
            ordered_edges.append(next_edge)  # Add the found edge to the ordered list
        else:
            break  # No more edges found, the loop is closed


def OrientLoop(ordered_edges, oriented_edges, loop_orientation):
    if loop_orientation.X() == 0 and loop_orientation.Y() == 0 and loop_orientation.Z() == 0:
        return  # Invalid orientation vector

    # Check if the current order is correct, if not reverse it
    if len(ordered_edges) > 1:
        first_edge = ordered_edges[0]
        last_edge = ordered_edges[-1]

        # Get the last point of the first edge and the first point of the last edge
        first_vertex = BRep_Tool.FirstVertex(first_edge)
        last_vertex = BRep_Tool.LastVertex(last_edge)

        # Compute direction vectors
        direction_to_first = gp_Vec(last_vertex, first_vertex)

        # Check the orientation
        if direction_to_first.Dot(loop_orientation) < 0:
            ordered_edges.reverse()  # Reverse the order if the orientation is incorrect

    # Append the ordered edges to the oriented_edges list
    oriented_edges.extend(ordered_edges)
def Gen_HTP(slice_height, segm_length, mvFeatureData, str_storePath):
    # Initialize necessary variables
    dia_T1 = 12.7
    b_inclined = False
    b_adaptive = True
    b_spiral_gen = False
    scallop_ht = slice_height if b_adaptive else 0

    # Create a compound shape
    my_compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(my_compound)

    # Adding all shapes to the compound
    for feature in mvFeatureData:
        shape = feature.GetFaceData()
        builder.Add(my_compound, shape)
    

    # Create directories to store slice data
    os.makedirs(str_storePath, exist_ok=True)

    store_msg = f"HT{slice_height}_segLen{segm_length}"
    str_storePath = os.path.join(str_storePath, store_msg)
    os.makedirs(str_storePath, exist_ok=True)

    # Creating files to store slice data
    pf_Cont = open(os.path.join(str_storePath, "Contour.txt"), "w")
    pfn_Cont = open(os.path.join(str_storePath, "nContour.txt"), "w")
    pfh_Cont = open(os.path.join(str_storePath, "hContour.txt"), "w")
    pf_Spiral = open(os.path.join(str_storePath, "Spiral.txt"), "w")
    pfn_Spiral = open(os.path.join(str_storePath, "nSpiral.txt"), "w")
    pfh_Spiral = open(os.path.join(str_storePath, "hSpiral.txt"), "w")
    pf_noPnts = open(os.path.join(str_storePath, "noPoints_Contour.txt"), "w")
    pf_noNormals = open(os.path.join(str_storePath, "noNormals_Contour.txt"), "w")
    pf_SliceHt = open(os.path.join(str_storePath, "SliceHeights.txt"), "w")

    # Create slicing plane and perform slicing
    dir_loop_orientation = gp_Dir(0, 0, 1)
    # pnt_axis1 = gp_Pnt(1633, -200, 1)
    # pnt_axis2 = gp_Pnt(1633, 200, 1)
    start_point = gp_Pnt(0, 0, 19.5)
    vContPnts1 = []
    for z_ht in [z for z in frange(0.01, 40, slice_height)]:
        vContPnts1.clear()
        pf_SliceHt.write(f"{z_ht:10.6f} {slice_height:10.6f}\n")
        pnt_section_pln = gp_Pnt(0, 0, z_ht)
        # vec1 = gp_Vec(pnt_section_pln, pnt_axis1)
        # vec2 = gp_Vec(pnt_section_pln, pnt_axis2)
        dir_outer = gp_Dir(0,0,1)

        gpln_slicing = Geom_Plane(pnt_section_pln, dir_outer)
        section_algo = BRepAlgoAPI_Section(my_compound, gpln_slicing, False)
        section_algo.Approximation(True)
        section_algo.Build()
        a_section = section_algo.Shape()

        edge_exp = TopExp_Explorer(a_section, TopAbs_EDGE)
        v_section_edges = []
        while edge_exp.More():
            v_section_edges.append(TopoDS_Edge(edge_exp.Current()))
            edge_exp.Next()
        vLoopNormals = []
        noPoints_Contour = 0
        noNormals_contour = 0
        v_slice_edges_s, v_ordered_edges, v_oriented_edges = [], [], []
        FindStart(v_section_edges, v_slice_edges_s, start_point)
        EdgesOrdering(v_slice_edges_s, v_ordered_edges)
        OrientLoop(v_ordered_edges, v_oriented_edges, dir_loop_orientation)

        for aEdge in v_oriented_edges:
            vContPnts = []
            vNormals = []

            # Generate points on the edge
            edgeLength = 0  # You can calculate actual edge length here
            GeneratePoints(aEdge, vContPnts, startPnt, edgeLength)
            PrintPointsToFile(vContPnts, pf_Cont)

            # Surface and normals calculation
            ancestorShape = TopoDS_Face()
            bAncestor1 = section_algo.HasAncestorFaceOn1(aEdge, ancestorShape)
            gSurface = BRep_Tool.Surface(ancestorShape) if bAncestor1 else None

            if gSurface:
                tool_normals(gSurface, vContPnts, vNormals, pfn_Cont)

            if vContPnts:
                startPnt = vContPnts[-1]

            vLoopNormals.extend(vNormals)
            noPoints_Contour += len(vContPnts)
            noNormals_contour += len(vNormals)

            if len(vContPnts) != len(vNormals):
                continue

            vContPnts1.extend(vContPnts)

        pf_noPnts.write(f"{noPoints_Contour:5.0f} \n")
        pf_noNormals.write(f"{noNormals_contour:5.0f} \n")
        PrintHeightsToFile(noPoints_Contour, slice_height, pfh_Cont)

    pf_Cont.close()
    pfn_Cont.close()
    pfh_Cont.close()
    pf_noPnts.close()
    pf_noNormals.close()
    pf_SliceHt.close()
