# from flask import Flask, jsonify, request
# import math

# app = Flask(__name__)

# # Placeholder classes for geometric entities (replace with actual geometry library classes)
# class Point:
#     def __init__(self, x, y, z=0):
#         self.x = x
#         self.y = y
#         self.z = z

#     def distance(self, other):
#         return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

#     def is_equal(self, other, tolerance=1e-3):
#         return self.distance(other) < tolerance

#     def __repr__(self):
#         return f"Point({self.x}, {self.y}, {self.z})"


# class Edge:
#     def __init__(self, p1, p2):
#         self.p1 = p1
#         self.p2 = p2

#     def reversed(self):
#         return Edge(self.p2, self.p1)

#     def __repr__(self):
#         return f"Edge({self.p1}, {self.p2})"


# # Function 1: EdgesToWire
# @app.route('/edges_to_wire', methods=['POST'])
# def edges_to_wire():
#     data = request.json
#     vInputEdges = data["input_edges"]
#     vFacesInputOrder = data["face_order"]

#     vEdgesOfWire = [vInputEdges[0]]
#     vSortedFacesOrder = [vFacesInputOrder[0]]
#     nEdges = len(vInputEdges)
#     bEdgeReversed = False

#     for i in range(nEdges):
#         iEdge = vInputEdges[i]
#         iPnt2 = iEdge.p2 if not bEdgeReversed else iEdge.p1

#         for j in range(i + 1, nEdges):
#             jEdge = vInputEdges[j]
#             jPnt1 = jEdge.p1
#             jPnt2 = jEdge.p2

#             if iPnt2.is_equal(jPnt1):
#                 vInputEdges[i + 1], vInputEdges[j] = vInputEdges[j], vInputEdges[i + 1]
#                 vEdgesOfWire.append(vInputEdges[i + 1])
#                 vFacesInputOrder[i + 1], vFacesInputOrder[j] = vFacesInputOrder[j], vFacesInputOrder[i + 1]
#                 vSortedFacesOrder.append(vFacesInputOrder[i + 1])
#                 bEdgeReversed = False
#                 break
#             elif iPnt2.is_equal(jPnt2):
#                 vInputEdges[i + 1], vInputEdges[j] = vInputEdges[j], vInputEdges[i + 1]
#                 vEdgesOfWire.append(vInputEdges[i + 1])
#                 vFacesInputOrder[i + 1], vFacesInputOrder[j] = vFacesInputOrder[j], vFacesInputOrder[i + 1]
#                 vSortedFacesOrder.append(vFacesInputOrder[i + 1])
#                 bEdgeReversed = True
#                 break

#     return jsonify({
#         "ordered_edges": vEdgesOfWire,
#         "sorted_faces_order": vSortedFacesOrder
#     })

# # # Example placeholder for Wire creation using a CAD library:
# # wire = BRepBuilderAPI_MakeWire()
# # for edge in vEdgesOfWire:
# #     wire.Add(edge)

# # Function 2: FindStart
# @app.route('/find_start', methods=['POST'])
# def find_start():
#     data = request.json
#     vInputEdges = data["input_edges"]
#     startPnt = Point(*data["start_point"])
#     distance = 100000
#     StartEdgeNo = 0

#     for i, edge in enumerate(vInputEdges):
#         dist1 = edge.p1.distance(startPnt)
#         if dist1 < distance:
#             StartEdgeNo = i
#             distance = dist1

#     vInputEdges[0], vInputEdges[StartEdgeNo] = vInputEdges[StartEdgeNo], vInputEdges[0]

#     return jsonify({
#         "ordered_edges": vInputEdges
#     })


# # Function 3: EdgesOrdering
# @app.route('/edges_ordering', methods=['POST'])
# def edges_ordering():
#     data = request.json
#     vInputEdges = data["input_edges"]
#     vOrderedEdges = [vInputEdges[0]]
#     nEdges = len(vInputEdges)
#     bEdgeReversed = False

#     for i in range(nEdges):
#         iEdge = vInputEdges[i]
#         iPnt2 = iEdge.p2 if not bEdgeReversed else iEdge.p1

#         for j in range(i + 1, nEdges):
#             jEdge = vInputEdges[j]
#             jPnt1 = jEdge.p1
#             jPnt2 = jEdge.p2

#             if iPnt2.is_equal(jPnt1):
#                 vInputEdges[i + 1], vInputEdges[j] = vInputEdges[j], vInputEdges[i + 1]
#                 vOrderedEdges.append(vInputEdges[i + 1])
#                 bEdgeReversed = False
#                 break
#             elif iPnt2.is_equal(jPnt2):
#                 vInputEdges[i + 1], vInputEdges[j] = vInputEdges[j], vInputEdges[i + 1]
#                 vOrderedEdges.append(vInputEdges[i + 1])
#                 bEdgeReversed = True
#                 break

#     return jsonify({
#         "ordered_edges": vOrderedEdges
#     })


# # Function 4: OrientLoop
# @app.route('/orient_loop', methods=['POST'])
# def orient_loop():
#     data = request.json
#     vOrderedEdges = data["ordered_edges"]
#     dir_Orientation = Point(*data["orientation_dir"])
#     nEdges = len(vOrderedEdges)
#     vFirstVertices = [edge.p1 for edge in vOrderedEdges]
#     vFirstVertices.append(vFirstVertices[0])

#     Area = 0
#     pnt0 = Point(0, 0, 0)
#     for i in range(1, nEdges + 1):
#         vec1 = Point(vFirstVertices[i - 1].x - pnt0.x, vFirstVertices[i - 1].y - pnt0.y)
#         vec2 = Point(vFirstVertices[i].x - pnt0.x, vFirstVertices[i].y - pnt0.y)
#         Area += (vec1.x * vec2.y - vec2.x * vec1.y)

#     vOrientedEdges = vOrderedEdges if Area * dir_Orientation.z >= 0 else [edge.reversed() for edge in reversed(vOrderedEdges)]

#     return jsonify({
#         "oriented_edges": vOrientedEdges
#     })


# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for
import os
from gen_htp import Gen_HTP  # Import the Gen_HTP function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Retrieve form data
    slice_height = float(request.form['slice_height'])
    segm_length = float(request.form['segm_length'])
    store_path = request.form['store_path']

    # Mock data for `mvFeatureData` (replace with actual data structure)
    mvFeatureData = []  # Placeholder; add your shape data here

    # Generate HTP using the provided parameters
    Gen_HTP(slice_height, segm_length, mvFeatureData, store_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
