from utils_math import angle_vectors, euclide_distance
from constants import TypeClassification
from enum import Enum


class Orientation(Enum):
    FORWARD = 0
    BACKWARD = 1
    UNDEFINED = 2


class Edge(object):
    def __init__(self, graph, ID, n1, n2, n1_coord, n2_coord, orientation=Orientation.UNDEFINED):
        self.graph = graph
        self.ID = ID
        self.n1 = n1
        self.n2 = n2
        self.n1_coord = n1_coord
        self.n2_coord = n2_coord

        self.orientation = orientation

        self.probabilites_class = {}

        self.length = None
        self.coord_pixels = None
        self.skel_coord_pixels = None
        self.class_nn = None
        self.class_ASP = None
        self.class_GT = None
        self.class_EU = None

        self.highlight = False

    def assign_forward_orientation(self, node1, node2):
        self.orientation = Orientation.FORWARD
        if self.n1 == node1 and self.n2 == node2:
            pass
        elif self.n1 == node2 and self.n2 == node1:
            self.n1 = node1
            self.n2 = node2

            self.n1_coord, self.n2_coord = self.n2_coord, self.n1_coord

        else:
            assert False, "i nodi indicati non corrispondono a questo arco"

    def get_class(self, type_class):
        if type_class == TypeClassification.NN:
            return self.class_nn
        elif type_class == TypeClassification.ASP:
            return self.class_ASP
        elif type_class == TypeClassification.EU:
            return self.class_EU
        elif type_class == TypeClassification.GT:
            return self.class_GT

    def is_adjacent_with(self, edge):
        return (self.n1 == edge.n1 or self.n1 == edge.n2) or (self.n2 == edge.n1 or self.n2 == edge.n2)

    def get_angle_with(self, edge):
        common_node = self.get_common_node_with(edge)
        first_coord_pixels = self.get_first_coord_pixels_node(common_node)
        lineA = [self.get_coord_node(common_node), first_coord_pixels[-1]]

        lineB = [self.get_coord_node(common_node), edge.get_first_coord_pixels_node(common_node)[-1]]

        angle = angle_vectors(lineA, lineB)

        return angle

    def get_diameter(self):
        # print(f"arco {self.ID} num_pixels {len(self.coord_pixels)} len {self.get_length()} diam {len(self.coord_pixels) / self.get_length()}")
        # il diametro è definito come numero_pixels/lunghezza
        return len(self.coord_pixels) / self.get_length()

    def get_length(self):
        return self.length
        # return euclide_distance(self.n1_coord, self.n2_coord)

    def get_edge_with_highest_angle(self, edges):
        max_edge = edges[0]
        max_angle = self.get_angle_with(max_edge)
        for i in range(1, len(edges)):
            angle = self.get_angle_with(edges[i])
            if angle > max_angle:
                max_angle = angle
                max_edge = edges[i]

        print(f"Angolo max: {max_angle}")

        return max_edge

    def get_first_coord_pixels_node(self, node, num_pixels=20):
        # TODO: mettere in cache anziché calcolarlo ogni volta
        coord_node = self.get_coord_node(node)

        coord_pixels = []
        if self.skel_coord_pixels[0] == coord_node:
            for i in range(num_pixels):
                if i < len(self.skel_coord_pixels):
                    coord_pixels.append(self.skel_coord_pixels[i])
        elif self.skel_coord_pixels[-1] == coord_node:
            for i in range(num_pixels):
                if i < len(self.skel_coord_pixels):
                    coord_pixels.append(self.skel_coord_pixels[-i])
        else:
            assert False, "Ne inizio ne fine!!!"

        return coord_pixels

    def get_common_node_with(self, edge):
        if self.n1 == edge.n1 or self.n1 == edge.n2:
            return self.n1
        elif self.n2 == edge.n1 or self.n2 == edge.n2:
            return self.n2
        else:
            return None

    def get_coord_node(self, node):
        if self.n1 == node:
            return self.n1_coord
        elif self.n2 == node:
            return self.n2_coord
        else:
            assert False, f"Sono arco {self.ID} e non ho il nodo {node}"
            return None

    def percentage_common_skel_coordinates_smaller_edge(self, edge):
        common_elements = set(self.skel_coord_pixels) & set(edge.skel_coord_pixels)
        percentage_common_elements = 100 * len(common_elements) / len(edge.skel_coord_pixels)

        return percentage_common_elements

    def include_other_edge(self, edge):
        if self.length < edge.length:
            return False

        # se quello più piccolo ha una percentuale bassa di elementi in comune non lo considero sottoinsieme di questo
        if self.percentage_common_skel_coordinates_smaller_edge(edge) <= 70:
            return False

        return True

    def check(self):
        assert self.length is not None, f"Nessuna lunghezza per arco {self.ID}"
        assert self.coord_pixels is not None, f"Nessuna coordinata per arco {self.ID}"

        assert self.class_nn is not None, f"Nessuna classe nn per arco {self.ID}"
        assert self.class_ASP is not None, f"Nessuna classe ASP per arco {self.ID}"
        assert self.class_GT is not None, f"Nessuna classe GT per arco {self.ID}"

        assert self.graph.nodes[self.n1] is not None, f"Nodo n1 {self.n1} non esiste per arco {self.ID}"
        assert self.graph.nodes[self.n2] is not None, f"Nodo n2 {self.n2} non esiste per arco {self.ID}"
