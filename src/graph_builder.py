import numpy as np
import pickle
import networkx as nx
import cv2
import constants
import re
from node import Node
from edge import Orientation, Edge
from skimage import morphology
from skimage.measure import label
from pixels_edge_finder import get_simple_path
from utils_math import is_in_quadrilateral, euclide_distance
from utils_image import show_cells, colorize_cells, show_image, get_matrix_cells, show_single_edge, show_angle_edges_tmp
from pixels_edge_finder import get_path_of_2_nodes


class GraphBuilder(object):
    def __init__(self, cache_graph):
        self.skel = None
        self.black_white_img = None
        self.graph = None

        self.graphs = None
        self.cache_graph = cache_graph
        if self.cache_graph:
            self.graphs = pickle.load(open("graphs.pickle", "rb"))

        self.source_nodes = None

    def generate_graph_from_black_white_img(self, black_white_img, coord_disk, idx_img=-1):
        if black_white_img is not None:
            self.black_white_img = black_white_img
            self.skel = self.preprocess_image(self.black_white_img, x=0, y=0, disk_radius=1)

        if self.cache_graph:
            assert idx_img >= 0, "Cache abilitata: inserire indice per l'immagine"
            self.graph = self.graphs[idx_img]
        else:
            self.graph = self.generate_graph(self.skel, black_white_img)

        # Quando genero il grafo, skel deve essere con il PAD per questioni di performance
        # una volta generato il grafo, il PAD crea solo problemi e quindi lo rimuovo per usi futuri
        if self.skel is not None:
            self.skel = self.remove_pad_to_img(self.skel)

        self.improve_graph()

        self.set_idx_to_edges()

        self.find_and_set_source_nodes(coord_disk)

        return self.graph

    def improve_graph(self):
        self.remove_small_edges()
        self.remove_small_end_edges()
        self.remove_nodes_with_2_edges()

        # self.merge_nodes_probable_crossover_with_angle()

        self.merge_nodes_probable_crossover()

    def set_idx_to_edges(self):
        for idx, edge in enumerate(self.graph.edges):
            self.graph[edge[0]][edge[1]]["ID"] = idx

    def find_and_set_source_nodes(self, coord_disk):
        self.source_nodes = self.get_nodes_in_circle(*coord_disk)
        assert self.source_nodes, f"Errore source nodes: nessun nodo trovato in {coord_disk}"

    def get_nodes_in_circle(self, r, c, radius):
        nodes = []
        dict_coord = nx.get_node_attributes(self.graph, "coordinates")
        for node in self.graph.nodes:
            coord = dict_coord[node]
            if euclide_distance(coord, (r, c)) < radius:
                nodes.append(node)

        return nodes

    def print_degree_nodes(self, nums):
        for node in self.graph.nodes:
            edges_connected = list(self.graph.edges(node))
            n = len(edges_connected)
            if n == 0:
                nums[0] += 1
            elif n == 1:
                nums[1] += 1
            elif n == 2:
                nums[2] += 1
            elif n == 3:
                nums[3] += 1
            elif n == 4:
                nums[4] += 1
            elif n == 5:
                nums[5] += 1
            elif n == 6:
                nums[6] += 1
            elif n == 7:
                nums[7] += 1
            else:
                nums[8] += 1

        return nums

    def generate_graph(self, skel, img):
        # crea un np array di 0 della shape di skel
        junctions = np.zeros_like(skel)

        # Find row and column locations that are non-zero
        (rows, cols) = np.nonzero(skel)

        # Initialize empty list of coordinates
        skel_coords = []

        # For each non-zero pixel...
        for (r, c) in zip(rows, cols):
            # Extract an 8-connected neighbourhood
            (col_neigh, row_neigh) = np.meshgrid(np.array([c - 1, c, c + 1]), np.array([r - 1, r, r + 1]))
            (col_neigh_js, row_neigh_js) = np.meshgrid(np.array([c - 2, c, c + 2]), np.array([r - 2, r, r + 2]))

            # Cast to int to index into image
            col_neigh = col_neigh.astype("int")
            row_neigh = row_neigh.astype("int")

            # Convert into a single 1D array and check for non-zero locations
            pix_neighbourhood = skel[row_neigh, col_neigh].ravel() != 0

            # If the number of non-zero locations equals 2, add this to
            # our list of co-ordinates
            if np.sum(pix_neighbourhood) >= 4:
                skel_coords.append((r, c))
                junctions[r, c] = 1
            elif np.sum(pix_neighbourhood) == 2:
                junctions[r, c] = 1

        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(junctions, kernel, iterations=1)

        branchs = cv2.subtract(skel, junctions)
        lbranchs = label(branchs)
        ljunctions = label(junctions)

        g = nx.Graph()

        kernel = np.ones((4, 4), np.uint8)
        for i in range(1, lbranchs.max()):
            dilation = cv2.dilate(np.uint8(lbranchs == i), kernel, iterations=1)
            tmp = dilation + junctions
            (rows, cols) = np.nonzero(tmp == tmp.max())

            # find the intersection with the nodes
            unique_c = []
            unique_i = []
            for (r, c) in zip(rows, cols):
                if not ljunctions[r, c] in unique_i:
                    unique_c.append((r, c))
                    unique_i.append(ljunctions[r, c])

            # compute edge properties
            (rows, cols) = np.nonzero(lbranchs == i)
            path, avg = 0, 0
            for (r, c) in zip(rows, cols):
                d = self.diam(r, c, img)
                avg += d
                path += 1
            avg_diam = avg / float(path)

            if len(unique_i) > 1:
                n1, n2 = unique_i[:2]

                x1 = unique_c[0][0] - constants.PAD
                y1 = unique_c[0][1] - constants.PAD

                x2 = unique_c[1][0] - constants.PAD
                y2 = unique_c[1][1] - constants.PAD

                # la rotazione di 90° la ottengo trasformando (x,y) in (y, altezza - x)
                # g.add_node(n1, x=y1, y=skel.shape[0] - x1)
                # g.add_node(n2, x=y2, y=skel.shape[0] - x2)
                # g.add_edge(n1, n2, length=path, weight=avg_diam, etype=None)

                # dict_nodes_coordinates[n1] = (y1, skel.shape[0] - x1)
                # dict_nodes_coordinates[n2] = (y2, skel.shape[0] - x2)

                g.add_node(n1, coordinates=(x1, y1))
                g.add_node(n2, coordinates=(x2, y2))
                g.add_edge(n1, n2, length=path, weight=avg_diam, etype=None)

                # dict_nodes_coordinates[n1] = (x1, y1)
                # dict_nodes_coordinates[n2] = (x2, y2)

                # print("Aggiunti nodi: ", n1, n2)

        for n in g.nodes():
            if g.degree(n) > 1:
                g.nodes[n]["ntype"] = "bif"
            else:
                g.nodes[n]["ntype"] = "end"

        return g

    def diam(self, x, y, not_ske):
        for r in range(1, min(not_ske.shape)):
            if np.sum(not_ske[x - r : x + r + 1, y - r : y + r + 1]) < ((2 * r) ** 2):
                return 2 * r
        return 0

    def preprocess_image(self, img, x, y, disk_radius):
        WHITE = 1
        # disegna il cerchio nel punto in cui c'è il disco ottico
        img = cv2.circle(img, (x, y), disk_radius, (0, 0, 0), -1)

        # trasforma l'immagine in una matrice di boolean: VERO se il valore è > di 255/2 FALSO altrimenti
        # praticamente ciò che è nero diventa FALSO, il resto (grigio-bianco) vero
        img = img > 0

        # rimuove oggetti più piccoli della size specificata
        # connectivity indica il vicinato dei pixel
        # penso che serva a togliere dei "blocchi" di pixel
        img = morphology.remove_small_objects(img, min_size=100, connectivity=12)

        # aggiunge una cornice di 0 intorno a tutta l'immagine, PAD = 10
        img = np.pad(img, pad_width=[(constants.PAD, constants.PAD), (constants.PAD, constants.PAD)], mode="constant", constant_values=0)

        # produce lo skeleton di una immagine binaria: praticamente riduce a singoli pixel gli oggetti grandi
        # le linee diventa larghe 1 pixel
        # l'operazione img // img.max() trasforma i True in 1 e i False in 0
        skel = morphology.skeletonize(img // img.max())
        skel = np.uint8(skel) * WHITE

        return skel

    def remove_small_edges(self):
        # fonde insieme due nodi se la loro distanza è minore di una soglia

        # se la distanza è inferiore alla soglia:
        # fondi il primo nodo con il secondo
        # ripeti il ciclo
        # se nessuno nodo è stato fuso con un altro esci dalla funzione
        threshold = 6
        nodes_merged = True
        while nodes_merged:
            nodes_merged = False
            nodes = list(self.graph.nodes)
            # per ogni nodo del grafo
            for node in nodes:
                adjacent_nodes = self.get_nodes_adjacent_to_node(node)
                # prendi la distanza con ognuno di questi nodi
                for adjacent_node in adjacent_nodes:
                    distance = self.get_distance_from_2_nodes(node, adjacent_node)
                    if distance < threshold:
                        self.merge_2_nodes(node, adjacent_node)
                        nodes_merged = True
                        break

                if nodes_merged:
                    break

        return self.graph

    def remove_small_end_edges(self):
        # fonde insieme due nodi se la loro distanza è minore di una soglia

        # se la distanza è inferiore alla soglia:
        # fondi il primo nodo con il secondo
        # ripeti il ciclo
        # se nessuno nodo è stato fuso con un altro esci dalla funzione
        threshold = 20
        nodes_merged = True
        while nodes_merged:
            nodes_merged = False
            nodes = list(self.graph.nodes)
            # per ogni nodo del grafo
            for node in nodes:
                adjacent_nodes = self.get_nodes_adjacent_to_node(node)
                if len(adjacent_nodes) == 1:
                    # prendi la distanza con ognuno di questi nodi
                    adjacent_node = adjacent_nodes[0]
                    distance = self.get_distance_from_2_nodes(node, adjacent_node)
                    if distance < threshold:
                        self.merge_2_nodes(node, adjacent_node)
                        nodes_merged = True
                        break

                    if nodes_merged:
                        break

        return self.graph

    def remove_nodes_with_2_edges(self):
        node_removed = True
        while node_removed:
            node_removed = False
            nodes = list(self.graph.nodes)
            for node in nodes:
                # il numero di nodi adiacenti è uguale al numero di archi attaccati al nodo
                adjacent_nodes = self.get_nodes_adjacent_to_node(node)
                if len(adjacent_nodes) == 2:
                    # rimuovo i due archi con questi due nodi
                    self.remove_edges_connected_to_node(node)

                    # rimuovo il nodo
                    self.graph.remove_node(node)

                    # aggiungo il singolo arco che connette i due nodi
                    nodeA = adjacent_nodes[0]
                    nodeB = adjacent_nodes[1]
                    self.graph.add_edge(nodeA, nodeB)

                    node_removed = True
                    break

        return self.graph

    def merge_nodes_probable_crossover(self):
        threshold = 12
        nodes_merged = True
        while nodes_merged:
            nodes_merged = False
            nodes = list(self.graph.nodes)
            # per ogni nodo del grafo
            for node in nodes:
                adjacent_nodes = self.get_nodes_adjacent_to_node(node)
                if len(adjacent_nodes) != 3:
                    continue

                # prendi la distanza con ognuno di questi nodi
                for adjacent_node in adjacent_nodes:
                    if len(self.graph.edges(adjacent_node)) != 3:
                        continue

                    distance = self.get_distance_from_2_nodes(node, adjacent_node)
                    if distance < threshold:
                        self.merge_node_to_node_middle_ground(node, adjacent_node)
                        nodes_merged = True
                        break

                if nodes_merged:
                    break

        return self.graph

    def merge_nodes_probable_crossover_with_angle(self):
        threshold = 12
        nodes_merged = True
        while nodes_merged:
            nodes_merged = False
            nodes = list(self.graph.nodes)
            # per ogni nodo del grafo
            for node in nodes:
                adjacent_nodes = self.get_nodes_adjacent_to_node(node)
                if len(adjacent_nodes) != 3:
                    continue

                # prendi la distanza con ognuno di questi nodi
                for adjacent_node in adjacent_nodes:
                    if len(self.graph.edges(adjacent_node)) != 3:
                        continue

                    distance = self.get_distance_from_2_nodes(node, adjacent_node)
                    if distance < threshold:

                        self.merge_node_to_node_middle_ground(node, adjacent_node)
                        nodes_merged = True
                        break

                if nodes_merged:
                    break

        return self.graph

    def get_nodes_adjacent_to_node(self, node):
        # individua tutti gli archi collegati ad esso
        edges_connected = list(self.graph.edges(node))

        # prendi i nodi di questi archi
        adjacent_nodes = []
        for edge in edges_connected:
            if edge[0] != node:
                adjacent_nodes.append(edge[0])
            elif edge[1] != node:
                adjacent_nodes.append(edge[1])

        return adjacent_nodes

    def get_distance_from_2_nodes(self, nodeA, nodeB):
        dict_nodes_coordinates = nx.get_node_attributes(self.graph, "coordinates")
        coordA = dict_nodes_coordinates[nodeA]
        coordB = dict_nodes_coordinates[nodeB]

        # return len(get_simple_path(coordA, coordB))
        return euclide_distance(coordA, coordB)

    def get_nodes_with_distance_from_any_source_nodes(self, distance):
        nodes = set()
        for node in self.graph.nodes:
            if self.get_minimum_distance_from_any_source_node_BFS(node) == distance:
                nodes.add(node)

        return nodes

    def merge_2_nodes(self, nodeA, nodeB):
        edges_connectedA = list(self.graph.edges(nodeA))
        edges_connectedB = list(self.graph.edges(nodeB))
        start = nodeA
        end = nodeB
        # # voglio unire il nodo che ha meno archi collegati a quello che ha più archi collegati
        if len(edges_connectedB) < len(edges_connectedA):
            start = nodeB
            end = nodeA

        self.merge_node_to_node(start, end)

    def merge_node_to_node(self, nodeA, nodeB):
        # unisco il nodo A al nodo B

        # prendo tutti i nodi adiacenti ad A
        nodes_adjacent_A = self.get_nodes_adjacent_to_node(nodeA)

        # tolgo il nodoB
        nodes_adjacent_A.remove(nodeB)

        # aggiungo un arco da ognuno di questi nodi al nodo B
        for node_adjacent in nodes_adjacent_A:
            self.graph.add_edge(node_adjacent, nodeB)

        # elimino tutti i vecchi archi
        self.remove_edges_connected_to_node(nodeA)

        # elimino il nodo che si è fuso
        self.graph.remove_node(nodeA)

    def get_coordinate_node(self, node):
        dict_coord = nx.get_node_attributes(self.graph, "coordinates")
        return dict_coord[node]

    def merge_node_to_node_middle_ground(self, nodeA, nodeB):
        # unisco il nodo A al nodo B creando un nuovo nodo a metà strada tra i due

        dict_coord = nx.get_node_attributes(self.graph, "coordinates")
        # prendo le coordinate del path tra A e B
        A_coord = dict_coord[nodeA]
        B_coord = dict_coord[nodeB]
        skel_path = get_path_of_2_nodes(self.skel, A_coord, B_coord)

        # prendo una coordinata "centrale"
        central_coord = skel_path[int(len(skel_path) / 2)]

        # creo il nuovo nodo
        new_node = self.get_new_ID_node()
        self.graph.add_node(new_node, coordinates=central_coord)

        # prendo tutti i nodi adiacenti ad A
        nodes_adjacent_A = self.get_nodes_adjacent_to_node(nodeA)

        # tolgo il nodoB
        nodes_adjacent_A.remove(nodeB)

        # aggiungo un arco da ognuno di questi nodi al nuovo nodo
        for node_adjacent in nodes_adjacent_A:
            self.graph.add_edge(node_adjacent, new_node)

        # elimino tutti i vecchi archi
        self.remove_edges_connected_to_node(nodeA)

        # elimino il nodoA
        self.graph.remove_node(nodeA)

        # prendo tutti i nodi adiacenti ad B
        nodes_adjacent_B = self.get_nodes_adjacent_to_node(nodeB)

        # aggiungo un arco dal nuovo nodo ad ognuno di questi nodi
        for node_adjacent in nodes_adjacent_B:
            self.graph.add_edge(new_node, node_adjacent)

        # elimino tutti i vecchi archi
        self.remove_edges_connected_to_node(nodeB)

        # elimino il nodoB
        self.graph.remove_node(nodeB)

    def remove_edges_connected_to_node(self, node):
        edges_connected = list(self.graph.edges(node))

        for edge in edges_connected:
            self.graph.remove_edge(*edge)

    def remove_pad_to_img(self, img):
        return img[constants.PAD : -constants.PAD, constants.PAD : -constants.PAD]

    def get_ID_from_edge(self, edge):
        try:
            ID = self.graph[edge[0]][edge[1]]["ID"]
        except KeyError:
            ID = self.graph[edge[1]][edge[0]]["ID"]
        return ID

    def get_nodes_inside_coords(self, coords_quadrilateral):
        dict_coord = nx.get_node_attributes(self.graph, "coordinates")
        coord_nodes_inside = []
        nodes_inside = []
        for node in self.graph.nodes():
            coord_node = dict_coord[node]
            if is_in_quadrilateral(coords_quadrilateral, coord_node):
                nodes_inside.append(node)

        return nodes_inside

    def find_source_nodes(self, data_edges):
        def get_coord_quadrilateral_in_point(start_cell, height, width):
            lowleft_cell = (start_cell[0] + height, start_cell[1])
            highright_cell = (start_cell[0], start_cell[1] + width)
            lowright_cell = (start_cell[0] + height, start_cell[1] + width)

            return [(start_cell[0], start_cell[1]), highright_cell, lowright_cell, lowleft_cell]

        img_height = 584 - 120
        img_width = 565

        start_cell = [200, 0]
        height = 70
        width = 15
        # height = 20
        # width = 20
        max_big_edges_inside = -1
        source_nodes = []
        # scandaglia tutti i rettangoli fino a trovare quello massimo
        while start_cell[0] + height < img_height:
            start_cell[1] = 0
            while start_cell[1] + width < img_width:
                coords = get_coord_quadrilateral_in_point(start_cell, height, width)
                # nodes = self.get_nodes_inside_coords(coords)
                big_edges = self.get_big_edges_inside_coords(coords, data_edges)

                if len(big_edges) > max_big_edges_inside:
                    max_big_edges_inside = len(big_edges)
                    # source_nodes = nodes
                    source_nodes = self.get_nodes_inside_coords(coords)
                    # show_cells(coords + coord_nodes)

                start_cell[1] += width

            start_cell[0] += height

        self.source_nodes = source_nodes

    def get_big_edges_inside_coords(self, coords, data_edges):
        nodes = self.get_nodes_inside_coords(coords)
        big_edges_IDs = set()
        for node in nodes:
            connected_edges = self.graph.edges(node)
            for edge in connected_edges:
                idx = self.get_ID_from_edge(edge)
                data_edge = data_edges[idx]
                if data_edge.get_diameter() >= 7:
                    big_edges_IDs.add(data_edge.ID)

        return big_edges_IDs

    def get_minimum_distance_from_nodes_BFS(self, start, end_nodes):
        to_explore_next_level = set()
        to_explore_next_level.add(start)
        distance = -1
        found = False
        visited = set()
        while to_explore_next_level and not found:
            distance += 1
            to_explore_this_level = to_explore_next_level.copy()
            to_explore_next_level = set()
            for node in to_explore_this_level:
                if node in end_nodes:
                    found = True
                    break

                visited.add(node)
                adjacents = self.get_nodes_adjacent_to_node(node)
                for adj_node in adjacents:
                    if adj_node not in visited:
                        to_explore_next_level.add(adj_node)

        if not found:
            distance = -1

        return distance

    def get_minimum_distance_from_nodes_BFS_with_visited(self, start, end_nodes, visited):
        to_explore_next_level = set()
        to_explore_next_level.add(start)
        distance = -1
        found = False
        while to_explore_next_level and not found:
            distance += 1
            to_explore_this_level = to_explore_next_level.copy()
            to_explore_next_level = set()
            for node in to_explore_this_level:
                if node in end_nodes:
                    found = True
                    break

                visited.add(node)
                adjacents = self.get_nodes_adjacent_to_node(node)
                for adj_node in adjacents:
                    if adj_node not in visited:
                        to_explore_next_level.add(adj_node)

        if not found:
            distance = -1

        return distance

    def get_minimum_distance_from_any_source_node_BFS(self, start):
        return self.get_minimum_distance_from_nodes_BFS(start, self.source_nodes)

    def get_min_distance_from_node_tmp(self, start, end_node):
        to_explore_next_level = set()
        to_explore_next_level.add(start)
        distance = -1
        found = False
        visited = set()
        while to_explore_next_level and not found:
            distance += 1
            to_explore_this_level = to_explore_next_level.copy()
            to_explore_next_level = set()
            for node in to_explore_this_level:
                visited.add(node)
                adjacents = self.get_nodes_adjacent_to_node(node)

                if distance == 0:
                    adjacents.remove(end_node)

                for adj_node in adjacents:
                    if adj_node == end_node:
                        found = True
                        break

                    if adj_node not in visited:
                        to_explore_next_level.add(adj_node)

        if not found:
            distance = -1

        return distance

    def get_minimum_euclidean_distance_from_any_source_node(self, start):
        dict_coord = nx.get_node_attributes(self.graph, "coordinates")
        start_coord = dict_coord[start]

        min_distance = euclide_distance(start_coord, dict_coord[self.source_nodes[0]])
        for idx in range(1, len(self.source_nodes)):
            distance = euclide_distance(start_coord, dict_coord[self.source_nodes[idx]])
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_nodes_with_degree(self, degree):
        nodes = set()
        for node in self.graph.nodes:
            edges = self.graph.edges(node)
            if len(edges) == degree:
                nodes.add(node)

        return nodes

    def get_ID_edges_connected_to_node(self, node):
        IDs = set()
        for edge in self.graph.edges(node):
            IDs.add(self.graph[edge[0]][edge[1]]["ID"])

        return IDs

    def get_ID_edges_connected_to_node_with_distance_from_any_source(self, distance):
        nodes = self.get_nodes_with_distance_from_any_source_nodes(distance)

        edges = set()
        # prendo ogni arco di ogni nodo
        for node in nodes:
            edges = edges.union(self.get_ID_edges_connected_to_node(node))

        return edges

    def get_new_ID_node(self):
        max_ID = -1
        for node in self.graph.nodes:
            if node > max_ID:
                max_ID = node
        return max_ID + 1

    def resolve_structural_node_4(self, cross_node, first_couple_data_edge, second_couple_data_edge, data_nodes, data_edges):
        def assign_orientation_edge(edge, node1, node2):
            # if edge.orientation == Orientation.FORWARD:
            #     edge.orientation = Orientation.UNDEFINED
            # else:
            #     edge.assign_forward_orientation(node1, node2)

            edge.assign_forward_orientation(node1, node2)

        def assign_orientation_couple(couple, source_node):
            e1 = couple[0]
            e2 = couple[1]

            common_node = e1.get_common_node_with(e2)
            outer_1 = e1.n2 if common_node == e1.n1 else e1.n1
            outer_2 = e2.n2 if common_node == e2.n1 else e2.n1
            assert common_node != outer_1 and common_node != outer_2 and outer_1 != outer_2, "due nodi uguali!!!"

            dis_common = self.get_minimum_distance_from_nodes_BFS(common_node, [source_node])
            dis_outer_1 = self.get_minimum_distance_from_nodes_BFS(outer_1, [source_node])
            dis_outer_2 = self.get_minimum_distance_from_nodes_BFS(outer_2, [source_node])

            # se non è possibile raggiungere la source, uso la distanza euclidea
            if dis_common == -1 or dis_outer_1 == -1 or dis_outer_2 == -1:
                assert dis_common == dis_outer_1 and dis_common == dis_outer_2, "non sono uguali!!!"
                dict_coord = nx.get_node_attributes(self.graph, "coordinates")

                dis_common = euclide_distance(dict_coord[common_node], dict_coord[source_node])
                dis_outer_1 = euclide_distance(dict_coord[outer_1], dict_coord[source_node])
                dis_outer_2 = euclide_distance(dict_coord[outer_2], dict_coord[source_node])

            if dis_outer_1 < dis_common or dis_outer_2 > dis_common:
                assign_orientation_edge(e1, outer_1, common_node)
                assign_orientation_edge(e2, common_node, outer_2)
            elif dis_outer_1 > dis_common or dis_outer_2 < dis_common:
                assign_orientation_edge(e1, common_node, outer_1)
                assign_orientation_edge(e2, outer_2, common_node)

            assert dis_outer_1 != dis_common or dis_outer_2 != dis_common, "distanze tutte e 3 uguali!!!"

        dict_coord = nx.get_node_attributes(self.graph, "coordinates")
        # rimuovo la prima coppia e la connetto al nuovo nodo

        # rimuovo i due archi
        self.graph.remove_edge(first_couple_data_edge[0].n1, first_couple_data_edge[0].n2)
        self.graph.remove_edge(first_couple_data_edge[1].n1, first_couple_data_edge[1].n2)

        # creo il nuovo nodo
        duplicate_node = self.get_new_ID_node()
        self.graph.add_node(duplicate_node, coordinates=dict_coord[cross_node])
        data_nodes[duplicate_node] = Node(ID=duplicate_node, id_duplicate_node_4_degree=cross_node)
        data_nodes[cross_node].id_duplicate_node_4_degree = duplicate_node

        # aggiorno il nuovo nodo ai due archi
        if first_couple_data_edge[0].n1 == cross_node:
            first_couple_data_edge[0].n1 = duplicate_node
        else:
            first_couple_data_edge[0].n2 = duplicate_node

        if first_couple_data_edge[1].n1 == cross_node:
            first_couple_data_edge[1].n1 = duplicate_node
        else:
            first_couple_data_edge[1].n2 = duplicate_node

        # aggiungo i due archi
        self.graph.add_edge(first_couple_data_edge[0].n1, first_couple_data_edge[0].n2, ID=first_couple_data_edge[0].ID)
        self.graph.add_edge(first_couple_data_edge[1].n1, first_couple_data_edge[1].n2, ID=first_couple_data_edge[1].ID)

        # aggiorno l'orientamento
        source_node = self.source_nodes[0]
        assign_orientation_couple(first_couple_data_edge, source_node)
        assign_orientation_couple(second_couple_data_edge, source_node)

        DG = self.get_DG_from_graph(data_edges)
        # if cross_node == 129 and duplicate_node == 153:
        #     print(DG.out_edges(duplicate_node))
        #     print(DG.in_edges(duplicate_node))
        # elif cross_node == 153 and duplicate_node == 129:
        #     print(DG.out_edges(cross_node))
        #     print(DG.in_edges(cross_node))

        assert len(DG.out_edges(cross_node)) == 1, "piu di un arco uscente per il cross node"
        assert len(DG.in_edges(cross_node)) == 1, "piu di un arco entrante per il cross node"

        assert len(DG.out_edges(duplicate_node)) == 1, "piu di un arco uscente per il duplicate node"
        assert len(DG.in_edges(duplicate_node)) == 1, "piu di un arco entrante per il duplicate node"

    def get_sub_tree(self, start):
        forbidden = set()
        for snode in self.source_nodes:
            forbidden.add(snode)

        to_explore_next_level = set()
        to_explore_next_level.add(start)
        visited = set()
        while to_explore_next_level:
            to_explore_this_level = to_explore_next_level.copy()
            to_explore_next_level = set()
            for node in to_explore_this_level:
                visited.add(node)
                adjacents = self.get_nodes_adjacent_to_node(node)
                for adj_node in adjacents:
                    if adj_node not in forbidden:
                        if adj_node not in visited:
                            to_explore_next_level.add(adj_node)

        return visited

    def get_edges_sub_tree(self, start):
        sub_tree = self.get_sub_tree(start)

        sub_tree_edges = set()
        for n in sub_tree:
            sub_tree_edges = sub_tree_edges.union(self.get_ID_edges_connected_to_node(n))

        return sub_tree_edges

    def get_min_distance_from_node_in_tree(self, start, end_node, tree, visited_):
        # visited = set()
        visited = visited_.copy()
        to_explore_next_level = set()
        to_explore_next_level.add(start)
        distance = -1
        found = False
        while to_explore_next_level and not found:
            distance += 1
            to_explore_this_level = to_explore_next_level.copy()
            to_explore_next_level = set()
            for node in to_explore_this_level:
                visited.add(node)
                adjacents = self.get_nodes_adjacent_to_node_in_tree(node, tree)

                if distance == 0:
                    adjacents.remove(end_node)

                for adj_node in adjacents:
                    if adj_node == end_node:
                        found = True
                        break
                    if adj_node not in visited:
                        to_explore_next_level.add(adj_node)

        if not found:
            distance = -1

        return distance

    def separate_trees(self, start, tree, data_edges, img):
        def is_moving_away(start, node, adj_node, d=0):
            dict_coord = nx.get_node_attributes(self.graph, "coordinates")
            start_coord = dict_coord[start]
            node_coord = dict_coord[node]
            adj_coord = dict_coord[adj_node]

            # per ogni dimensione
            # sottraggo il valore di start sia a nodo che ad adjnode
            # prendo il valore assoluto della differenza
            # faccio la differenza tra adiacente e nodo corrente
            # se la differenza è positiva mi sto allontanando altrimenti avvicinando
            return abs(adj_coord[d] - start_coord[d]) - abs(node_coord[d] - start_coord[d]) >= 0

        def accept_node_in_tree(accepted_tree, rejected_tree, to_visit_next_round, father, node):
            for f, n in accepted_tree:
                if n == node:
                    return

            for n in rejected_tree:
                if n == node:
                    return

            to_visit_next_round.add((father, node))
            accepted_tree.add((father, node))

            self.draw_node(node, green, img)

        def reject_node_in_tree(rejected_tree, node, tree):
            if len(self.get_nodes_adjacent_to_node_in_tree(node, tree)) != 4:
                rejected_tree.add(node)

                self.draw_node(node, yellow, img)

        def get_father(node, tree):
            for father, node_ in tree:
                if node == node_:
                    return father

            return None

        def get_diameter_weighted_ancestors(node, accepted_tree, n_ancestors, data_edges):
            total_pixels = 0
            total_length = 0
            i = 0
            while i < n_ancestors:
                father = get_father(node, accepted_tree)

                if father == node:
                    break

                total_pixels += len(data_edges[self.get_ID_from_edge((father, node))].coord_pixels)
                total_length += data_edges[self.get_ID_from_edge((father, node))].get_length()

                node = father

                i += 1

            weighted_diameter = 0
            if total_length != 0:
                weighted_diameter = total_pixels / total_length
            return weighted_diameter

        # if start != 199:
        #     return

        green = [0, 255, 0]
        yellow = [255, 255, 0]
        white = [255, 255, 255]
        cyan = [0, 255, 255]

        # for n in tree:
        #     self.draw_node(n, green, img)

        self.draw_node(start, white, img)

        print("\nNUOVA IMMAGINE")
        accepted_tree = set()
        accepted_tree.add((start, start))

        rejected_tree = set()

        to_visit_next_round = set()
        # i nodi sono coppie (nodo_padre, nodo_figlio)
        to_visit_next_round.add((start, start))

        visited = set()
        while to_visit_next_round:
            show_image(img)
            to_visit_now = to_visit_next_round.copy()
            to_visit_next_round = set()
            for current_father_child in to_visit_now:
                print(f"\nCOPPIA FATHER CHILD {current_father_child}")
                father = current_father_child[0]
                node = current_father_child[1]

                self.draw_node(node, cyan, img)

                print("NODO", node)
                visited.add(node)
                adj_nodes = list(self.get_nodes_adjacent_to_node_in_tree(node, tree))
                # rimuovo il padre dagli adiacenti
                if father != node:
                    adj_nodes.remove(father)

                if len(adj_nodes) == 1:
                    # se è solo lui lo prendo
                    adj_node = adj_nodes[0]

                    accept_node_in_tree(accepted_tree, rejected_tree, to_visit_next_round, node, adj_node)
                elif len(adj_nodes) == 2:
                    # ci sono due biforcazioni, devo capire se una è falsa

                    # se uno dei due nodi è un nodo da 4, prendo per buone entrambe le biforcazioni
                    if len(self.get_nodes_adjacent_to_node_in_tree(adj_nodes[0], tree)) == 4 or len(self.get_nodes_adjacent_to_node_in_tree(adj_nodes[1], tree)) == 4:
                        print("UN NODO È DA 4!!!!")

                        accept_node_in_tree(accepted_tree, rejected_tree, to_visit_next_round, node, adj_nodes[0])
                        accept_node_in_tree(accepted_tree, rejected_tree, to_visit_next_round, node, adj_nodes[1])

                        continue

                    # posso ritornare al nodo corrente da almeno una delle due biforcazioni?
                    if (
                        self.get_min_distance_from_node_in_tree(adj_nodes[0], node, tree, visited) != -1
                        or self.get_min_distance_from_node_in_tree(adj_nodes[1], node, tree, visited) != -1
                    ):
                        print("BIFORCAZIONE POSSO TORNARE!!!")
                        # una delle due è falsa perché posso tornare indietro
                        # scelgo quella che ha il diametro più simile all'arco precedente
                        edge_father_child = data_edges[self.get_ID_from_edge((father, node))]
                        edge_child_adj_node_0 = data_edges[self.get_ID_from_edge((node, adj_nodes[0]))]
                        edge_child_adj_node_1 = data_edges[self.get_ID_from_edge((node, adj_nodes[1]))]

                        diff_0 = abs(edge_father_child.get_diameter() - edge_child_adj_node_0.get_diameter())
                        diff_1 = abs(edge_father_child.get_diameter() - edge_child_adj_node_1.get_diameter())
                        if diff_0 < diff_1:
                            good_adj = adj_nodes[0]
                        else:
                            good_adj = adj_nodes[1]

                        print(f"DIFF DIAM ARCO ADIACENTE 0 {diff_0}")
                        print(f"DIFF DIAM ARCO ADIACENTE 1 {diff_1}")

                        print(f"DIAMETRO PADRE FIGLIO {edge_father_child.get_diameter()}")
                        print(f"DIAMETRO 2 ANCESTORI {get_diameter_weighted_ancestors(node, accepted_tree, 2, data_edges)}")

                        diff_a_0 = abs(get_diameter_weighted_ancestors(node, accepted_tree, 2, data_edges) - edge_child_adj_node_0.get_diameter())
                        diff_a_1 = abs(get_diameter_weighted_ancestors(node, accepted_tree, 2, data_edges) - edge_child_adj_node_1.get_diameter())

                        print(f"DIFF DIAM ARCO ADIACENTE 0 {diff_a_0}")
                        print(f"DIFF DIAM ARCO ADIACENTE 1 {diff_a_1}")

                        accept_node_in_tree(accepted_tree, rejected_tree, to_visit_next_round, node, good_adj)

                        adj_nodes.remove(good_adj)

                        reject_node_in_tree(rejected_tree, adj_nodes[0], tree)
                        continue

                    print("NESSUNA BIFORCAZIONE TORNA!!!")

                    accept_node_in_tree(accepted_tree, rejected_tree, to_visit_next_round, node, adj_nodes[0])
                    accept_node_in_tree(accepted_tree, rejected_tree, to_visit_next_round, node, adj_nodes[1])

                elif len(adj_nodes) == 3:
                    # è un incrocio da 4, 2 biforcazioni non sono buone, 1 sì. Prendo quella con l'angolo più alto con l'arco precedente
                    edge_father_child = data_edges[self.get_ID_from_edge((father, node))]
                    edge_child_adj_node_0 = data_edges[self.get_ID_from_edge((node, adj_nodes[0]))]
                    edge_child_adj_node_1 = data_edges[self.get_ID_from_edge((node, adj_nodes[1]))]
                    edge_child_adj_node_2 = data_edges[self.get_ID_from_edge((node, adj_nodes[2]))]
                    angles = [
                        edge_father_child.get_angle_with(edge_child_adj_node_0),
                        edge_father_child.get_angle_with(edge_child_adj_node_1),
                        edge_father_child.get_angle_with(edge_child_adj_node_2),
                    ]
                    idx_max = angles.index(max(angles))

                    accept_node_in_tree(accepted_tree, rejected_tree, to_visit_next_round, node, adj_nodes[idx_max])

                    del adj_nodes[idx_max]
                    reject_node_in_tree(rejected_tree, adj_nodes[0], tree)
                    reject_node_in_tree(rejected_tree, adj_nodes[1], tree)

                    continue

                self.draw_node(node, green, img)

                # for adj_node in adj_nodes:
                #     show_image(img)

                #     print("ADIACENTE", adj_node)
                #     if adj_node not in visited:
                #         # se è solo lui lo prendo
                #         if len(adj_nodes) == 1:
                #             to_visit_next_round.add((node, adj_node))
                #             tree1.add(adj_node)

                #             self.draw_node(adj_node, green, img)
                #             continue

                #         # se è un dead end lo prendo
                #         if len(self.get_nodes_adjacent_to_node_in_tree(adj_node, tree)) == 1:
                #             to_visit_next_round.add((node, adj_node))
                #             tree1.add(adj_node)

                #             self.draw_node(adj_node, green, img)
                #             continue

                #         # se questo adiacente è un nodo da 4 lo prendo
                #         if len(self.get_nodes_adjacent_to_node_in_tree(adj_node, tree)) == 4:
                #             to_visit_next_round.add((node, adj_node))
                #             tree1.add(adj_node)

                #             self.draw_node(adj_node, green, img)
                #             continue

                #         # se ci sono due biforcazioni, voglio capire se una delle due è finta
                #         if len(adj_nodes) == 2:
                #             # TODO: se da una biforcazione si può raggiungere il nodo di partenza (c'è un ciclo)
                #             # significa che una delle due biforcazioni è falsa
                #             # scelgo quella che ha il diametro simile all'arco precedente perché essendo una biforcazione finta significa che
                #             # il diametro non deve cambiare

                #         # se si avvicina in entrambe le direzioni rispetto allo start allora lo rifiuto
                #         # TODO: provare rispetto al padre anziché allo start
                #         if not is_moving_away(start, node, adj_node, 0) and not is_moving_away(start, node, adj_node, 1):
                #             print("SI AVVICINA IN ENTRAMBE LE DIREZIONI!!!!")

                #             self.draw_node(adj_node, yellow, img)
                #             continue

                #         # TODO: si può fare qualcosa in base a quello precedente:
                #         # cioè se si allontana o avvicina rispetto a quello precedente
                #         # in un certo senso se negli ultimi due sto aumentando ad esempio l'altezza e poi con il terzo la diminuisco
                #         # c'è un errore perché sto tornando indietro, so rifacendo la stessa strada che ho percorso e non è possibile

                #         # TODO: i blu mi sembrano più grossi rispetto ai rossi

                #         # TODO: posso inserire il nonno oltre il padre, fare la media tra nonno-padre e padre-figlio per vedere il diametro
                #         # se il diametro è superiore al diametro medio dei miei predecessori allora non lo prendo
                #         # posso fare qualcosa tipo media mobile ponderata?

                #         # TODO: eliminare archi piccoli per vedere pezzi grossi e lunghi una volta individuato i tronconi principali
                #         # poi si vanno a ragionare sulle ramificazioni piccole
                #         # 1 ragiona eliminando tutti gli archi piccoli
                #         # 2 identifica i troncone di vena e di arteria
                #         # 3 vedi gli archi piccoli se sono attaccati alle vene o alle arterie
                #         # 4 le vene sono più grosse e più "dritte" meno tortuose e si ramificano con meno frequenza
                #         # 5 le arterie sono più piccole, sono più tortuose, meno dritte e si ramificano con più frequenza (soprattutto vicino la sorgente)
                #         # 6 le arterie si ramificano più spesso quando sono vicino alla source
                #         # 7 le vene si ramificano meno spesso quando sono vicine la source
                #         # 8 dato che le vene sono più grosse, immagino che la rete riesca ad identificare più vene che non arterie
                #         # il grado di tortuosità lo posso esprimere come rapporto tra lunghezza skel e lunghezza euclidea: maggiore di 1 significa che è tortuoso
                #         # se il grado di tortuosità differisce molto da quello medio di tutti gli ancestori allora magari è un altro vaso

                #         if father != node:
                #             edge_father_child = data_edges[self.get_ID_from_edge((father, node))]
                #             edge_child_adj_node = data_edges[self.get_ID_from_edge((node, adj_node))]

                #             diam_father_node = edge_father_child.get_diameter()
                #             diam_node_adj_node = edge_child_adj_node.get_diameter()

                #             print(f"PADRE {father} CHILD {node} ADIACENTE {adj_node}")
                #             print(f"DIAM ARCO TRA PADRE E FIGLIO {diam_father_node}")
                #             print(f"DIAM ARCO TRA CHILD E ADIACENTE {diam_node_adj_node}")
                #             print(f"ANGOLO TRA ARCO PADRE-FIGLIO E ARCO FIGLIO-ADIACENTE {edge_father_child.get_angle_with(edge_child_adj_node)}")
                #             show_angle_edges_tmp([edge_father_child, edge_child_adj_node])

                #             # se devia di poco rispetto all'arco precedente allora lo prendo
                #             if edge_father_child.get_angle_with(edge_child_adj_node) > 157:
                #                 to_visit_next_round.add((node, adj_node))
                #                 tree1.add(adj_node)

                #                 self.draw_node(adj_node, green, img)
                #                 continue

                #             # se l'arco tra il nodo adiacente e il nodo corrente ha un diametro superiore dell'arco
                #             # tra il nodo corrente e suo padre, allora lo rifiuto

                #             if diam_node_adj_node > diam_father_node:
                #                 show_single_edge(data_edges[self.get_ID_from_edge((father, node))])
                #                 show_single_edge(data_edges[self.get_ID_from_edge((node, adj_node))])
                #                 print("ARCO PIU GRANDE RISPETTO A QUELLO TRA FATHER E CHILD")

                #                 self.draw_node(adj_node, yellow, img)
                #                 continue

                #         to_visit_next_round.add((node, adj_node))
                #         tree1.add(adj_node)

                #         self.draw_node(adj_node, green, img)

        for father, node in accepted_tree:
            self.draw_node(node, white, img)
        print("\nRISULTATO FINALE!!\n")
        show_image(img)

    def get_nodes_adjacent_to_node_in_tree(self, node, tree):
        adj_nodes = set()
        for n in tree:
            if self.graph.has_edge(n, node) or self.graph.has_edge(node, n):
                adj_nodes.add(n)

        return adj_nodes

    def draw_and_show(self, tree, img):
        dict_coord = nx.get_node_attributes(self.graph, "coordinates")
        color = [0, 255, 0]
        for node in tree:
            coord = dict_coord[node]
            colorize_cells(color, get_matrix_cells(coord, 2), img)

        show_image(img)

    def draw_node(self, node, color, img):
        dict_coord = nx.get_node_attributes(self.graph, "coordinates")
        coord = dict_coord[node]
        colorize_cells(color, get_matrix_cells(coord, 2), img)

        # show_image(img)

    def get_DG_from_graph(self, data_edges):
        # ottengo un DG a partire dagli archi
        DG = nx.DiGraph(directed=True)
        for idx in data_edges:
            edge = data_edges[idx]
            # la rotazione di 90° la ottengo trasformando (x,y) in (y, altezza - x)
            DG.add_node(edge.n1, coordinates=(edge.n1_coord[1], 584 - edge.n1_coord[0]))
            DG.add_node(edge.n2, coordinates=(edge.n2_coord[1], 584 - edge.n2_coord[0]))
            DG.add_edge(edge.n1, edge.n2, ID=idx)
            # if edge.orientation == Orientation.FORWARD:
            #     DG.add_edge(edge.n1, edge.n2)
            if edge.orientation == Orientation.UNDEFINED:
                DG.add_edge(edge.n2, edge.n1, ID=idx)
        return DG

    def get_degree_node(self, node):
        return len(self.get_nodes_adjacent_to_node(node))

    def get_nodes_with_data(self):
        nodes = {}
        for node in self.graph.nodes:
            idx = node
            nodes[idx] = Node(idx)

        return nodes

    def get_edges_with_data(self):
        edges = {}
        dict_nodes_coordinates = nx.get_node_attributes(self.graph, "coordinates")
        for idx, edge in enumerate(self.graph.edges):
            n1_coord = dict_nodes_coordinates[edge[0]]
            n2_coord = dict_nodes_coordinates[edge[1]]

            edges[idx] = Edge(self.graph, idx, edge[0], edge[1], n1_coord, n2_coord)

        return edges
