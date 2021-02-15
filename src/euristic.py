import networkx as nx
from matplotlib import pyplot as plt
from edge import Orientation
from utils_math import euclide_distance, angle_vectors
from utils_image import show_directed_graph, show_edges
import constants
from constants import TypeClassification
from node import Node


class Euristic(object):
    def __init__(self, data_edges, data_nodes, graph_builder):
        self.data_edges = data_edges
        self.data_nodes = data_nodes
        self.graph_builder = graph_builder

    def apply_corrections(self):
        self.resolve_4_cross_over_turned_3()
        self.resolve_nodes_4_degree()
        self.resolve_periferical_triplets_simple_EU()
        self.resolve_surrounded_edges()
        self.overwrite_euristic(prob=0.65)

    def assign_orientation(self):
        # parto da una source node qualsiasi
        source_node = self.graph_builder.source_nodes[0]

        # faccio una BFS a partire da questa source_node per assegnare l'orientamento
        to_explore_next_level = set()
        to_explore_next_level.add(source_node)
        visited = set()
        while to_explore_next_level:
            to_explore_this_level = to_explore_next_level.copy()
            to_explore_next_level = set()
            for node in to_explore_this_level:
                visited.add(node)
                adjacents = self.graph_builder.get_nodes_adjacent_to_node(node)
                for adj_node in adjacents:
                    if adj_node not in visited:
                        id_edge = self.graph_builder.get_ID_from_edge((node, adj_node))
                        self.data_edges[id_edge].assign_forward_orientation(
                            node, adj_node
                        )

                        to_explore_next_level.add(adj_node)

        DG = self.graph_builder.get_DG_from_graph(self.data_edges)
        # show_directed_graph(DG)
        for node in DG.nodes():
            edges_adj = DG.in_edges(node)
            if len(edges_adj) >= 2:
                for e in edges_adj:
                    idx = DG[e[0]][e[1]]["ID"]
                    self.data_edges[idx].orientation = Orientation.UNDEFINED

        # correggo l'orientamento dei nodi
        # coord_source = self.graph_builder.get_coordinate_node(source_node)
        # # coord_source = self.graph_builder.coordinates_optic_disk
        # print(coord_source)
        # for edge in self.data_edges.values():
        #     if edge.orientation == Orientation.FORWARD:
        #         # cerco gli archi per cui il secondo nodo si avvicina alla source anziché allontanarsi
        #         n1_dis = self.graph_builder.get_minimum_distance_from_nodes_BFS(edge.n1, [source_node])
        #         n2_dis = self.graph_builder.get_minimum_distance_from_nodes_BFS(edge.n2, [source_node])
        #         if n2_dis < n1_dis:
        #             edge.assign_forward_orientation(edge.n2, edge.n1)
        #         elif n2_dis == n1_dis:
        #             # edge.orientation = Orientation.UNDEFINED
        #             if euclide_distance(coord_source, edge.n2_coord) < euclide_distance(coord_source, edge.n1_coord):
        #                 edge.assign_forward_orientation(edge.n2, edge.n1)

        # cerco di capire se il source è ai lati oppure al centro
        # if coord_source[1] < 150 or coord_source[1] > 400:
        #     if abs(edge.n2_coord[1] - coord_source[1]) < abs(edge.n1_coord[1] - coord_source[1]):
        #         # se c'è un arco orientato verso questo secondo nodo allora cambio il verso di questo arco
        #         for edge2 in self.data_edges.values():
        #             if edge != edge2:
        #                 if edge2.n2 == edge.n2:
        #                     edge.assign_forward_orientation(edge.n2, edge.n1)
        #                     break
        # else:
        #     if abs(edge.n2_coord[0] - coord_source[0]) < abs(edge.n1_coord[0] - coord_source[0]):
        #         # se c'è un arco orientato verso questo secondo nodo allora cambio il verso di questo arco
        #         for edge2 in self.data_edges.values():
        #             if edge != edge2:
        #                 if edge2.n2 == edge.n2:
        #                     edge.assign_forward_orientation(edge.n2, edge.n1)
        #                     break

        # if abs(edge.n2_coord[0] - coord_source[0]) < abs(edge.n1_coord[0] - coord_source[0]):
        #     # se c'è un arco orientato verso questo secondo nodo allora cambio il verso di questo arco
        #     for edge2 in self.data_edges.values():
        #         if edge != edge2:
        #             if edge2.n2 == edge.n2:
        #                 edge.assign_forward_orientation(edge.n2, edge.n1)
        #                 break

        # if euclide_distance(coord_source, edge.n2_coord) < euclide_distance(coord_source, edge.n1_coord):
        #     for edge2 in self.data_edges.values():
        #         if edge != edge2:
        #             if edge2.n2 == edge.n2:
        #                 edge.assign_forward_orientation(edge.n2, edge.n1)
        #                 break

        # DG = self.graph_builder.get_DG_from_graph(self.data_edges)
        # show_directed_graph(DG)

    def resolve_nodes_4_degree(self):
        def get_edge_to_couple_with_max_sum_of_angles(edges):
            max_sum_angles = -1
            edge_to_couple = None
            for e in edges:
                couple = [edge0, e]

                other_couple = edges.copy()
                other_couple.remove(e)

                sum_angles = couple[0].get_angle_with(couple[1]) + other_couple[
                    0
                ].get_angle_with(other_couple[1])
                if sum_angles > max_sum_angles:
                    max_sum_angles = sum_angles
                    edge_to_couple = e

            return edge_to_couple

        def married(couple):
            return couple[0].class_EU == couple[1].class_EU

        def flipped_class(class_):
            if class_ == constants.ARTERY_CLASS:
                return constants.VEIN_CLASS

            if class_ == constants.VEIN_CLASS:
                return constants.ARTERY_CLASS

        def can_reach_source(couple, cross_node):
            # inserisco i nodi da cui far partire la ricerca: escludo il nodo centrale
            nodes = []
            if couple[0].n1 != cross_node:
                nodes.append(couple[0].n1)
            else:
                nodes.append(couple[0].n2)

            if couple[1].n1 != cross_node:
                nodes.append(couple[1].n1)
            else:
                nodes.append(couple[1].n2)

            visited = set()
            # faccio in modo che non si passa dal nodo centrale per raggiungere la source
            visited.add(cross_node)
            for node in nodes:
                if (
                    self.graph_builder.get_minimum_distance_from_nodes_BFS_with_visited(
                        node, self.graph_builder.source_nodes, visited
                    )
                    != -1
                ):
                    return True

            return False

        def fake_crossover(node, first_couple, second_couple):
            # se l'angolo formato dalla almeno 1 coppia è <= 120° (considerando i nodi estremi) allora è un falso incrocio
            e1 = second_couple[0]
            e2 = second_couple[1]
            common_node = e1.get_common_node_with(e2)
            outer_1 = e1.n2 if common_node == e1.n1 else e1.n1
            outer_2 = e2.n2 if common_node == e2.n1 else e2.n1

            lineA = [e1.get_coord_node(common_node), e1.get_coord_node(outer_1)]
            lineB = [e2.get_coord_node(common_node), e2.get_coord_node(outer_2)]
            if angle_vectors(lineA, lineB) <= 90:
                return True

            e1 = first_couple[0]
            e2 = first_couple[1]
            common_node = e1.get_common_node_with(e2)
            outer_1 = e1.n2 if common_node == e1.n1 else e1.n1
            outer_2 = e2.n2 if common_node == e2.n1 else e2.n1

            lineA = [e1.get_coord_node(common_node), e1.get_coord_node(outer_1)]
            lineB = [e2.get_coord_node(common_node), e2.get_coord_node(outer_2)]
            if angle_vectors(lineA, lineB) <= 90:
                return True

            # sfrutto l'orientamento: se il nodo ha esattamente un entrante e 3 uscenti è un fake crossover
            # DG = self.graph_builder.get_DG_from_graph(self.data_edges)
            # in_edges = list(DG.in_edges(node))
            # out_edges = list(DG.out_edges(node))
            # if len(in_edges) == 1 and len(out_edges) == 3:
            #     return True

            # Se una delle due coppie è fatta da due nodi terminali allora è un fake node
            # if has_terminal_nodes(first_couple, node) or has_terminal_nodes(second_couple, node):
            #     return True

            # devo poter raggiungere una source da entrambe le parti
            # can_reach1 = can_reach_source(first_couple, node)
            # can_reach2 = can_reach_source(second_couple, node)
            # if can_reach1 or can_reach2:
            #     if not can_reach_source(first_couple, node) or not can_reach_source(second_couple, node):
            #         return True
            # self.data_nodes[node].fake_cross_over = True
            # continue

            return False

        def has_terminal_nodes(couple, central_node):
            other_node_1 = (
                couple[0].n1 if central_node == couple[0].n2 else couple[0].n2
            )
            other_node_2 = (
                couple[1].n1 if central_node == couple[1].n2 else couple[1].n2
            )

            if (
                self.graph_builder.get_degree_node(other_node_1) == 1
                and self.graph_builder.get_degree_node(other_node_2) == 1
            ):
                return True

            return False

        nodes_4_degree = self.graph_builder.get_nodes_with_degree(4)
        for node in nodes_4_degree:
            # skippo il nodo se è un source
            if node in self.graph_builder.source_nodes:
                continue

            # skippo se il nodo è molto vicino ad una qualsiasi source
            # min_dis = self.graph_builder.get_minimum_distance_from_any_source_node_BFS(node)
            # if min_dis != -1 and min_dis <= 2:
            #     continue

            if (
                self.graph_builder.get_minimum_euclidean_distance_from_any_source_node(
                    node
                )
                <= 25
            ):
                continue

            # ottengo i 4 archi
            id_edges = self.graph_builder.get_ID_edges_connected_to_node(node)
            four_edges = []
            for idx in id_edges:
                four_edges.append(self.data_edges[idx])

            # prendo un arco a caso dei 4
            edge0 = four_edges.pop(0)

            # scelgo di accoppiarmi con l'arco per cui la somma degli angoli delle coppie formate è massima
            edge_to_couple = get_edge_to_couple_with_max_sum_of_angles(four_edges)
            four_edges.remove(edge_to_couple)

            # ho determinato le due coppie di archi
            first_couple = [edge0, edge_to_couple]
            second_couple = [four_edges[0], four_edges[1]]

            # # skippo l'incrocio se ho capito che è un falso incrocio
            # if fake_crossover(node, first_couple, second_couple):
            #     self.data_nodes[node].fake_cross_over = True

            #     # assegno l'orientamento
            #     can_reach1 = can_reach_source(first_couple, node)
            #     can_reach2 = can_reach_source(second_couple, node)
            #     if can_reach1 or can_reach2:
            #         other_node_1 = first_couple[0].n1 if node == first_couple[0].n2 else first_couple[0].n2
            #         other_node_2 = first_couple[1].n1 if node == first_couple[1].n2 else first_couple[1].n2
            #         other_node_3 = second_couple[0].n1 if node == second_couple[0].n2 else second_couple[0].n2
            #         other_node_4 = second_couple[1].n1 if node == second_couple[1].n2 else second_couple[1].n2
            #         other_nodes = [other_node_1, other_node_2, other_node_3, other_node_4]
            #         self.resolve_orientation_fake_cross_over(other_nodes, node)
            #     continue
            # else:
            #     self.data_nodes[node].fake_cross_over = False

            # se la prima coppia è in accordo
            if married(first_couple):
                if not married(second_couple):
                    second_couple[0].class_EU = flipped_class(first_couple[0].class_EU)
                    second_couple[1].class_EU = flipped_class(first_couple[0].class_EU)

                    self.data_nodes[node].easy_cross_over = True
            else:
                # se la seconda coppia è in accordo
                if married(second_couple):
                    first_couple[0].class_EU = flipped_class(second_couple[0].class_EU)
                    first_couple[1].class_EU = flipped_class(second_couple[0].class_EU)

                    self.data_nodes[node].easy_cross_over = True

            # faccio il cambio strutturale del nodo da 4
            self.graph_builder.resolve_structural_node_4(
                node, first_couple, second_couple, self.data_nodes, self.data_edges
            )

        # self.resolve_fake_crossover()

        # DG = self.graph_builder.get_DG_from_graph(self.data_edges)
        # show_directed_graph(DG)

    def resolve_fake_crossover(self):
        def can_reach_source(couple, cross_node):
            # inserisco i nodi da cui far partire la ricerca: escludo il nodo centrale
            nodes = []
            if couple[0].n1 != cross_node:
                nodes.append(couple[0].n1)
            else:
                nodes.append(couple[0].n2)

            if couple[1].n1 != cross_node:
                nodes.append(couple[1].n1)
            else:
                nodes.append(couple[1].n2)

            visited = set()
            # faccio in modo che non si passa dal nodo centrale per raggiungere la source
            visited.add(cross_node)
            for node in nodes:
                if (
                    self.graph_builder.get_minimum_distance_from_nodes_BFS_with_visited(
                        node, self.graph_builder.source_nodes, visited
                    )
                    != -1
                ):
                    return True

            return False

        def fake_crossover(node, first_couple, other_node, second_couple):
            can_reach1 = can_reach_source(first_couple, node)
            can_reach2 = can_reach_source(second_couple, other_node)

            # da una coppia devo raggiungere la source e da almeno una coppia non devo raggiungerla
            if can_reach1 or can_reach2:
                if not can_reach1 or not can_reach2:
                    return True

            return False

        visited = set()
        nodes_2_degree = self.graph_builder.get_nodes_with_degree(2)
        for node in nodes_2_degree:
            if node in visited:
                continue

            other_node = self.data_nodes[node].id_duplicate_node_4_degree
            if other_node is None:
                continue
            assert (
                self.data_nodes[other_node].id_duplicate_node_4_degree == node
            ), "nodi non combaciano"

            visited.add(node)
            visited.add(other_node)

            first_couple_ids = list(
                self.graph_builder.get_ID_edges_connected_to_node(node)
            )
            first_couple = [self.data_edges[idx] for idx in first_couple_ids]

            second_couple_ids = list(
                self.graph_builder.get_ID_edges_connected_to_node(other_node)
            )
            second_couple = [self.data_edges[idx] for idx in second_couple_ids]

            if fake_crossover(node, first_couple, other_node, second_couple):
                # voglio mergiare il nodo più grande al nodo più piccolo
                if node > other_node:
                    node, other_node = other_node, node
                    first_couple, second_couple = second_couple, first_couple

                # prendo i due nodi esterni
                other_node_1 = (
                    second_couple[0].n1
                    if other_node == second_couple[0].n2
                    else second_couple[0].n2
                )
                other_node_2 = (
                    second_couple[1].n1
                    if other_node == second_couple[1].n2
                    else second_couple[1].n2
                )

                # rimuovo i due archi
                self.graph_builder.graph.remove_edge(
                    second_couple[0].n1, second_couple[0].n2
                )
                self.graph_builder.graph.remove_edge(
                    second_couple[1].n1, second_couple[1].n2
                )

                # rimuovo nodo centrale
                self.graph_builder.graph.remove_node(other_node)
                self.data_nodes[other_node] = None

                # aggancio i due nodi esterni al vecchio nodo centrale
                self.graph_builder.graph.add_edge(
                    node, other_node_1, ID=second_couple[0].ID
                )
                self.graph_builder.graph.add_edge(
                    node, other_node_2, ID=second_couple[1].ID
                )

                # aggiorno i dati dei due archi
                if second_couple[0].n1 == other_node:
                    second_couple[0].n1 = node
                elif second_couple[0].n2 == other_node:
                    second_couple[0].n2 = node

                if second_couple[1].n1 == other_node:
                    second_couple[1].n1 = node
                elif second_couple[1].n2 == other_node:
                    second_couple[1].n2 = node

                # indico che è un fake crossover
                self.data_nodes[node].id_duplicate_node_4_degree = None
                self.data_nodes[node].fake_cross_over = True

                # indico l'orientamento degli archi
                other_node_3 = (
                    first_couple[0].n1
                    if node == first_couple[0].n2
                    else first_couple[0].n2
                )
                other_node_4 = (
                    first_couple[1].n1
                    if node == first_couple[1].n2
                    else first_couple[1].n2
                )
                other_nodes = [other_node_1, other_node_2, other_node_3, other_node_4]
                self.resolve_orientation_fake_cross_over(other_nodes, node)

                # reinserisco le classi di nn per tutti e 4 gli archi
                for e in first_couple + second_couple:
                    e.class_EU = e.class_nn
            else:
                self.data_nodes[node].fake_cross_over = False

    def resolve_orientation_fake_cross_over(self, other_nodes, node):
        def get_dis_nodes(nodes, central_node):
            dis_nodes = []
            for node in nodes:
                vis_nodes = set()
                vis_nodes.add(central_node)
                dis_nodes.append(
                    self.graph_builder.get_minimum_distance_from_nodes_BFS_with_visited(
                        node, self.graph_builder.source_nodes, vis_nodes
                    )
                )

            return dis_nodes

        dis_nodes = get_dis_nodes(other_nodes, node)

        min_idx = None
        min_ = None
        for i in range(0, len(dis_nodes)):
            if dis_nodes[i] == -1:
                continue

            if min_ is None or dis_nodes[i] < min_:
                min_ = dis_nodes[i]
                min_idx = i

        in_node = other_nodes[min_idx]

        other_nodes.remove(in_node)
        for n in other_nodes:
            idx = self.graph_builder.get_ID_from_edge((n, node))
            self.data_edges[idx].assign_forward_orientation(node, n)

        idx = self.graph_builder.get_ID_from_edge((in_node, node))
        self.data_edges[idx].assign_forward_orientation(in_node, node)

        # controlli
        DG = self.graph_builder.get_DG_from_graph(self.data_edges)
        in_edges = list(DG.in_edges(node))
        assert len(in_edges) == 1, "piu di un arco entrante per il nodo!!!"

        out_edges = list(DG.out_edges(node))
        assert len(out_edges) == 3, "non ci sono 3 archi uscenti per il nodo!!!"

    def resolve_nodes_3_degree(self):
        def get_edges_sub_tree_DG(start_node, ancestor_node, max_distance):
            edges = []
            edges.append((ancestor_node, start_node))
            to_explore_next_level = set()
            to_explore_next_level.add(start_node)
            visited = set()
            visited.add(ancestor_node)
            distance = 0
            while to_explore_next_level:
                distance += 1
                if distance >= max_distance:
                    break

                to_explore_this_level = to_explore_next_level.copy()
                to_explore_next_level = set()
                for node in to_explore_this_level:
                    visited.add(node)
                    adjacents_edges = DG.out_edges(node)
                    adjacents_nodes = []
                    for adj_edge in adjacents_edges:
                        if adj_edge not in edges:
                            edges.append(adj_edge)
                        adjacents_nodes.append(adj_edge[1])

                    for adj_node in adjacents_nodes:
                        if adj_node not in visited:
                            to_explore_next_level.add(adj_node)

            return edges

        def get_major_class_edges(edges):
            e_artery_class = 0
            e_vein_class = 0
            for edge in edges:
                idx = self.graph_builder.get_ID_from_edge(edge)
                if self.data_edges[idx].class_EU == constants.ARTERY_CLASS:
                    e_artery_class += 1
                elif self.data_edges[idx].class_EU == constants.VEIN_CLASS:
                    e_vein_class += 1

            # print(f"archi arteria {e_artery_class} archi vena {e_vein_class}")

            if e_artery_class > e_vein_class:
                return constants.ARTERY_CLASS
            elif e_artery_class < e_vein_class:
                return constants.VEIN_CLASS
            else:
                return None

        def show_edges_tree(tree):
            data_edges_sub_tree = []
            for edge in tree:
                idx = self.graph_builder.get_ID_from_edge(edge)
                data_edges_sub_tree.append(self.data_edges[idx])
            show_edges(data_edges_sub_tree, type_class=TypeClassification.EU)

        def get_data_in_edge(node, idx_out_e_1, idx_out_e_2):
            adjs_e = list(self.graph_builder.get_ID_edges_connected_to_node(node))
            # print(adjs_e)
            adjs_e.remove(idx_out_e_1)
            adjs_e.remove(idx_out_e_2)
            if len(adjs_e) > 1:
                assert False, "più di 3 adiacenti"
            elif len(adjs_e) == 0:
                assert False, "solo 2 adiacenti"

            data_in_edge = self.data_edges[adjs_e[0]]
            return data_in_edge

        def same_edges(t1, t2):
            for e in t1:
                if e in t2:
                    return True
            return False

        splitted = True
        while splitted:
            splitted = False

            nodes_3_degree = self.graph_builder.get_nodes_with_degree(3)
            for node in nodes_3_degree:
                # magari ho modificato questo nodo in precedenza, ricontrollo che ha effettivamente 3 archi
                if len(self.graph_builder.get_ID_edges_connected_to_node(node)) != 3:
                    continue

                # escludo i nodi source
                if node in self.graph_builder.source_nodes:
                    continue

                # prendo quelli che hanno esattamente due archi uscenti
                DG = self.graph_builder.get_DG_from_graph(self.data_edges)
                out_edges = list(DG.out_edges(node))
                if len(out_edges) == 2:
                    # per ogni biforcazione prendo albero profondità massimo 3
                    sub_tree_edges_1 = get_edges_sub_tree_DG(out_edges[0][1], node, 3)
                    sub_tree_edges_2 = get_edges_sub_tree_DG(out_edges[1][1], node, 3)

                    # se l'albero è troppo piccolo lo skippo
                    if len(sub_tree_edges_1) < 3 or len(sub_tree_edges_2) < 3:
                        continue

                    # # se i due alberi hanno archi in comune li skippo
                    if same_edges(sub_tree_edges_1, sub_tree_edges_2):
                        continue

                    # show_edges_tree(sub_tree_edges_1)
                    # show_edges_tree(sub_tree_edges_2)

                    # il nodo è da splittare?
                    major_class_sub_tree_1 = get_major_class_edges(sub_tree_edges_1)
                    major_class_sub_tree_2 = get_major_class_edges(sub_tree_edges_2)
                    if major_class_sub_tree_1 is None or major_class_sub_tree_2 is None:
                        continue

                    if major_class_sub_tree_1 == major_class_sub_tree_2:
                        continue

                    # quale dei due devo sganciare?
                    idx_out_e_1 = self.graph_builder.get_ID_from_edge(out_edges[0])
                    idx_out_e_2 = self.graph_builder.get_ID_from_edge(out_edges[1])
                    data_in_edge = get_data_in_edge(node, idx_out_e_1, idx_out_e_2)

                    # faccio lo split solo se la rete + euristica ha assegnato una classe diversa ai due archi
                    if (
                        self.data_edges[idx_out_e_1].class_EU
                        == self.data_edges[idx_out_e_2].class_EU
                    ):
                        continue

                    # print("\n")
                    # show_directed_graph(DG)
                    # print(major_class_sub_tree_1)
                    # print(len(sub_tree_edges_1))
                    # show_edges_tree(sub_tree_edges_1)

                    # print(len(sub_tree_edges_2))
                    # print(major_class_sub_tree_2)
                    # show_edges_tree(sub_tree_edges_2)

                    # assegno ad ogni arco dei due alberi la classe maggioritaria
                    for e in sub_tree_edges_1:
                        idx = self.graph_builder.get_ID_from_edge(e)
                        self.data_edges[idx].class_EU = major_class_sub_tree_1
                    for e in sub_tree_edges_2:
                        idx = self.graph_builder.get_ID_from_edge(e)
                        self.data_edges[idx].class_EU = major_class_sub_tree_2

                    # print("\nMODIFICA FATTA!!!")

                    # print(f"Archi uscenti nodo: {node} ", out_edges)
                    # print(f"Archi entranti nodo: {node} ", list(DG.in_edges(node)))

                    # print(f"Arco entrante riferimento da {data_in_edge.n1} a {data_in_edge.n2}")

                    data_out_edge_1 = self.data_edges[idx_out_e_1]
                    data_out_edge_2 = self.data_edges[idx_out_e_2]

                    assert (
                        data_in_edge.class_nn != major_class_sub_tree_1
                        or data_in_edge.class_nn != major_class_sub_tree_2
                    ), "uguale ad entrambi!!!"

                    if data_in_edge.class_nn != major_class_sub_tree_1:
                        to_remove = data_out_edge_1
                        to_stay = data_out_edge_2
                        outer_node_remove = out_edges[0][1]
                        outer_node_stay = out_edges[1][1]
                    elif data_in_edge.class_nn != major_class_sub_tree_2:
                        to_remove = data_out_edge_2
                        to_stay = data_out_edge_1
                        outer_node_remove = out_edges[1][1]
                        outer_node_stay = out_edges[0][1]

                    # print(f"rimuovo arco da {to_remove.n1} a {to_remove.n2}")
                    self.graph_builder.graph.remove_edge(to_remove.n1, to_remove.n2)

                    dict_coord = nx.get_node_attributes(
                        self.graph_builder.graph, "coordinates"
                    )
                    new_node = self.graph_builder.get_new_ID_node()
                    self.graph_builder.graph.add_node(
                        new_node, coordinates=dict_coord[node]
                    )

                    self.data_nodes[new_node] = Node(
                        ID=new_node, id_duplicate_node_3_degree=node
                    )
                    self.data_nodes[node].id_duplicate_node_3_degree = new_node

                    # print(f"vecchio nodo {node}")
                    # print(f"nuovo nodo {new_node}")

                    to_remove.n1 = new_node
                    to_remove.n2 = outer_node_remove
                    # print(f"assegno orientamento to remove da {to_remove.n1} a {to_remove.n2}")
                    to_remove.assign_forward_orientation(to_remove.n1, to_remove.n2)

                    # print(f"aggiungo arco da {to_remove.n1} a {to_remove.n2}")
                    self.graph_builder.graph.add_edge(
                        to_remove.n1, to_remove.n2, ID=to_remove.ID
                    )

                    # print(f"assegno orientamento to stay da {to_stay.n1} a {to_stay.n2}")
                    to_stay.n1 = node
                    to_stay.n2 = outer_node_stay
                    to_stay.assign_forward_orientation(to_stay.n1, to_stay.n2)

                    # print(f"vecchio nodo {node} nuovo nodo {new_node}")
                    # print("nodi agganciati vecchio nodo ", self.graph_builder.get_nodes_adjacent_to_node(node))
                    # print("nodi agganciati nuovo nodo ", self.graph_builder.get_nodes_adjacent_to_node(new_node))

                    # DG = self.graph_builder.get_DG_from_graph(self.data_edges)
                    # show_directed_graph(DG)

                    assert (
                        len(self.graph_builder.get_ID_edges_connected_to_node(new_node))
                        == 1
                    ), "più di un arco agganciato al nuovo nodo"
                    assert (
                        len(self.graph_builder.get_ID_edges_connected_to_node(node))
                        == 2
                    ), "più di due archi agganciati al vecchio nodo"

                    DG = self.graph_builder.get_DG_from_graph(self.data_edges)
                    # print(list(DG.out_edges(node)))
                    # print(list(DG.in_edges(node)))
                    if len(list(DG.out_edges(node))) < 1:
                        assert False, "non c'è l'arco uscente dal vecchio nodo"
                    elif len(list(DG.out_edges(node))) > 1:
                        assert False, "più di un arco uscente dal vecchio nodo"

                    if len(list(DG.in_edges(node))) < 1:
                        assert False, "non c'è l'arco entrante dal vecchio nodo"
                    elif len(list(DG.in_edges(node))) > 1:
                        assert False, "più di un arco entrante dal vecchio nodo"

                    if len(list(DG.out_edges(new_node))) < 1:
                        assert False, "non c'è l'arco uscente dal nuovo nodo"
                    elif len(list(DG.out_edges(new_node))) > 1:
                        assert False, "più di un arco uscente dal nuovo nodo"

                    splitted = True

            # DG = self.graph_builder.get_DG_from_graph(self.data_edges)
            # show_directed_graph(DG)

    def resolve_periferical_triplets(self):
        DG = self.graph_builder.get_DG_from_graph(self.data_edges)

        for node in DG.nodes():
            out_edges = list(DG.out_edges(node))
            in_edges = list(DG.in_edges(node))

            # se non ha esattamente 1 entrante e 0 uscenti lo skippo
            if len(out_edges) != 0 or len(in_edges) != 1:
                continue

            other_node = in_edges[0][0]
            assert other_node != node, "stesso nodo!!"

            out_other_edges = list(DG.out_edges(other_node))
            # se l'altro non ha esattamente due uscenti lo skippo
            if len(out_other_edges) != 2:
                continue

            out_other_edges.remove(in_edges[0])
            assert len(out_other_edges) == 1, "più di un arco uscente dall'altro"

            in_other_edges = list(DG.in_edges(other_node))
            # se l'altro non ha esattamente 1 entrante lo skippo
            if len(in_other_edges) != 1:
                continue

            # se quello entrante e uscente dell'altro nodo hanno la stessa classe, allora l'arco di questo nodo deve avere la stessa classe
            idx_in_other_edge = DG[in_other_edges[0][0]][in_other_edges[0][1]]["ID"]
            idx_out_other_edge = DG[out_other_edges[0][0]][out_other_edges[0][1]]["ID"]
            idx_in_edge = DG[in_edges[0][0]][in_edges[0][1]]["ID"]

            if (
                self.data_edges[idx_in_other_edge].class_EU
                == self.data_edges[idx_out_other_edge].class_EU
            ):
                if (
                    self.data_edges[idx_in_edge].class_EU
                    != self.data_edges[idx_in_other_edge].class_EU
                ):
                    self.data_edges[idx_in_edge].class_EU = self.data_edges[
                        idx_in_other_edge
                    ].class_EU
                    self.data_nodes[node].periferical_triplet = True

        # DG = self.graph_builder.get_DG_from_graph(self.data_edges)
        # show_directed_graph(DG)

    def resolve_periferical_triplets_simple(self):
        for edge in self.data_edges.values():
            n1 = edge.n1
            n2 = edge.n2
            id_edges_connected_1 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n1)
            )
            id_edges_connected_1.remove(edge.ID)

            id_edges_connected_2 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n2)
            )
            id_edges_connected_2.remove(edge.ID)

            # l'arco deve avere esattamente un nodo con 1 arco associato e l'altro con 2 archi associati
            if len(id_edges_connected_1) == 0 and len(id_edges_connected_2) == 2:
                # se i due archi hanno classe uguale, allora questo arco deve avere la stessa classe
                edge1 = self.data_edges[id_edges_connected_2[0]]
                edge2 = self.data_edges[id_edges_connected_2[1]]
                if edge1.class_nn == edge2.class_nn:
                    edge.class_EU = edge1.class_nn
            elif len(id_edges_connected_2) == 0 and len(id_edges_connected_1) == 2:
                # se i due archi hanno classe uguale, allora questo arco deve avere la stessa classe
                edge1 = self.data_edges[id_edges_connected_1[0]]
                edge2 = self.data_edges[id_edges_connected_1[1]]
                if edge1.class_nn == edge2.class_nn:
                    edge.class_EU = edge1.class_nn

    def resolve_periferical_triplets_simple_EU(self):
        for edge in self.data_edges.values():
            n1 = edge.n1
            n2 = edge.n2
            id_edges_connected_1 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n1)
            )
            id_edges_connected_1.remove(edge.ID)

            id_edges_connected_2 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n2)
            )
            id_edges_connected_2.remove(edge.ID)

            # l'arco deve avere esattamente un nodo con 1 arco associato e l'altro con 2 archi associati
            if len(id_edges_connected_1) == 0 and len(id_edges_connected_2) == 2:
                # se i due archi hanno classe uguale, allora questo arco deve avere la stessa classe
                edge1 = self.data_edges[id_edges_connected_2[0]]
                edge2 = self.data_edges[id_edges_connected_2[1]]
                if edge1.class_EU == edge2.class_EU:
                    edge.class_EU = edge1.class_EU
            elif len(id_edges_connected_2) == 0 and len(id_edges_connected_1) == 2:
                # se i due archi hanno classe uguale, allora questo arco deve avere la stessa classe
                edge1 = self.data_edges[id_edges_connected_1[0]]
                edge2 = self.data_edges[id_edges_connected_1[1]]
                if edge1.class_EU == edge2.class_EU:
                    edge.class_EU = edge1.class_EU

    def resolve_surrounded_edges(self):
        for edge in self.data_edges.values():
            # prendi tutti gli archi adiacenti
            n1 = edge.n1
            n2 = edge.n2
            id_edges_connected_1 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n1)
            )
            id_edges_connected_1.remove(edge.ID)

            id_edges_connected_2 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n2)
            )
            id_edges_connected_2.remove(edge.ID)

            # voglio che non sia un arco terminale: il numero di adiacenti da entrambi i lati deve essere maggiore di 0
            if len(id_edges_connected_1) <= 0 or len(id_edges_connected_2) <= 0:
                continue

            adjacent_edges = id_edges_connected_1 + id_edges_connected_2

            all_same_class = True
            for id_edge in adjacent_edges:
                if (
                    self.data_edges[id_edge].class_EU
                    != self.data_edges[adjacent_edges[0]].class_EU
                ):
                    all_same_class = False

            if all_same_class:
                if edge.class_EU != self.data_edges[adjacent_edges[0]].class_EU:
                    edge.class_EU = self.data_edges[adjacent_edges[0]].class_EU

    def resolve_wrong_bifurcation(self):
        for edge in self.data_edges.values():
            n1 = edge.n1
            n2 = edge.n2
            id_edges_connected_1 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n1)
            )
            id_edges_connected_1.remove(edge.ID)

            id_edges_connected_2 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n2)
            )
            id_edges_connected_2.remove(edge.ID)

            # non voglio che sia terminale
            if len(id_edges_connected_1) <= 0 or len(id_edges_connected_2) <= 0:
                continue

            # da una parte ho esattamente 2 archi associati
            if len(id_edges_connected_1) == 2:
                # devono avere lo stesso colore e deve essere diverso dal mio
                if (
                    self.data_edges[id_edges_connected_1[0]].class_EU
                    == self.data_edges[id_edges_connected_1[1]].class_EU
                    and self.data_edges[id_edges_connected_1[0]].class_EU
                    != edge.class_EU
                ):
                    # se gli archi dall'altra parte hanno lo stesso colore mio, allora sono i due archi ad essere sbagliati
                    all_equal_me = True
                    all_equal_other = True
                    for id_edge in id_edges_connected_2:
                        if edge.class_EU != self.data_edges[id_edge].class_EU:
                            all_equal_me = False

                        if (
                            self.data_edges[id_edges_connected_1[0]].class_EU
                            != self.data_edges[id_edge].class_EU
                        ):
                            all_equal_other = False

                    if all_equal_me:
                        self.data_edges[
                            id_edges_connected_1[0]
                        ].class_EU = edge.class_EU
                        self.data_edges[
                            id_edges_connected_1[1]
                        ].class_EU = edge.class_EU
                    elif all_equal_other:
                        edge.class_EU = self.data_edges[
                            id_edges_connected_1[0]
                        ].class_EU

            if len(id_edges_connected_2) == 2:
                # devono avere lo stesso colore e deve essere diverso dal mio
                if (
                    self.data_edges[id_edges_connected_2[0]].class_EU
                    == self.data_edges[id_edges_connected_2[1]].class_EU
                    and self.data_edges[id_edges_connected_2[0]].class_EU
                    != edge.class_EU
                ):
                    # se gli archi dall'altra parte hanno lo stesso colore mio, allora sono i due archi ad essere sbagliati
                    all_equal_me = True
                    all_equal_other = True
                    for id_edge in id_edges_connected_1:
                        if edge.class_EU != self.data_edges[id_edge].class_EU:
                            all_equal_me = False

                        if (
                            self.data_edges[id_edges_connected_2[0]].class_EU
                            != self.data_edges[id_edge].class_EU
                        ):
                            all_equal_other = False

                    if all_equal_me:
                        self.data_edges[
                            id_edges_connected_2[0]
                        ].class_EU = edge.class_EU
                        self.data_edges[
                            id_edges_connected_2[1]
                        ].class_EU = edge.class_EU
                    elif all_equal_other:
                        edge.class_EU = self.data_edges[
                            id_edges_connected_2[0]
                        ].class_EU

    def resolve_wrong_bifurcation_DG(self):
        def is_terminal(edge):
            n1 = edge.n1
            n2 = edge.n2
            id_edges_connected_1 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n1)
            )
            id_edges_connected_1.remove(edge.ID)

            id_edges_connected_2 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n2)
            )
            id_edges_connected_2.remove(edge.ID)

            return len(id_edges_connected_1) == 0 or len(id_edges_connected_2) == 0

        DG = self.graph_builder.get_DG_from_graph(self.data_edges)

        for node in DG.nodes():
            out_edges = list(DG.out_edges(node))
            in_edges = list(DG.in_edges(node))

            # se non ha esattamente 1 entrante e 2 uscenti lo skippo
            if len(in_edges) != 1 or len(out_edges) != 2:
                continue

            in_e1 = self.data_edges[DG[in_edges[0][0]][in_edges[0][1]]["ID"]]
            out_e2 = self.data_edges[DG[out_edges[0][0]][out_edges[0][1]]["ID"]]
            out_e3 = self.data_edges[DG[out_edges[1][0]][out_edges[1][1]]["ID"]]

            # i due archi uscenti devono avere la stessa classe e deve essere diversa da quello entrante
            if out_e2.class_EU == out_e3.class_EU and in_e1 != out_e2.class_EU:
                # i due archi uscenti devono essere terminali
                if is_terminal(out_e2) and is_terminal(out_e3):
                    # uso gli archi associati all'arco entrante per risolvere l'errore
                    nodes = [in_e1.n1, in_e1.n2]
                    nodes.remove(in_e1.get_common_node_with(out_e2))

                    adjacent_edges = list(
                        self.graph_builder.get_ID_edges_connected_to_node(nodes[0])
                    )
                    adjacent_edges.remove(in_e1.ID)

                    # se tutti questi archi hanno la stessa classe faccio qualcosa
                    all_class_equal = True
                    for id_e in adjacent_edges:
                        if (
                            self.data_edges[id_e].class_EU
                            != self.data_edges[adjacent_edges[0]].class_EU
                        ):
                            all_class_equal = False

                    if all_class_equal:
                        # se la classe è uguale alla mia allora sono quei due ad essere sbagliati
                        if self.data_edges[adjacent_edges[0]].class_EU == in_e1:
                            out_e2.class_EU = in_e1.class_EU
                            out_e3.class_EU = in_e1.class_EU
                        # altrimenti sono io ad essere sbagliato
                        else:
                            in_e1.class_EU = out_e2.class_EU

    def resolve_4_cross_over_turned_3(self):
        # prendo tutti i nodi di grado 3
        nodes_3_degree = self.graph_builder.get_nodes_with_degree(3)
        for node in nodes_3_degree:
            id_edges = list(self.graph_builder.get_ID_edges_connected_to_node(node))
            # devo trovare un arco che è a circa 180° dall'altro e 90° da un altro
            for id_e in id_edges:
                id_edges_copy = id_edges.copy()
                id_edges_copy.remove(id_e)

                e1 = self.data_edges[id_e]
                e2 = self.data_edges[id_edges_copy[0]]
                e3 = self.data_edges[id_edges_copy[1]]

                # show_edges([e1, e2, e3], type_class=TypeClassification.EU)
                # print("e1:")
                # show_edges([e1], type_class=TypeClassification.EU)
                # print(f"Angolo e1 con e2: {e1.get_angle_with(e2)}")
                # print(f"Angolo e1 con e3: {e1.get_angle_with(e3)}")
                # print()

                if (
                    e1.get_angle_with(e2) >= 160
                    and e1.get_angle_with(e3) >= 85
                    and e1.get_angle_with(e3) <= 95
                ):
                    # faccio qualcosa solo se sono di colore diverso
                    if e1.class_EU != e2.class_EU:
                        # se ho il colore diverso dal 90° allora deve cambiare l'altro, altrimenti cambio io
                        if e1.class_EU == e3.class_EU:
                            e1.class_EU = e2.class_EU
                        else:
                            e2.class_EU = e1.class_EU

                    break

    def resolve_after_structural_4_crossover(self):
        for edge in self.data_edges.values():
            n1 = edge.n1
            n2 = edge.n2
            id_edges_connected_1 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n1)
            )
            id_edges_connected_1.remove(edge.ID)

            id_edges_connected_2 = list(
                self.graph_builder.get_ID_edges_connected_to_node(n2)
            )
            id_edges_connected_2.remove(edge.ID)

            # voglio che abbia esattamente 1 arco da una parte e 2 dall'altra
            if len(id_edges_connected_1) == 1 and len(id_edges_connected_2) == 2:
                # se quei due hanno lo stesso colore
                e1 = self.data_edges[id_edges_connected_2[0]]
                e2 = self.data_edges[id_edges_connected_2[1]]
                e3 = self.data_edges[id_edges_connected_1[0]]
                if e1.class_EU == e2.class_EU:
                    if e1.class_EU != edge.class_EU:
                        edge.class_EU = e1.class_EU

                    if e1.class_EU != e3.class_EU:
                        e3.class_EU = e1.class_EU
            elif len(id_edges_connected_2) == 1 and len(id_edges_connected_1) == 2:
                # se quei due hanno lo stesso colore
                e1 = self.data_edges[id_edges_connected_1[0]]
                e2 = self.data_edges[id_edges_connected_1[1]]
                e3 = self.data_edges[id_edges_connected_2[0]]
                if e1.class_EU == e2.class_EU:
                    if e1.class_EU != edge.class_EU:
                        edge.class_EU = e1.class_EU

                    if e1.class_EU != e3.class_EU:
                        e3.class_EU = e1.class_EU

    def overwrite_euristic(self, prob):
        for edge in self.data_edges.values():
            if edge.probabilites_class[edge.class_nn] >= prob:
                edge.class_EU = edge.class_nn

    def highlight_all_errors(self):
        for edge in self.data_edges.values():
            if edge.class_nn != edge.class_GT:
                edge.highlight = True

    def highlight_euristic_errors(self):
        for edge in self.data_edges.values():
            if edge.class_EU != edge.class_GT:
                edge.highlight = True
            else:
                edge.highlight = False

    def highlight_euristic_periferic_errors(self):
        for edge in self.data_edges.values():
            if edge.class_EU != edge.class_GT:
                # l'arco ha almeno un nodo con un solo arco associato
                n1 = edge.n1
                n2 = edge.n2
                if (
                    len(list(self.graph_builder.graph.edges(n1))) == 1
                    or len(list(self.graph_builder.graph.edges(n2))) == 1
                ):
                    edge.highlight = True
            else:
                edge.highlight = False

    def highlight_periferic_errors(self):
        for edge in self.data_edges.values():
            if edge.class_nn != edge.class_GT:
                # l'arco ha almeno un nodo con un solo arco associato
                n1 = edge.n1
                n2 = edge.n2
                if (
                    len(list(self.graph_builder.graph.edges(n1))) == 1
                    or len(list(self.graph_builder.graph.edges(n2))) == 1
                ):
                    edge.highlight = True

