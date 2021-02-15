import constants
from utils_image import get_blank_rgb_image, colorize_cells, show_image, show_multiple_edges, get_rgb_color_from_class


def assign_EU_class_to_edges(data_edges, data_nodes, graph_builder):

    data_edges = resolve_nodes_4_degree(data_edges, data_nodes, graph_builder)

    return data_edges


def resolve_nodes_4_degree(edges, data_nodes, graph_builder):
    def flipped_class(class_):
        if class_ == constants.ARTERY_CLASS:
            return constants.VEIN_CLASS

        if class_ == constants.VEIN_CLASS:
            return constants.ARTERY_CLASS

    def married(couple):
        return couple[0].class_nn == couple[1].class_nn

    def get_edge_to_couple_with_max_sum_of_angles(edges):
        max_sum_angles = -1
        edge_to_couple = None
        for e in edges:
            couple = [edge0, e]

            other_couple = edges.copy()
            other_couple.remove(e)

            sum_angles = couple[0].get_angle_with(couple[1]) + other_couple[0].get_angle_with(other_couple[1])
            if sum_angles > max_sum_angles:
                max_sum_angles = sum_angles
                edge_to_couple = e

        return edge_to_couple

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
            if graph_builder.get_minimum_distance_from_nodes_BFS(node, graph_builder.source_nodes, visited) != -1:
                return True

        return False

    def show_original_crossover(edges):
        blank_img = get_blank_rgb_image()
        for edge in edges:
            color = get_rgb_color_from_class(edge.class_nn)
            colorize_cells(color, edge.skel_coord_pixels, blank_img)
        show_image(blank_img)

    def show_corrected_crossover(edges):
        blank_img = get_blank_rgb_image()
        for edge in edges:
            color = get_rgb_color_from_class(edge.class_EU)
            colorize_cells(color, edge.skel_coord_pixels, blank_img)
        show_image(blank_img)

    def show_skel_coupled_crossover(first_couple, second_couple):
        blank_img = get_blank_rgb_image()
        color = [0, 255, 0]
        colorize_cells(color, first_couple[0].skel_coord_pixels, blank_img)
        # show_image(blank_img)

        color = [0, 0, 255]
        colorize_cells(color, first_couple[1].skel_coord_pixels, blank_img)
        # show_image(blank_img)

        color = [255, 0, 0]
        colorize_cells(color, second_couple[0].skel_coord_pixels, blank_img)
        # show_image(blank_img)

        colorize_cells(color, second_couple[1].skel_coord_pixels, blank_img)
        show_image(blank_img)

    def show_coupled_crossover(first_couple, second_couple):
        blank_img = get_blank_rgb_image()
        color = [0, 255, 0]
        colorize_cells(color, first_couple[0].coord_pixels, blank_img)
        # show_image(blank_img)

        color = [0, 255, 0]
        colorize_cells(color, first_couple[1].coord_pixels, blank_img)
        # show_image(blank_img)

        color = [255, 255, 0]
        colorize_cells(color, second_couple[0].coord_pixels, blank_img)
        # show_image(blank_img)

        colorize_cells(color, second_couple[1].coord_pixels, blank_img)
        show_image(blank_img)

    # per ogni nodo da 4 faccio qualcosa
    nodes = list(graph_builder.graph.nodes.keys()).copy()
    for node in nodes:
        continue
        # skippa quelli troppo vicini ai nodi sorgente
        # if graph_builder.get_minimum_distance_from_any_source_node_BFS(node) <= 3:
        #     continue

        # if graph_builder.get_minimum_euclidean_distance_from_any_source_node(node) <= 100:
        #     continue

        edges_connected = list(graph_builder.graph.edges(node))
        if len(edges_connected) != 4:
            continue

        # dammi l'oggetto arco a partire dall'arco ottenuto da networkx
        four_edges = []
        skip = False
        for e in edges_connected:
            idx = graph_builder.get_ID_from_edge(e)
            four_edges.append(edges[idx])
            if edges[idx].length < 12:
                skip = True
        # if skip:
        #     continue

        # prendo un arco a caso dei 4
        edge0 = four_edges.pop(0)

        # scelgo di accoppiarmi con l'arco per cui la somma degli angoli delle coppie formate è massima
        edge_to_couple = get_edge_to_couple_with_max_sum_of_angles(four_edges)
        four_edges.remove(edge_to_couple)

        # ho determinato le due coppie di archi
        first_couple = [edge0, edge_to_couple]
        second_couple = [four_edges[0], four_edges[1]]

        # skippo l'incrocio se ho capito che è un falso incrocio
        # devo poter raggiungere una source da entrambe le parti
        can_reach1 = can_reach_source(first_couple, node)
        can_reach2 = can_reach_source(second_couple, node)
        if can_reach1 or can_reach2:
            if not can_reach_source(first_couple, node) or not can_reach_source(second_couple, node):
                data_nodes[node].fake_cross_over = True
                continue

        # se la prima coppia è in accordo
        if married(first_couple):
            if not married(second_couple):
                # print("AGISCO QUI!")
                second_couple[0].class_EU = flipped_class(first_couple[0].class_nn)
                second_couple[1].class_EU = flipped_class(first_couple[0].class_nn)

                data_nodes[node].corrected_by_EU = True
        else:
            # se la seconda coppia è in accordo
            if married(second_couple):
                # print("SONO QUI!!")
                first_couple[0].class_EU = flipped_class(second_couple[0].class_nn)
                first_couple[1].class_EU = flipped_class(second_couple[0].class_nn)

                data_nodes[node].corrected_by_EU = True

        # faccio il cambio strutturale del nodo da 4
        graph_builder.resolve_structural_node_4(node, first_couple, second_couple, data_nodes)

        # show_skel_coupled_crossover(first_couple, second_couple)
        # show_coupled_crossover(first_couple, second_couple)
        # show_original_crossover(first_couple + second_couple)
        # show_corrected_crossover(first_couple + second_couple)

        # prendo un arco e calcolo l'angolo con gli altri 3
        # quello con l'angolo più elevato dei 3 è il mio accoppiato
        # creo due coppie separate, io e l'accoppiato e gli altri due
        # se la mia coppia è d'accordo e l'altra è diversa, metti la classe diversa da noi a loro
        # se la mia coppia è in disaccordo e l'altra è uguale, metti la classe diversa da loro
        # negli altri casi non fare nulla

        # la modifica del grafo e duplicazione del nodo poi la farò in un secondo momento
    return edges
