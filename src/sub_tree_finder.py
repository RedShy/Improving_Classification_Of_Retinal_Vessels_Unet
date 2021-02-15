import networkx as nx
from networkx import NetworkXNoCycle
from utils_image import get_blank_rgb_image, show_edges, show_image, colorize_cells, get_matrix_cells


def check_sub_trees(data_edges, data_nodes, graph_builder):
    nodes_distance_1 = graph_builder.get_nodes_with_distance_from_any_source_nodes(1)
    assert nodes_distance_1, "NIENTE NODES DIS 1"

    for node in nodes_distance_1:
        sub_tree_data_edges = get_data_edges_sub_tree(node, graph_builder, data_edges)
        img_tmp = get_blank_rgb_image()
        show_edges(edges=sub_tree_data_edges, img=img_tmp, show=False)

        show_image(img_tmp)

        sub_tree_nodes = graph_builder.get_sub_tree(node)
        if is_a_good_tree(sub_tree_nodes, graph_builder, img_tmp, data_edges, data_nodes):
            print("è un buon albero!!!!!")
        else:
            print("non è un buon albero!!!!!!")

        show_image(img_tmp)

        # come radice dell'albero prendo il nodo non presente che è agganciato ad un arco dello start
        # nodes_adjacents_to_start = graph_builder.get_nodes_adjacent_to_node(node)
        # for n in nodes_adjacents_to_start:
        #     if n not in sub_tree_nodes:
        #         new_start = n
        # sub_tree_nodes.add(new_start)

        # graph_builder.separate_trees(node, sub_tree_nodes, data_edges, img_tmp)


def is_a_good_tree(sub_tree_nodes, graph_builder, img, data_edges, data_nodes):
    sub_graph = graph_builder.graph.subgraph(sub_tree_nodes)
    try:
        cycle = nx.find_cycle(sub_graph, orientation="ignore")
        print("Trovato ciclo!!!", list(cycle))
        colorize_cycle(cycle, graph_builder, img, data_edges)
        return False
    except NetworkXNoCycle:
        pass

    for node in sub_tree_nodes:
        if len(graph_builder.get_nodes_adjacent_to_node(node)) == 4 and not data_nodes[node].fake_cross_over:
            print("nodo con grado 4!!!!!!")

            id_edges = graph_builder.get_ID_edges_connected_to_node(node)
            for id_ in id_edges:
                colorize_cells([255, 255, 0], data_edges[id_].coord_pixels, img)
            return False
        elif len(graph_builder.get_nodes_adjacent_to_node(node)) > 4:
            print("nodo con grado > di 4!!!!!!")

            id_edges = graph_builder.get_ID_edges_connected_to_node(node)
            for id_ in id_edges:
                colorize_cells([255, 255, 0], data_edges[id_].coord_pixels, img)
            return False

    return True


def colorize_cycle(cycle, graph_builder, img, data_edges):
    for edge_ in cycle:
        id_edge = graph_builder.get_ID_from_edge((edge_[0], edge_[1]))
        edge = data_edges[id_edge]

        rgb_edge = [255, 255, 0]
        colorize_cells(rgb_edge, edge.coord_pixels, img)

        rgb_node = [0, 255, 0]
        colorize_cells(rgb_node, get_matrix_cells(edge.n1_coord, 2), img)
        colorize_cells(rgb_node, get_matrix_cells(edge.n2_coord, 2), img)


def get_data_edges_sub_tree(node, graph_builder, data_edges):
    sub_tree_edges = graph_builder.get_edges_sub_tree(node)

    sub_tree_data_edges = []
    for id_edge_ in sub_tree_edges:
        sub_tree_data_edges.append(data_edges[id_edge_])

    return sub_tree_data_edges
