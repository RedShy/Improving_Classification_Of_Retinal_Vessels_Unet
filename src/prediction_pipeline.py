import torch
import math
import numpy as np
import pickle
import sys
import networkx as nx
import pandas as pd
import gc
from IPython.display import display
from matplotlib import pyplot as plt
from enum import Enum
import constants
from reporter import Reporter
from constants import TypeClassification
from edge import Edge
from node import Node
from graph_builder import GraphBuilder
from logic_program import LogicProgram
from pixels_edge_finder import get_coord_pixels_for_all_edges, get_simple_path
from utils_image import (
    colorize_cells,
    get_rgb_from_black_white,
    show_image,
    show_graph,
    get_colored_img_from_labels,
    show_pandas_dataframe,
    get_matrix_cells,
    get_rgb_color_from_class,
    show_images_single_row,
    show_big_plots_images,
    show_cells,
    show_multiple_edges,
    get_blank_rgb_image,
    show_multiple_edges_tmp1,
    generate_black_white_img_from_probs,
)
from euristic_edge_labeller import assign_EU_class_to_edges
from utils_metrics import print_confusion_matrix
from utils_edges import get_probabilities_for_edges, assign_NN_class_to_edges, assign_GT_class_to_edges
from unet import UNet
from dataset import DatasetRetinalImagesRITE
from sub_tree_finder import check_sub_trees
from euristic import Euristic

# IN_COLAB = "google.colab" in sys.modules
IN_COLAB = False

if IN_COLAB:
    print("Running on CoLab")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    print("Not running on CoLab")
    device = "cpu"

CACHE_NET = True
CACHE_GRAPH = True
CACHE_IMG_LABEL = True
CACHE_EDGES = True
CACHE_BLACK_WHITE = True

SAVE_EDGES = False
if CACHE_EDGES:
    SAVE_EDGES = False


def pipeline():
    if CACHE_NET:
        net = None
        disk_nn_probs = pickle.load(open("nn_probs.pickle", "rb"))
    else:
        disk_nn_probs = None
        net = UNet(num_classes=4)
        # net.load_state_dict(torch.load("./models/model_training_loss_0.03.pt"))
        # net.load_state_dict(torch.load("./models/model_test_loss_0.36.pt"))
        net.load_state_dict(torch.load("../models/model20x_test_loss_0.138.pt"))
        net = net.to(device)

    if CACHE_IMG_LABEL:
        disk_colored_imgs_label = pickle.load(open("colored_labels.pickle", "rb"))

    if CACHE_EDGES:
        disk_data_edges = pickle.load(open("data_edges.pickle", "rb"))

    if CACHE_BLACK_WHITE:
        disk_black_white_imgs = pickle.load(open("black_white_imgs.pickle", "rb"))

    # training_dataset = DatasetRetinalImagesRITE("../data/training/images/", "tif", "../data/training/av/", "png")
    test_dataset = DatasetRetinalImagesRITE("../data/test/images/", "tif", "../data/test/av/", "png")

    colored_imgs_label = []
    colored_imgs_nn = []
    colored_imgs_nn_structure_highlight = []
    colored_imgs_ASP = []

    colored_imgs_NN_ASP_highlight = []
    colored_imgs_NN_GT_highlight = []
    colored_imgs_NN_EU_highlight = []
    colored_imgs_EU_GT_highlight = []

    accuracies_NN = []
    accuracies_ASP = []
    accuracies_EU = []

    # dataset_tmp = []
    # dataset_tmp.append(test_dataset.__getitem__(9))

    if IN_COLAB:
        indices = range(test_dataset.__len__())
        # indices = [1, 2]
        # dataset = test_dataset
    else:
        # indices = [0]
        indices = range(20)
        # dataset = dataset_tmp

    # colored_imgs_label = []
    list_edges = []
    # list_blacks = []

    graph_builder = GraphBuilder(CACHE_GRAPH, "../data/test/coord_disks.txt")
    reporter = Reporter(graph_builder, 20)
    # per ogni immagine del test set
    for idx in indices:
        gc.collect()

        print(f"IMMAGINE {idx}...")

        # if not CACHE_IMG_LABEL or not CACHE_BLACK_WHITE or not CACHE_EDGES:
        input_img, label_img = test_dataset.__getitem__(idx)
        input_img = torch.from_numpy(input_img)
        input_img = input_img.permute(2, 0, 1)
        input_img = input_img.unsqueeze(0)
        input_img = input_img.to(device)

        reporter.labels.append(label_img)

        # if not CACHE_BLACK_WHITE or not CACHE_EDGES:
        nn_probs = get_probs_net(net, input_img, idx, disk_nn_probs)

        # print("ottieni immagine a colori dall'output rete")
        # colored_img_nn_output = generate_colored_img_from_probs(nn_probs)
        # show_image(colored_img_nn_output)

        print("generiamo l'immagine colorata a partire dall'etichetta")
        if CACHE_IMG_LABEL:
            colored_img_label = disk_colored_imgs_label[idx]
        else:
            colored_img_label = get_colored_img_from_labels(label_img)
        reporter.colored_imgs_label.append(colored_img_label)
        # show_image(img=colored_img_nn, save=True)

        print("ottieni immagine in bianco e nero dall'output rete")
        if CACHE_BLACK_WHITE:
            black_white_img = disk_black_white_imgs[idx]
        else:
            black_white_img = generate_black_white_img_from_probs(nn_probs)

        # list_blacks.append(black_white_img)

        print("produciamo il grafo")
        graph = graph_builder.generate_graph_from_black_white_img(black_white_img, idx)
        # show_image(graph_builder.skel)
        # show_graph(graph=graph, height_img=black_white_img.shape[0], save=True)

        print("inizializzo dati nodi")
        data_nodes = graph_builder.get_nodes_with_data()
        reporter.data_nodes = data_nodes

        print("inizializzo dati archi")
        if CACHE_EDGES:
            edges = disk_data_edges[idx]
        else:
            edges = graph_builder.get_edges_with_data()
            assert len(edges) == len(list(graph.edges)), "Numero classificazioni diverso dal numero di archi!!"

            print("otteniamo i pixel per ogni arco")
            edges = get_coord_pixels_for_all_edges(edges=edges, skeleton_img=graph_builder.skel, black_white_img=graph_builder.black_white_img)

            # print("ricerca dei nodi sorgente")
            # graph_builder.find_source_nodes(edges)

            print("otteniamo le probabilità per ogni arco")
            probabilities_edges = get_probabilities_for_edges(edges, nn_probs)

            print("assegnazione classi rete neurale")
            edges = assign_NN_class_to_edges(edges, probabilities_edges)

            print("assegnazione classi ground truth")
            edges = assign_GT_class_to_edges(edges, colored_img_label)

        if SAVE_EDGES:
            list_edges.append(edges)

        reporter.data_edges = edges

        # print("assegnazione classi ASP")
        # logic_program = LogicProgram(graph)
        # edges = logic_program.assign_ASP_class_to_edges(edges)

        print("assegnazione classi euristica")
        if idx >= 0:
            euristic = Euristic(edges, data_nodes, graph_builder)
            euristic.assign_orientation()
            euristic.resolve_nodes_4_degree()
            # euristic.resolve_nodes_3_degree()
            # euristic.resolve_periferical_triplets()

        # euristic.resolve_class_easy_nodes_4_degree()

        # edges = assign_EU_class_to_edges(edges, data_nodes, graph_builder)

        # print("generiamo l'immagine colorata dalle classi NN")
        # colored_img_NN = colorize_edges_to_img(TypeClassification.NN, edges=edges, black_white_img=graph_builder.black_white_img)

        # print("generiamo l'immagine colorata dalle classi ASP")
        # colored_img_ASP = colorize_edges_to_img(TypeClassification.ASP, edges=edges, black_white_img=graph_builder.black_white_img)

        # print("generiamo l'immagine colorata dalle classi EU")
        # colored_img_EU = colorize_edges_to_img(TypeClassification.EU, edges=edges, black_white_img=graph_builder.black_white_img)

        print("calcoliamo i valori di accuratezza")
        # accuracy_NN = get_accuracy_from_image(img=colored_img_NN, target_img=colored_img_label)
        # accuracy_ASP = get_accuracy_from_image(img=colored_img_ASP, target_img=colored_img_label)
        # accuracy_EU = get_accuracy_from_image(img=colored_img_EU, target_img=colored_img_label)

        # accuracies_NN.append(accuracy_NN)
        # accuracies_ASP.append(accuracy_ASP)
        # accuracies_EU.append(accuracy_EU)

        # print(f"\nAccuratezze\n NN: {accuracy_NN} EU: {accuracy_EU} ASP: {accuracy_ASP}\n")

        print("generazione dati su immagine")
        # reporter.generate_data_nodes_4(idx)
        # reporter.generate_data_good_classification(idx)
        # reporter.generate_data_nodes_3(idx)

        # reporter.generate_data_4_cross_over(idx)
        # reporter.generate_data_fake_4_cross_over(idx)
        # reporter.generate_data_split_nodes_3(idx)
        # reporter.generate_data_periferical_triplets(idx)
        # reporter.generate_data_corrected_edges(idx)
        # reporter.generate_metrics_fake_4_cross_over(idx)
        reporter.generate_metrics_single_img(idx, nn_probs)
        reporter.print_data_for_img(idx)
        # reporter.show_highlight_errors_EU_GT()

        # df = reporter.get_DataFrame_from_data()
        # print(df)

        # checks(edges, graph)

        # print("individuazione e elaborazione dei sotto-alberi")
        # check_sub_trees(edges, data_nodes, graph_builder)

        # print("generazione highlights differenze NN vs GT")
        # colored_img_NN_GT_highlight = get_img_difference_classes(edges, TypeClassification.NN, TypeClassification.GT, colored_img_NN)

        # print("generazione highlights differenze NN vs ASP")
        # colored_img_NN_ASP_highlight = get_img_difference_classes(edges, TypeClassification.NN, TypeClassification.ASP, colored_img_NN)

        # print("generazione highlights differenze NN vs EU")
        # colored_img_NN_EU_highlight = get_img_difference_classes(edges, TypeClassification.NN, TypeClassification.EU, colored_img_NN)

        # print("generazione highlights differenze EU vs GT")
        # colored_img_EU_GT_highlight = get_img_difference_classes(edges, TypeClassification.EU, TypeClassification.GT, colored_img_EU)

        # print("Disegno grafo sopra immagine")
        # colorize_structure_graph(edges, colored_img_NN)
        # colorize_source_nodes(graph_builder.source_nodes, graph, colored_img_NN)

        print("Generazione immagini...")
        # show_images_single_row([colored_img_NN, colored_img_label])
        # show_images_single_row([colored_img_NN, colored_img_NN_EU_highlight, colored_img_EU_GT_highlight, colored_img_label])
        # show_images_single_row([colored_img_nn_output, colored_img_NN])
        # show_images_single_row([colored_img_nn_structure_highlight, colored_img_label])
        # show_images_single_row([colored_img_nn, colored_img_ASP_highlight, colored_img_ASP, colored_img_label], "allfigures")
        # show_images_single_row([colored_img_nn, colored_img_ASP_highlight], "nnVSasp")
        # show_images_single_row([colored_img_nn, colored_img_label], "nnVSlabel")
        # show_images_single_row([colored_img_ASP_highlight, colored_img_label], "aspVSlabel")

        # colored_imgs_label.append(colored_img_label)
        # colored_imgs_nn.append(colored_img_NN)
        # colored_imgs_ASP.append(colored_img_ASP)

        # colored_imgs_NN_ASP_highlight.append(colored_img_NN_ASP_highlight)
        # colored_imgs_NN_GT_highlight.append(colored_img_NN_GT_highlight)
        # colored_imgs_NN_EU_highlight.append(colored_img_NN_EU_highlight)
        # colored_imgs_EU_GT_highlight.append(colored_img_EU_GT_highlight)

        # break

        # if not IN_COLAB:
        #     break

    # reporter.show_DataFrame()
    # reporter.show_dataframe_fake_4_crossover()
    # reporter.show_dataframe_edges_EU()
    reporter.show_dataframe_split_nodes_3()
    # reporter.report_all_images_metrics_fake_4_cross_over()

    # reporter.save_run_on_file()
    # reporter.show_all_runs()

    if SAVE_EDGES:
        pickle.dump(list_edges, open("data_edges.pickle", "wb"))

    # pickle.dump(list_blacks, open("black_white_imgs.pickle", "wb"))
    # pickle.dump(colored_imgs_label, open("colored_labels.pickle", "wb"))
    # pickle.dump(list_nn_probs, open("nn_probs.pickle", "wb"))
    # pickle.dump(list_graphs, open("graphs2.pickle", "wb"))

    # display_df_accuracies(accuracies_NN, accuracies_ASP, accuracies_EU)

    # show_big_plots_images([colored_imgs_nn, colored_imgs_NN_ASP_highlight, colored_imgs_ASP, colored_imgs_label])
    # show_big_plots_images([colored_imgs_nn_structure_highlight, colored_imgs_label])
    # show_big_plots_images([colored_imgs_nn, colored_imgs_NN_GT_highlight, colored_imgs_label])
    # show_big_plots_images([colored_imgs_NN_GT_highlight, colored_imgs_label])
    # show_big_plots_images([colored_imgs_nn, colored_imgs_NN_EU_highlight, colored_imgs_label])
    # show_big_plots_images([colored_imgs_nn, colored_imgs_NN_EU_highlight, colored_imgs_EU_GT_highlight, colored_imgs_label], split=True)


def get_probs_net(net, input_img, idx, disk_nn_probs):
    if CACHE_NET:
        return disk_nn_probs[idx]
    else:
        print("ottieni output rete")
        nn_output = net(input_img)

        print("ottieni probabilità rete")
        nn_probs = torch.nn.Softmax(1)(nn_output)

        return nn_probs


def display_df_accuracies(accuracies_nn, accuracies_ASP, accuracies_EU):
    df = pd.DataFrame(data={"Accuracy NN": accuracies_nn, "Accuracy ASP": accuracies_ASP, "Accuracy EU": accuracies_EU})
    df["Accuracy ASP - NN"] = df.apply(lambda row: row[1] - row[0], axis=1)
    df["Accuracy EU - NN"] = df.apply(lambda row: row[2] - row[0], axis=1)
    show_pandas_dataframe(df)

    display(df.agg({"Accuracy ASP - NN": ["min", "max", "median", "mean", "std"]}))
    display(df.agg({"Accuracy EU - NN": ["min", "max", "median", "mean", "std"]}))


def generate_colored_img_from_probs(nn_probs):
    labels = torch.argmax(nn_probs, dim=1).cpu().numpy()

    image = np.zeros(shape=(labels.shape[1], labels.shape[2], 3), dtype="uint8")
    for i in range(labels.shape[1]):
        for j in range(labels.shape[2]):
            if labels[0][i][j] == constants.BACKGROUND_CLASS:
                image[i][j] = constants.BACKGROUND_RGB_COLOR
            elif labels[0][i][j] == constants.ARTERY_CLASS:
                image[i][j] = constants.ARTERY_RGB_COLOR
            elif labels[0][i][j] == constants.VEIN_CLASS:
                image[i][j] = constants.VEIN_RGB_COLOR
            elif labels[0][i][j] == constants.UNCERTAINTY_CLASS:
                image[i][j] = constants.UNCERTAINTY_RGB_COLOR
    return image


def get_img_difference_GT(edges, colored_img_nn):
    colored_img_nn_highlight = np.copy(colored_img_nn)

    rgb_color = [255, 255, 0]
    for edge in edges.values():
        if edge.class_nn != edge.class_GT:
            colorize_cells(rgb_color, edge.coord_pixels, colored_img_nn_highlight)

            rgb_node = [0, 255, 0]
            rgb_edge = get_rgb_color_from_class(edge.class_GT)
            colorize_structure_edge(edge.n1_coord, edge.n2_coord, rgb_edge, rgb_node, colored_img_nn_highlight)

    return colored_img_nn_highlight


def get_img_difference_EU(edges, colored_img_nn):
    colored_img_nn_highlight = np.copy(colored_img_nn)

    rgb_color = [255, 255, 0]
    for edge in edges.values():
        if edge.class_nn != edge.class_EU:
            colorize_cells(rgb_color, edge.coord_pixels, colored_img_nn_highlight)

            rgb_node = [0, 255, 0]
            rgb_edge = get_rgb_color_from_class(edge.class_EU)
            colorize_structure_edge(edge.n1_coord, edge.n2_coord, rgb_edge, rgb_node, colored_img_nn_highlight)

    return colored_img_nn_highlight


def get_img_difference_classes(edges, type_first, type_second, colored_img):
    colored_img_highlight = np.copy(colored_img)

    rgb_color = [255, 255, 0]
    for edge in edges.values():
        if edge.get_class(type_first) != edge.get_class(type_second):
            colorize_cells(rgb_color, edge.coord_pixels, colored_img_highlight)

            rgb_node = [0, 255, 0]
            rgb_edge = get_rgb_color_from_class(edge.get_class(type_second))
            colorize_structure_edge(edge.n1_coord, edge.n2_coord, rgb_edge, rgb_node, colored_img_highlight)

    return colored_img_highlight


def colorize_edges_to_img(type_class, edges, black_white_img):
    colored_img = get_rgb_from_black_white(black_white_img)

    for edge in edges.values():
        class_ = edge.get_class(type_class)
        rgb_color = get_rgb_color_from_class(class_)

        # TODO: eliminare
        # cells = []
        # for i in range(10):
        #     if i < len(edge.skel_coord_pixels):
        #         cells.append(edge.skel_coord_pixels[i])
        colorize_cells(rgb_color, edge.coord_pixels, colored_img)

        # print(f"arco {edge.ID} num_pixels {len(edge.coord_pixels)} len {edge.get_length()} diam {len(edge.coord_pixels) / edge.get_length()}")
        # show_image(colored_img)

    return colored_img


def get_accuracy_from_image(img, target_img):
    correct_pixels = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y][0] == target_img[x][y][0] and img[x][y][1] == target_img[x][y][1] and img[x][y][2] == target_img[x][y][2]:
                correct_pixels += 1

    return correct_pixels / (img.shape[0] * img.shape[1])


def get_img_difference_ASP(edges, colored_img_ASP):
    colored_img_ASP_highlight = np.copy(colored_img_ASP)

    rgb_color = [255, 255, 0]
    for edge in edges.values():
        if edge.class_nn != edge.class_ASP:
            colorize_cells(rgb_color, edge.coord_pixels, colored_img_ASP_highlight)

            rgb_node = [0, 255, 0]
            rgb_edge = get_rgb_color_from_class(edge.class_ASP)
            colorize_structure_edge(edge.n1_coord, edge.n2_coord, rgb_edge, rgb_node, colored_img_ASP_highlight)

    return colored_img_ASP_highlight


def colorize_source_nodes(nodes, graph, img):
    dict_nodes_coordinates = nx.get_node_attributes(graph, "coordinates")
    color = [255, 255, 0]
    for node in nodes:
        coord = dict_nodes_coordinates[node]
        colorize_cells(color, get_matrix_cells(coord, 2), img)


def colorize_structure_graph(edges, colored_img):
    rgb_edge = [255, 255, 255]
    rgb_node = [0, 255, 0]
    for edge in edges.values():
        # colorize_skel_edge(edge, rgb_edge, rgb_node, colored_img)
        colorize_structure_edge(edge.n1_coord, edge.n2_coord, rgb_edge, rgb_node, colored_img)


def colorize_structure_edge(n1_coord, n2_coord, rgb_edge, rgb_node, img):
    cells = get_simple_path(n1_coord, n2_coord)
    colorize_cells(rgb_edge, cells, img)

    colorize_cells(rgb_node, get_matrix_cells(n1_coord, 2), img)
    colorize_cells(rgb_node, get_matrix_cells(n2_coord, 2), img)


def colorize_skel_edge(edge, rgb_edge, rgb_node, img):
    cells = edge.skel_coord_pixels
    colorize_cells(rgb_edge, cells, img)

    colorize_cells(rgb_node, get_matrix_cells(edge.n1_coord, 2), img)
    colorize_cells(rgb_node, get_matrix_cells(edge.n2_coord, 2), img)


def checks(edges, graph):
    assert len(edges) == len(graph.edges), "NUMERO ARCHI DIVERSO!!!"

    for edge in edges.values():
        edge.check()


def prova():
    a = [0.1, 0.2]
    b = [0.3, 0.1]
    df = pd.DataFrame(data={"Accuracy NN": a, "Accuracy ASP": b})
    df["Difference accuracy ASP - NN"] = df.apply(lambda row: row[1] - row[0], axis=1)
    show_pandas_dataframe(df)


# prova()
pipeline()

