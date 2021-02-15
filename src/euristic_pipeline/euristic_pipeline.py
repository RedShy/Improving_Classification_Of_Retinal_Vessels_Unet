import torch
import sys
import pickle
import numpy as np
import pandas as pd
import time
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import display, Image

sys.path.append("../")
from dataset import DatasetRetinalImagesRITE
from output_dataset import OutputDataset
from unet import UNet
from utils_metrics import get_metrics
from utils_edges import (
    get_probabilities_for_edges,
    assign_NN_class_to_edges,
    assign_GT_class_to_edges,
)
from utils_image import (
    get_colored_img_from_labels,
    show_images_single_row,
    parallel_get_colored_img_from_labels,
    show_edges,
    show_big_plots_images,
    generate_black_white_img_from_probs,
    get_labels_from_colored_img,
    show_image,
    get_square_inside_image,
)
from graph_builder import GraphBuilder
from pixels_edge_finder import get_coord_pixels_for_all_edges
from euristic import Euristic
import constants
from constants import TypeClassification
from reporter import Reporter


def generate_output_dataset():
    # caricamento rete
    file_net = "../../ld20_1.pt"
    net = UNet(num_classes=4)
    net.load_state_dict(torch.load(file_net))
    net = net.to(device)

    # produzione di output e grafo
    all_dataset = DatasetRetinalImagesRITE()
    type_folder = "all"
    all_dataset.initialize(
        f"../../data/{type_folder}/images/",
        "png",
        f"../../data/{type_folder}/av/",
        "png",
        f"../../data/{type_folder}/coord_disks.txt",
    )

    output_dataset = OutputDataset()
    output_dataset.initialize(
        f"../../data/{type_folder}/images/",
        "png",
        f"../../data/{type_folder}/av/",
        "png",
        f"../../data/{type_folder}/coord_disks.txt",
    )

    idx = 0
    for x, y, coord_disk in all_dataset:
        print(f"ELABORANDO IMMAGINE {idx}")

        input_img = torch.from_numpy(x)
        images = input_img.unsqueeze(0)

        images = images.permute(0, 3, 1, 2)
        images = images.to(device)
        print("OTTENIMENTO OUTPUT RETE")
        nn_output = net(images)
        nn_probs = torch.nn.Softmax(1)(nn_output)

        # ottenimento grafo
        print("OTTENIMENTO GRAFO")
        graph_builder = GraphBuilder(cache_graph=False)
        black_white_img = generate_black_white_img_from_probs(nn_probs)
        skel = graph_builder.preprocess_image(black_white_img, x=0, y=0, disk_radius=1)
        graph = graph_builder.generate_graph(skel, black_white_img)
        skel_no_pad = graph_builder.remove_pad_to_img(skel)

        print("SALVATAGGIO DATI")
        output_dataset.set_nn_output(idx, nn_probs.detach().cpu().numpy())
        output_dataset.set_graph(idx, graph)
        output_dataset.set_black_white_img(idx, black_white_img)
        output_dataset.set_skel(idx, skel_no_pad)

        print()

        idx += 1

    # salvataggio output_dataset
    f = open("./output_dataset.pkl", "wb")
    pickle.dump(output_dataset, f)
    f.close()


def euristic_pipeline():
    def append_metrics(metrics, dict):
        accuracy, precision, recall, f1 = metrics

        dict["accuracy"].append(accuracy)
        dict["precision"].append(precision)
        dict["recall"].append(recall)
        dict["f1"].append(f1)

    def get_means(dict_metrics):
        means = []
        for m in dict_metrics:
            means.append(np.mean(dict_metrics[m]))
        return means

    def get_diff(before, after):
        diff = []
        for i in range(len(before)):
            diff.append(after[i] - before[i])

        return diff

    def get_percentage_diff(before, after):
        perc = []
        for i in range(len(before)):
            perc.append(((after[i] / before[i]) - 1) * 100)

        return perc

    IMAGES_SHOW = True

    ld_number = 3
    ranges = {1: range(0, 10), 2: range(20, 30), 3: range(30, 40)}

    # caricamento output dataset
    with open(f"./data/output_dataset_ld20_{ld_number}.pkl", "rb") as f:
        output_dataset = pickle.load(f)

    metrics_imgs_NN = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    metrics_imgs_EU = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    # per ogni immagine, applico l'euristica
    start = time.time()
    range_ = ranges[ld_number]
    for idx in range_:
        print(f"\nELABORAZIONE IMMAGINE {idx}")
        (
            x,
            y,
            coord_disk,
            nn_probs,
            graph,
            skel,
            black_white_img,
        ) = output_dataset.__getitem__(idx)

        nn_probs = torch.from_numpy(nn_probs)

        graph_builder = GraphBuilder(cache_graph=False)
        graph_builder.graph = graph
        graph_builder.skel = skel
        graph_builder.black_white_img = black_white_img
        graph_builder.improve_graph()
        graph_builder.set_idx_to_edges()
        graph_builder.find_and_set_source_nodes(coord_disk)

        print("\n--- Generazione dati sui nodi... ---")
        data_nodes = graph_builder.get_nodes_with_data()

        print("\n--- Generazione dati sugli archi... ---")
        data_edges = graph_builder.get_edges_with_data()

        print("Ottenimento pixels per ogni arco...")
        data_edges = get_coord_pixels_for_all_edges(
            edges=data_edges,
            skeleton_img=graph_builder.skel,
            black_white_img=graph_builder.black_white_img,
        )

        print("Ottenimento probabilità su ogni arco...")
        probabilities_edges = get_probabilities_for_edges(data_edges, nn_probs)

        print("Assegnazione classi NN su ogni arco...")
        data_edges = assign_NN_class_to_edges(data_edges, probabilities_edges)

        print("Assegnazione classi GT su ogni arco...")
        colored_img_label_gt = get_colored_img_from_labels(y)
        data_edges = assign_GT_class_to_edges(data_edges, colored_img_label_gt)

        print("\n--- Euristica ---")
        euristic = Euristic(data_edges, data_nodes, graph_builder)

        print("Assegnazione orientamento archi...")
        euristic.assign_orientation()

        print("Applicazione correzioni")
        euristic.apply_corrections()

        print("\n--- Generazione immagine finale corretta ---")
        print("Generazione immagine vanilla rete...")
        labels_nn = torch.argmax(nn_probs, dim=1).permute(1, 2, 0).cpu().numpy()
        img_NN = get_colored_img_from_labels(labels_nn)

        print("Generazione immagine corretta...")
        img_EU = show_edges(
            data_edges.values(),
            nodes=False,
            structure=False,
            highlight=False,
            type_class=TypeClassification.EU,
            show=False,
            img=img_NN.copy(),
        )

        if IMAGES_SHOW:
            print("NN VS GT")
            show_images_single_row([img_NN, colored_img_label_gt])

            print("NN VS EU")
            show_images_single_row([img_NN, img_EU])

            print("NN VS ERRORI NN")
            euristic.highlight_all_errors()
            img_error = show_edges(
                data_edges.values(),
                nodes=True,
                structure=True,
                highlight=True,
                type_class=TypeClassification.NN,
                show=False,
                img=img_NN.copy(),
            )
            img_recolored_NN = show_edges(
                data_edges.values(),
                nodes=False,
                structure=False,
                highlight=False,
                type_class=TypeClassification.NN,
                show=False,
                img=img_NN.copy(),
            )
            show_images_single_row([img_recolored_NN, img_error])

            print("EU VS ERRORI EU")
            euristic.highlight_euristic_errors()
            img_error_EU = show_edges(
                data_edges.values(),
                nodes=True,
                structure=True,
                highlight=True,
                type_class=TypeClassification.EU,
                show=False,
                img=img_NN.copy(),
            )
            show_images_single_row([img_EU, img_error_EU])

            print("ERRORI NN VS ERRORI EU")
            show_images_single_row([img_error, img_error_EU])

        print("Ottenimento etichette da immagine corretta...")
        labels_EU = get_labels_from_colored_img(img_EU)

        print("Ottenimento metriche vanilla ed euristica")
        append_metrics(get_metrics(y, labels_nn), metrics_imgs_NN)
        append_metrics(get_metrics(y, labels_EU), metrics_imgs_EU)

    print("Visualizzazione sulle singole immagini")
    data = {
        "F1 NN": metrics_imgs_NN["f1"],
        "F1 EU": metrics_imgs_EU["f1"],
        "diff": get_diff(metrics_imgs_NN["f1"], metrics_imgs_EU["f1"]),
        "%": get_percentage_diff(metrics_imgs_NN["f1"], metrics_imgs_EU["f1"]),
    }
    df = pd.DataFrame(data=data, index=range_)
    display(df)

    print("Visualizzazione metriche medie")
    data = {
        "CD": get_means(metrics_imgs_NN),
        "CD corrected": get_means(metrics_imgs_EU),
        "diff": get_diff(get_means(metrics_imgs_NN), get_means(metrics_imgs_EU)),
        "%": get_percentage_diff(
            get_means(metrics_imgs_NN), get_means(metrics_imgs_EU)
        ),
    }
    df = pd.DataFrame(data=data, index=list(metrics_imgs_NN.keys()))
    display(df)

    end = time.time()
    print(f"\n\nTempo esecuzione pipeline: {end-start} s")


if "google.colab" in sys.modules:
    print("Running on CoLab")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    print("Not running on CoLab")
    device = "cpu"

# generate_output_dataset()
euristic_pipeline()

