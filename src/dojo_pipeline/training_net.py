import torch
import torch.optim as optim
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


sys.path.append("../")
from graph_builder import GraphBuilder
from pixels_edge_finder import get_coord_pixels_for_all_edges
from euristic import Euristic
import constants
from constants import TypeClassification
from reporter import Reporter
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
)
from unet import UNet

if "google.colab" in sys.modules:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

NORMALIZE_INPUT = False


def normalize_images(images, norm_parameters):
    means = norm_parameters[0]
    stds = norm_parameters[1]

    # sottraggo la media e divido per la deviazione standard sui 3 channels
    normalized_images = (images - torch.from_numpy(means)) / torch.from_numpy(stds)

    return normalized_images


def get_f1_on_dataset(net, dataset, norm_parameters):
    # conterranno tutte le etichette di tutte le immagini
    y_true_global = []
    nn_labels_global = []
    # per ogni immagine estraggo le classi fornite dalla rete e le etichette per ogni pixel
    for input_img, labels, coord_disk in dataset:
        input_img = torch.from_numpy(input_img)
        images = input_img.unsqueeze(0)

        # normalizzazione immagini in input
        if NORMALIZE_INPUT:
            images = normalize_images(images, norm_parameters)

        images = images.permute(0, 3, 1, 2)
        images = images.to(device)
        nn_output = net(images)

        nn_probs = torch.nn.Softmax(1)(nn_output)
        nn_labels = torch.argmax(nn_probs, dim=1).squeeze().cpu().numpy()

        y_true = labels.reshape(-1)
        nn_labels = nn_labels.reshape(-1)

        # uso solo le classi arteria e vena
        y_true_arteir_vein_only = []
        labels_NN_artery_vein_only = []
        for i in range(len(y_true)):
            if (
                y_true[i] != constants.BACKGROUND_CLASS
                and y_true[i] != constants.UNCERTAINTY_CLASS
            ):
                y_true_arteir_vein_only.append(y_true[i])
                labels_NN_artery_vein_only.append(nn_labels[i])

        y_true = y_true_arteir_vein_only
        nn_labels = labels_NN_artery_vein_only

        y_true_global += y_true
        nn_labels_global += nn_labels

    return f1_score(y_true_global, nn_labels_global, average="weighted")


def train_net(
    training_dataset,
    validation_set,
    norm_parameters,
    net=None,
    epochs=200,
    shuffle=False,
):
    if net is None:
        net = UNet(num_classes=4)
        net = net.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    train_loader = DataLoader(training_dataset, batch_size=1, shuffle=shuffle)
    test_loader = DataLoader(validation_set, batch_size=1, shuffle=False)

    training_losses_epochs = []
    test_losses_epochs = []
    min_test_loss = None
    min_test_loss_file_name = None
    min_training_loss = None

    test_f1_epochs = []
    max_test_f1_file_name = None
    print(
        f"Numero immagini di training {training_dataset.__len__()} validation {validation_set.__len__()}"
    )
    for epoch in range(epochs):
        start_time = time.time()
        training_epoch_loss = 0.0
        net.train()
        for i, batch_data in enumerate(train_loader):
            (images, labels, coord_disk) = batch_data

            if NORMALIZE_INPUT:
                images = normalize_images(images, norm_parameters)

            images = images.permute(0, 3, 1, 2)

            optimizer.zero_grad()

            output = net(images.to(device))

            output = output.permute(0, 2, 3, 1).view(-1, 4)
            labels = labels.view(-1)

            loss = loss_function(output, labels.to(device).long())
            loss.backward()
            optimizer.step()

            training_epoch_loss += loss.item()

        last_training_loss = training_epoch_loss / training_dataset.__len__()
        training_losses_epochs.append(last_training_loss)

        print(
            "Epoch {}/{} TRAINING LOSS {:.6f}".format(
                epoch + 1, epochs, last_training_loss
            )
        )

        # validazione sul test set
        net.eval()
        with torch.no_grad():
            test_epoch_loss = 0.0
            for i, batch_data in enumerate(test_loader):
                (images, labels, coord_disk) = batch_data

                if NORMALIZE_INPUT:
                    images = normalize_images(images, norm_parameters)

                images = images.permute(0, 3, 1, 2)

                output = net(images.to(device))
                output = output.permute(0, 2, 3, 1).view(-1, 4)
                labels = labels.view(-1)

                loss = loss_function(output, labels.to(device).long())

                test_epoch_loss += loss.item()

            last_test_loss = test_epoch_loss / validation_set.__len__()
            test_losses_epochs.append(last_test_loss)

            print("VALIDATION LOSS {:.6f}".format(last_test_loss))

            # salva il modello con la test loss più bassa
            if min_test_loss is None:
                min_test_loss = last_test_loss
            elif last_test_loss < min_test_loss:
                min_test_loss = last_test_loss
                min_test_loss_file_name = (
                    "./models/model_validation_loss_{:.3f}".format(last_test_loss)
                    + ".pt"
                )

                torch.save(net.state_dict(), min_test_loss_file_name)

            # calcolo F1 sul test set
            f1_score = get_f1_on_dataset(net, validation_set, norm_parameters)

            # salva il modello con la F1 più alta
            if f1_score > max(test_f1_epochs, default=0.0):
                max_test_f1_file_name = (
                    "./models/model_f1_{:.4f}".format(f1_score) + ".pt"
                )
                print(f"Saving model {max_test_f1_file_name}")
                torch.save(net.state_dict(), max_test_f1_file_name)

            test_f1_epochs.append(f1_score)
            print("F1 SCORE {:.6f}".format(f1_score))

        end_time = time.time()
        print("Tempo Epoca: {:0.2f} s\n".format(end_time - start_time))

    # salva training e test loss
    np.save("training_losses.npy", training_losses_epochs)
    np.save("validation_losses.npy", test_losses_epochs)

    # visualizza plot training e test loss
    df = pd.DataFrame(
        data=training_losses_epochs,
        index=[i for i in range(len(training_losses_epochs))],
        columns=["training loss"],
    )
    df.insert(1, "validation loss", test_losses_epochs)
    df.insert(2, "F1 score", test_f1_epochs)

    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    sns.lineplot(data=df, ax=axs)
    axs.set_ylabel("Loss")
    axs.set_xlabel("Epochs")
    fig.tight_layout()
    plt.show()

    # prendi la rete con la test loss più bassa
    net = UNet(num_classes=4)
    net.load_state_dict(torch.load(max_test_f1_file_name))
    print(f"Rete caricata: {max_test_f1_file_name}")

    return net.to(device)


def visualize_output_net(output, labels):
    probs = nn.Softmax(1)(output)

    output_labels = torch.argmax(probs, dim=1)

    # print(labels[0])
    # print(output_labels[0].cpu().numpy())
    start = time.time()

    image = parallel_get_colored_img_from_labels(output_labels[0].cpu().numpy())
    label_image = parallel_get_colored_img_from_labels(labels[0].cpu().numpy())

    end = time.time()
    print(end - start)

    show_images_single_row([image, label_image])


def correction_graph_pipeline(nn_probs, coord_disk, labels_gt):
    # prende l'output della rete e fornisce le etichette per ogni pixel corretto dall'euristica
    print("-- Avvio pipeline grafo su output rete --")
    print("Generazione immagine bianco e nero...")
    black_white_img = generate_black_white_img_from_probs(nn_probs)
    # show_images_single_row([black_white_img, black_white_img])

    print("Generazione grafo...")
    graph_builder = GraphBuilder(cache_graph=False)
    graph = graph_builder.generate_graph_from_black_white_img(
        black_white_img, coord_disk
    )

    print("Generazione dati sui nodi...")
    data_nodes = graph_builder.get_nodes_with_data()

    print("-- Generazione dati sugli archi... --")
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

    print("-- Applicazioni correzioni --")
    euristic = Euristic(data_edges, data_nodes, graph_builder)

    print("Assegnazione orientamento archi...")
    euristic.assign_orientation()

    euristic.apply_corrections()

    print("-- Generazione immagine finale corretta --")
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

    print("Visualizzazione immagini vanilla vs corretta...")
    show_images_single_row([img_NN, img_EU])

    print("Ottenimento etichette da immagine corretta...")
    labels_EU = get_labels_from_colored_img(img_EU)

    return labels_EU


def train_net_with_corrections(CD_dataset, net, norm_parameters, corrections=True):
    print("Avvio addestramento su CD...")

    print("Generazione etichette corrette...")
    idx = 0
    for input_img, labels, coord_disk in CD_dataset:
        print(f"Ottengo output rete su immagine {idx} CD")

        input_img = torch.from_numpy(input_img)
        input_img = input_img.unsqueeze(0)

        if NORMALIZE_INPUT:
            input_img = normalize_images(input_img, norm_parameters)

        input_img = input_img.permute(0, 3, 1, 2)
        input_img = input_img.to(device)

        nn_output = net(input_img)
        nn_probs = torch.nn.Softmax(1)(nn_output)

        new_labels = torch.argmax(nn_probs, dim=1).permute(1, 2, 0).cpu().numpy()
        if corrections:
            new_labels = correction_graph_pipeline(nn_probs, coord_disk, labels)

        print("Salvo le etichette per questa immagine...")
        CD_dataset.set_label(idx, new_labels)

        idx += 1

    print("---- Avvio training su CD ----")
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    train_loader = DataLoader(CD_dataset, batch_size=1, shuffle=False)

    training_losses_epochs = []
    min_training_loss = None

    for epoch in range(5):
        training_epoch_loss = 0.0
        net.train()
        for i, batch_data in enumerate(train_loader):
            (images, labels, coord_disk) = batch_data

            if NORMALIZE_INPUT:
                images = normalize_images(images, norm_parameters)

            images = images.permute(0, 3, 1, 2)

            optimizer.zero_grad()

            output = net(images.to(device))

            output = output.permute(0, 2, 3, 1).view(-1, 4)
            labels = labels.view(-1)

            loss = loss_function(output, labels.to(device).long())
            loss.backward()
            optimizer.step()

            training_epoch_loss += loss.item()

        last_training_loss = training_epoch_loss / CD_dataset.__len__()
        training_losses_epochs.append(last_training_loss)

        print("Epoch {} TRAINING LOSS {:.6f}".format(epoch, last_training_loss))

    # visualizza plot training loss
    df = pd.DataFrame(
        data=training_losses_epochs,
        index=[i for i in range(len(training_losses_epochs))],
        columns=["training loss"],
    )

    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    sns.lineplot(data=df, ax=axs)
    axs.set_ylabel("Loss")
    axs.set_xlabel("Epochs")
    fig.tight_layout()
    plt.show()

    return net


def train_net_with_corrections_plus_LD(
    CD_dataset, net, norm_parameters, training_dataset, validation_set, corrections=True
):
    print("Avvio addestramento su CD...")

    print("Generazione etichette corrette...")
    idx = 0
    for input_img, labels, coord_disk in CD_dataset:
        print(f"Ottengo output rete su immagine {idx} CD")

        input_img = torch.from_numpy(input_img)
        input_img = input_img.unsqueeze(0)

        if NORMALIZE_INPUT:
            input_img = normalize_images(input_img, norm_parameters)

        input_img = input_img.permute(0, 3, 1, 2)
        input_img = input_img.to(device)

        nn_output = net(input_img)
        nn_probs = torch.nn.Softmax(1)(nn_output)

        new_labels = torch.argmax(nn_probs, dim=1).permute(1, 2, 0).cpu().numpy()
        if corrections:
            new_labels = correction_graph_pipeline(nn_probs, coord_disk, labels)

        print("Salvo le etichette per questa immagine...")
        CD_dataset.set_label(idx, new_labels)

        idx += 1

    print("Merging TR + CD")
    TR = training_dataset.merge(CD_dataset)
    net = train_net(
        TR, validation_set, norm_parameters, net=net, epochs=50, shuffle=True
    )

    return net


def show_metrics_on_TD(net, test_dataset, norm_parameters, apply_correction=False):
    # contiene le metriche calcolate sulle singole immagini
    metrics_imgs = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    colored_imgs = []
    colored_labels_imgs = []

    # per ogni immagine estraggo le classi fornite dalla rete e le etichette per ogni pixel
    for input_img, labels, coord_disk in test_dataset:
        input_img = torch.from_numpy(input_img)
        images = input_img.unsqueeze(0)

        # normalizzazione immagini in input
        if NORMALIZE_INPUT:
            images = normalize_images(images, norm_parameters)

        images = images.permute(0, 3, 1, 2)
        images = images.to(device)
        nn_output = net(images)

        nn_probs = torch.nn.Softmax(1)(nn_output)
        nn_labels = torch.argmax(nn_probs, dim=1).squeeze().cpu().numpy()

        if apply_correction:
            nn_labels = correction_graph_pipeline(nn_probs, coord_disk, labels)

        nn_img = get_colored_img_from_labels(nn_labels)
        label_img = get_colored_img_from_labels(labels)
        # show_images_single_row([nn_img, label_img])

        colored_imgs.append(nn_img)
        colored_labels_imgs.append(label_img)

        print(f"Calcolo metriche per immagine {len(metrics_imgs['accuracy'])}")
        accuracy, precision, recall, f1 = get_metrics(labels, nn_labels)

        metrics_imgs["accuracy"].append(accuracy)
        metrics_imgs["precision"].append(precision)
        metrics_imgs["recall"].append(recall)
        metrics_imgs["f1"].append(f1)

    show_big_plots_images([colored_imgs, colored_labels_imgs])

    # calcolo metriche
    return (colored_imgs, metrics_imgs)

