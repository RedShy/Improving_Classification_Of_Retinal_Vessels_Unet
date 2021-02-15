import torch
import math
import constants


def get_probabilities_for_edges(edges, nn_probs):
    n_classes = nn_probs.shape[1]
    n_edges = len(edges)
    probabilities_edges = torch.zeros(n_edges, n_classes)

    for edge in edges.values():
        mean_probs = get_mean_probs(cells=edge.coord_pixels, nn_probs=nn_probs)

        for class_ in range(n_classes):
            probabilities_edges[edge.ID][class_] = mean_probs[class_]
            edge.probabilites_class[class_] = mean_probs[class_].item()

    return probabilities_edges


def get_mean_probs(cells, nn_probs):
    # sommiamo le probabilità di tutte le celle per ogni classe
    sum_probs = torch.zeros(nn_probs.shape[1])

    for class_idx in range(len(sum_probs)):
        # carico nella CPU la matrice corrispondente a questa classe
        matrix_class = nn_probs[0][class_idx].cpu()

        # sommo le probabilità delle celle e le inserisco nel contenitore delle somme
        for cell in cells:
            sum_probs[class_idx] += matrix_class[cell[0]][cell[1]]

    # divido ogni probabilità per il numero totale di celle "buone" visitate
    mean_probs = torch.zeros(sum_probs.shape[0])
    for p_idx in range(len(sum_probs)):
        mean_probs[p_idx] = sum_probs[p_idx] / len(cells)

        # TODO: eliminare
        if mean_probs[p_idx] > 1 or math.isnan(mean_probs[p_idx]):
            print("ERRORE!!!!!")
            print("Classe: {} Somma probs: {} good_cells: {}".format(p_idx, sum_probs[p_idx], len(cells)))
            exit(0)

    return mean_probs


def assign_NN_class_to_edges(data_edges, probabilities_edges):
    for edge in data_edges.values():
        prob_artery = probabilities_edges[edge.ID][constants.ARTERY_CLASS]
        prob_vein = probabilities_edges[edge.ID][constants.VEIN_CLASS]

        max_class = constants.ARTERY_CLASS if prob_artery > prob_vein else constants.VEIN_CLASS

        edge.class_nn = max_class
        edge.class_EU = edge.class_nn

    return data_edges


def assign_GT_class_to_edges(edges, label_img):
    # per ogni arco, vado ad ispezionare i pixel nell'etichetta ed estraggo la classe più alta
    for edge in edges.values():
        edge.class_GT = get_major_class_for_label_pixels(edge.skel_coord_pixels, label_img)

    return edges


def get_major_class_for_label_pixels(skel_coord_pixels, colored_label_img):
    artery_pixels = 0
    vein_pixels = 0
    range_ = range(2, len(skel_coord_pixels) - 2) if len(skel_coord_pixels) > 4 else range(2, len(skel_coord_pixels) - 2)
    for idx in range_:
        cell = skel_coord_pixels[idx]
        rgb_color = [colored_label_img[cell[0]][cell[1]][0], colored_label_img[cell[0]][cell[1]][1], colored_label_img[cell[0]][cell[1]][2]]
        if rgb_color == constants.ARTERY_RGB_COLOR:
            artery_pixels += 1
        elif rgb_color == constants.VEIN_RGB_COLOR:
            vein_pixels += 1

    if artery_pixels > vein_pixels:
        return constants.ARTERY_CLASS

    return constants.VEIN_CLASS
