import numpy as np
import networkx as nx
import torch
import constants
from IPython.display import display
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from constants import TypeClassification


def show_directed_graph(DG, with_labels=False):
    # visualizzo il grafo
    fig, axs = plt.subplots(1, 1, figsize=(5, 20))
    options = {
        "node_color": "black",
        "node_size": 25,
        "width": 1,
        "arrowstyle": "-|>",
        "arrowsize": 10,
        "with_labels": with_labels,
    }
    nx.draw_networkx(
        DG, pos=nx.get_node_attributes(DG, "coordinates"), ax=axs, **options
    )
    plt.draw()
    plt.show()


def show_edges(
    edges,
    nodes=True,
    structure=True,
    img=None,
    show=True,
    highlight=True,
    type_class=TypeClassification.GT,
):
    if img is None:
        img = get_blank_rgb_image()

    for idx, edge in enumerate(edges):
        if highlight and edge.highlight:
            rgb_edge = [255, 255, 0]
        else:
            rgb_edge = get_rgb_color_from_class(edge.get_class(type_class))
        colorize_cells(rgb_edge, edge.coord_pixels, img)

    if structure:
        for idx, edge in enumerate(edges):
            rgb_structure = [255, 255, 255]
            colorize_cells(rgb_structure, edge.skel_coord_pixels, img)

    if nodes:
        for idx, edge in enumerate(edges):
            rgb_node = [0, 255, 0]
            colorize_cells(rgb_node, get_matrix_cells(edge.n1_coord, 2), img)
            colorize_cells(rgb_node, get_matrix_cells(edge.n2_coord, 2), img)

    if show:
        show_image(img)

    return img


def show_edges_good_bad(
    all_edges,
    good_edges,
    bad_edges,
    nodes=True,
    structure=True,
    img=None,
    show=True,
    highlight=True,
    type_class=TypeClassification.GT,
):
    if img is None:
        img = get_blank_rgb_image()

    for idx, edge in enumerate(all_edges):
        rgb_edge = get_rgb_color_from_class(edge.get_class(type_class))
        colorize_cells(rgb_edge, edge.coord_pixels, img)

    for idx, edge in enumerate(good_edges):
        rgb_edge = [0, 255, 0]
        colorize_cells(rgb_edge, edge.coord_pixels, img)

    for idx, edge in enumerate(bad_edges):
        rgb_edge = [255, 255, 0]
        colorize_cells(rgb_edge, edge.coord_pixels, img)

    if structure:
        for idx, edge in enumerate(all_edges):
            rgb_structure = [255, 255, 255]
            colorize_cells(rgb_structure, edge.skel_coord_pixels, img)

    if nodes:
        for idx, edge in enumerate(all_edges):
            rgb_node = [0, 255, 0]
            colorize_cells(rgb_node, get_matrix_cells(edge.n1_coord, 2), img)
            colorize_cells(rgb_node, get_matrix_cells(edge.n2_coord, 2), img)

    if show:
        show_image(img)
    else:
        return img


def show_multiple_edges_tmp1(edges, with_class=False, img=None):
    if img is None:
        img = get_blank_rgb_image()

    for idx, edge in enumerate(edges):
        rgb_edge = [255, 255, 255]
        # if with_class:
        #     rgb_edge = get_rgb_color_from_class(edge.class_nn)
        # colorize_cells(rgb_edge, edge.coord_pixels, img)

        # shifted = [(cell[0] + 260, cell[1]) for cell in edge.coord_pixels]
        rgb_edge = get_rgb_color_from_class(edge.class_GT)
        colorize_cells(rgb_edge, edge.coord_pixels, img)
        # colorize_cells(rgb_edge, shifted, img)

        # rgb_node = [0, 255, 0]
        # colorize_cells(rgb_node, get_matrix_cells(edge.n1_coord, 2), img)
        # colorize_cells(rgb_node, get_matrix_cells(edge.n2_coord, 2), img)

    show_image(img)


def show_angle_edges_tmp(edges, img=None):
    if img is None:
        img = get_blank_rgb_image()

    for idx, edge in enumerate(edges):
        rgb_edge = [255, 255, 255]

        colorize_cells(rgb_edge, edge.skel_coord_pixels, img)

    show_image(img)


def get_blank_rgb_image(rows=584, columns=565):
    return np.zeros(shape=(rows, columns, 3), dtype="uint8")


def show_single_edge(edge):
    img = get_blank_rgb_image()

    rgb_edge = [255, 255, 255]
    colorize_cells(rgb_edge, edge.coord_pixels, img)

    rgb_node = [0, 255, 0]
    colorize_cells(rgb_node, get_matrix_cells(edge.n1_coord, 2), img)
    colorize_cells(rgb_node, get_matrix_cells(edge.n2_coord, 2), img)

    show_image(img)


def show_multiple_edges(edges, with_class=False, img=None):
    if img is None:
        img = get_blank_rgb_image()

    for idx, edge in enumerate(edges):
        rgb_edge = [255, 255, 255]
        # if with_class:
        #     rgb_edge = get_rgb_color_from_class(edge.class_nn)
        colorize_cells(rgb_edge, edge.coord_pixels, img)

        shifted = [(cell[0] + 260, cell[1]) for cell in edge.coord_pixels]
        rgb_edge = get_rgb_color_from_class(edge.class_GT)
        colorize_cells(rgb_edge, shifted, img)

        rgb_node = [0, 255, 0]
        colorize_cells(rgb_node, get_matrix_cells(edge.n1_coord, 2), img)
        colorize_cells(rgb_node, get_matrix_cells(edge.n2_coord, 2), img)

    show_image(img)


def show_cells(cells, img=None):
    if img is None:
        img = get_blank_rgb_image()

    colorize_cells([255, 255, 255], cells, img)
    colorize_cells([255, 0, 0], [cells[0]], img)

    show_image(img)


def colorize_cells(rgb_color, cells, img):
    for cell in cells:
        r = cell[0]
        c = cell[1]

        if r >= 0 and r < img.shape[0] and c >= 0 and c < img.shape[1]:
            img[r][c][0] = rgb_color[0]
            img[r][c][1] = rgb_color[1]
            img[r][c][2] = rgb_color[2]

    return img


def get_rgb_from_black_white(black_white_img):
    rgb_img = np.zeros(
        shape=(black_white_img.shape[0], black_white_img.shape[1], 3), dtype="uint8"
    )
    for x in range(black_white_img.shape[0]):
        for y in range(black_white_img.shape[1]):
            if black_white_img[x][y] > 0:
                rgb_img[x][y][0] = 255
                rgb_img[x][y][1] = 255
                rgb_img[x][y][2] = 255

    return rgb_img


def show_image(img, save=False, name_file="figure"):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.imshow(img, interpolation="nearest")
    axs.axis("off")
    fig.tight_layout()
    if save:
        plt.savefig(f"./images/{name_file}.png", bbox_inches="tight")
    plt.show()


def show_graph(graph, height_img, save=False):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    dict_nodes_coordinates_rotated = {}
    dict_nodes_coordinates = nx.get_node_attributes(graph, "coordinates")
    for node in dict_nodes_coordinates.keys():
        coord = dict_nodes_coordinates[node]
        dict_nodes_coordinates_rotated[node] = (coord[1], height_img - coord[0])
    nx.draw(
        G=graph,
        pos=dict_nodes_coordinates_rotated,
        ax=axs,
        node_size=12,
        edge_color=(1.0, 1.0, 0.0),
        node_color=(0.0, 1.0, 0.0),
    )
    plt.draw()
    if save:
        plt.savefig("./graph.png", bbox_inches="tight")
    plt.show()


def get_colored_img_from_labels(labels):
    image = np.zeros(shape=(labels.shape[0], labels.shape[1], 3), dtype="uint8")
    red_idx = 0
    green_idx = 1
    blue_idx = 2

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] == 0:
                image[i][j][red_idx] = 0
                image[i][j][green_idx] = 0
                image[i][j][blue_idx] = 0
            elif labels[i][j] == 1:
                image[i][j][red_idx] = 255
                image[i][j][green_idx] = 0
                image[i][j][blue_idx] = 0
            elif labels[i][j] == 2:
                image[i][j][red_idx] = 0
                image[i][j][green_idx] = 0
                image[i][j][blue_idx] = 255
            elif labels[i][j] == 3:
                image[i][j][red_idx] = 255
                image[i][j][green_idx] = 255
                image[i][j][blue_idx] = 255

    return image


def parallel_get_colored_img_from_labels(labels):
    def color_label(start_row, increment):
        end_row = start_row + increment
        # print(f"Eseguo colorize_prov  srow{start_row} erow{end_row}")

        red_idx = 0
        green_idx = 1
        blue_idx = 2
        for i in range(start_row, end_row):
            for j in range(labels.shape[1]):
                if labels[i][j] == 0:
                    image[i][j][red_idx] = 0
                    image[i][j][green_idx] = 0
                    image[i][j][blue_idx] = 0
                elif labels[i][j] == 1:
                    image[i][j][red_idx] = 255
                    image[i][j][green_idx] = 0
                    image[i][j][blue_idx] = 0
                elif labels[i][j] == 2:
                    image[i][j][red_idx] = 0
                    image[i][j][green_idx] = 0
                    image[i][j][blue_idx] = 255
                elif labels[i][j] == 3:
                    image[i][j][red_idx] = 255
                    image[i][j][green_idx] = 255
                    image[i][j][blue_idx] = 255

    image = np.zeros(shape=(labels.shape[0], labels.shape[1], 3), dtype="uint8")

    num_threads = 8
    increment = int(labels.shape[0] / num_threads)
    start = 0
    # ogni thread prende un certo numero di righe, prende image e lo va a modificare
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            future = executor.submit(color_label, start, increment)

            start += increment

    return image


def show_pandas_dataframe(df):
    def highlight_number(column):
        tmp = []
        for idx, cell in enumerate(column):
            if idx < 2:
                tmp.append("background-color: black; color: white")
            elif cell >= 0:
                tmp.append("background-color: green; color: white")
            elif cell < 0:
                tmp.append("background-color: red; color: white")
        return tmp

    s = df.style.format("{:.5f}")
    s.apply(highlight_number, axis=1)
    display(s)
    print(df)


def get_matrix_nine_cells(cell):
    cells = []

    for i in range(-1, 1):
        for j in range(-1, 1):
            cells.append((cell[0] + i, cell[1] + j))

    return cells


def get_matrix_cells(cell, dim=3):
    cells = []

    for i in range(-dim, dim):
        for j in range(-dim, dim):
            cells.append((cell[0] + i, cell[1] + j))

    return cells


def get_rgb_color_from_class(class_):
    if class_ == constants.ARTERY_CLASS:
        return constants.ARTERY_RGB_COLOR
    elif class_ == constants.VEIN_CLASS:
        return constants.VEIN_RGB_COLOR
    elif class_ == constants.BACKGROUND_CLASS:
        return constants.BACKGROUND_RGB_COLOR
    elif class_ == constants.UNCERTAINTY_CLASS:
        return constants.UNCERTAINTY_RGB_COLOR

    assert False, f"Non c'è un colore per la classe {class_}"


def get_class_from_rgb_color(color):
    if color == constants.BACKGROUND_RGB_COLOR:
        return constants.BACKGROUND_CLASS
    elif color == constants.ARTERY_RGB_COLOR:
        return constants.ARTERY_CLASS
    elif color == constants.VEIN_RGB_COLOR:
        return constants.VEIN_CLASS
    elif color == constants.UNCERTAINTY_RGB_COLOR:
        return constants.UNCERTAINTY_CLASS

    assert False, f"Non c'è una classe per il colore {color}"


def show_images_single_row(images, file_name=None):
    columns = len(images)
    rows = 1
    fig, axs = plt.subplots(rows, columns, figsize=(10 * columns, 10 * rows))
    for idx, image in enumerate(images):
        axs[idx].imshow(image, interpolation="nearest")
        axs[idx].axis("off")
    fig.tight_layout()
    if file_name is not None:
        try:
            plt.savefig("../figures/" + file_name + ".png", bbox_inches="tight")
        except FileNotFoundError:
            plt.savefig("./" + file_name + ".png", bbox_inches="tight")
    plt.show()


def show_big_plots_images(list_of_list_images, split=False):
    if split:
        columns = len(list_of_list_images)
        rows = 5
        outer_counter = 0
        for idx_plot in range(int(len(list_of_list_images[0]) / rows)):
            print(f"Immagini {outer_counter} - {outer_counter + rows - 1}")
            fig, axs = plt.subplots(rows, columns, figsize=(columns * 10, rows * 10))
            for row in range(rows):
                for column in range(columns):
                    axs[row][column].imshow(
                        list_of_list_images[column][outer_counter],
                        interpolation="nearest",
                    )
                    axs[row][column].axis("off")

                outer_counter += 1
            fig.tight_layout()
            plt.show()
    else:
        columns = len(list_of_list_images)
        rows = len(list_of_list_images[0])
        fig, axs = plt.subplots(rows, columns, figsize=(columns * 10, rows * 10))
        for row in range(rows):
            for column in range(columns):
                axs[row][column].imshow(
                    list_of_list_images[column][row], interpolation="nearest"
                )
                axs[row][column].axis("off")
        fig.tight_layout()
        plt.show()

    print("----------")


def get_labels_from_colored_img(img):
    labels = np.zeros(shape=(img.shape[0], img.shape[1], 1), dtype="uint8")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            labels[i][j] = get_class_from_rgb_color(
                [img[i][j][0], img[i][j][1], img[i][j][2]]
            )

    return labels


def generate_black_white_img_from_probs(nn_probs):
    labels = torch.argmax(nn_probs, dim=1).cpu().numpy()

    image = np.zeros(shape=(labels.shape[1], labels.shape[2]), dtype="uint8")
    for i in range(labels.shape[1]):
        for j in range(labels.shape[2]):
            if labels[0][i][j] == 0:
                image[i][j] = 0
            else:
                image[i][j] = 255
    return image

    # nn_probs = nn_probs[0].cpu().numpy()
    # image = np.zeros(shape=(nn_probs.shape[1], nn_probs.shape[2]), dtype="uint8")
    # for i in range(nn_probs.shape[1]):
    #     for j in range(nn_probs.shape[2]):
    #         vascular_prob = nn_probs[1][i][j] + nn_probs[2][i][j] + nn_probs[0][i][j]
    #         if nn_probs[0][i][j] < vascular_prob:
    #             image[i][j] = 0
    #         else:
    #             image[i][j] = 255
    # return image


def get_square_inside_image(img, r, c, side_square, show=False):
    new_img = get_blank_rgb_image(side_square, side_square)

    for i in range(side_square):
        for j in range(side_square):
            new_img[i][j][0] = img[i + r][j + c][0]
            new_img[i][j][1] = img[i + r][j + c][1]
            new_img[i][j][2] = img[i + r][j + c][2]

    if show:
        show_image(new_img)

    return new_img

