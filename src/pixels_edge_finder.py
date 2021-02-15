import networkx as nx
import numpy as np
from edge import Edge
from utils_image import colorize_cells, get_matrix_nine_cells, get_matrix_cells, show_single_edge, show_cells, show_image


def get_coord_pixels_for_all_edges(edges, skeleton_img, black_white_img):
    for edge in edges.values():
        edge.coord_pixels = get_coord_pixels_for_single_edge(edge, skeleton_img, black_white_img)

    resolve_small_edges_subset_of_bigger_edges(edges, skeleton_img, black_white_img)

    return edges


def get_coord_pixels_for_single_edge(edge, skeleton_img, black_white_img):
    n1_coord = edge.n1_coord
    n2_coord = edge.n2_coord
    path = get_path_of_2_nodes(skeleton_img, n1_coord, n2_coord)

    edge.skel_coord_pixels = path
    edge.length = len(path)

    # bigger_path = dilate_path(path=path, diam=3)
    bigger_path = dilate_path_squared(path=path)

    clean_path = get_only_vessel_cells(cells=bigger_path, gray_image=black_white_img)

    assert clean_path, f"Nessuna cella trovata per il path tra il nodo {n1_coord} e {n2_coord}"

    return clean_path
    # return path
    # return get_simple_path(n1_coord, n2_coord)


def get_path_of_2_nodes(img_processed, n1_coord, n2_coord):
    # otteniamo il path dal nodo 1 al nodo 2
    # path1, error1 = get_path(img_processed, n1_coord, n2_coord)
    path1, error1 = get_path_multiple_starting_points(img_processed, n1_coord, n2_coord)

    # otteniamo il path dal nodo 2 al nodo 1
    # path2, error2 = get_path(img_processed, n2_coord, n1_coord)
    path2, error2 = get_path_multiple_starting_points(img_processed, n2_coord, n1_coord)

    # prendiamo il path che non è andato in errore
    # gli errori sono di due tipi: non è riuscito a raggiungere la destinazione oppure è andato in loop
    if error1 == 0 and error2 == 0:
        final_path = path1
        if len(path2) < len(path1):
            final_path = path2
    elif error1 == 0 and error2 == 1:
        final_path = path1
    elif error1 == 1 and error2 == 0:
        final_path = path2
    elif error1 == 1 and error2 == 1:
        # se entrambi sono in errore prendiamo il path più semplice dettato dal grafo
        final_path = get_simple_path(n1_coord, n2_coord)

    return final_path
    # return path2
    # return get_simple_path(n1_coord, n2_coord)


def get_path_multiple_starting_points(skel, n1_coord, n2_coord):
    # otteniamo i possibili punti di partenza
    # osserva le 9 celle e vedi dove puoi startare
    start = n1_coord
    starting_points = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if skel[start[0] + i][start[1] + j] > 0:
                starting_points.append((start[0] + i, start[1] + j))

    copy_skel = np.copy(skel)

    possible_paths = []
    for start_point in starting_points:
        possible_path, error = get_path(copy_skel, start_point, n2_coord)
        if error == 0:
            possible_paths.append(possible_path)
        else:
            # elimino i pixel del path sbagliato
            for cell in possible_path:
                copy_skel[cell[0]][cell[1]] = 0

    if not possible_paths:
        error = 1
        path = []
    else:
        error = 0

        # prendiamo il più piccolo tra tutti i path validi trovati
        min_index = 0
        for i in range(1, len(possible_paths)):
            if len(possible_paths[i]) < len(possible_paths[min_index]):
                min_index = i

        path = []
        # reinserisco il punto iniziale
        path.append(n1_coord)

        for cell in possible_paths[min_index]:
            path.append(cell)

    return path, error


def get_path(skel, n1_coord, n2_coord):
    path = []
    error = 0

    destination_cell = (n2_coord[0], n2_coord[1])
    current_cell = (n1_coord[0], n1_coord[1])
    last_cell = current_cell
    while current_cell != destination_cell:
        # osserva le 8 celle intorno e vedi dove puoi andare
        cells_to_go = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i != 0 or j != 0) and skel[current_cell[0] + i][current_cell[1] + j] > 0:
                    cells_to_go.append((current_cell[0] + i, current_cell[1] + j))

        # elimina la cella da cui sei partito
        for cell in cells_to_go:
            if cell == last_cell:
                cells_to_go.remove(cell)
                break

        # fermati se non ci sono celle in cui andare
        if len(cells_to_go) == 0:
            error = 1
            break

        # prendi la distanza dalla cella corrente
        min_distance_x = abs(n2_coord[0] - current_cell[0])
        min_distance_y = abs(n2_coord[1] - current_cell[1])

        # prendi la prima cella in mancanza di altro
        choosen_cell = cells_to_go[0]

        # RICERCA GREEDY
        # cerca la migliore cella possibile: quella che avvicina sia su x che su y
        best_found = False
        for c in range(0, len(cells_to_go)):
            dis_x = abs(n2_coord[0] - cells_to_go[c][0])
            dis_y = abs(n2_coord[1] - cells_to_go[c][1])

            # se c'è un miglioramento in entrambe le dimensioni prendi quella cella e basta
            if dis_x < min_distance_x and dis_y < min_distance_y:
                min_distance_x = dis_x
                min_distance_y = dis_y
                choosen_cell = cells_to_go[c]
                best_found = True
                break

        # se non hai trovato la migliore, cerca quelle che avvicinano almeno x o almeno y
        if not best_found:
            for c in range(0, len(cells_to_go)):
                dis_x = abs(n2_coord[0] - cells_to_go[c][0])
                dis_y = abs(n2_coord[1] - cells_to_go[c][1])

                # se la distanza più grande è x, prioritizzo la cella che si avvicina lungo x
                if min_distance_x > min_distance_y:
                    if dis_x < min_distance_x:
                        min_distance_x = dis_x
                        min_distance_y = dis_y
                        choosen_cell = cells_to_go[c]
                        break
                    elif dis_y < min_distance_y:
                        min_distance_x = dis_x
                        min_distance_y = dis_y
                        choosen_cell = cells_to_go[c]
                # se la distanza più grande è y, prioritizzo la cella che si avvicina lungo y
                else:
                    if dis_y < min_distance_y:
                        min_distance_x = dis_x
                        min_distance_y = dis_y
                        choosen_cell = cells_to_go[c]
                        break
                    elif dis_x < min_distance_x:
                        min_distance_x = dis_x
                        min_distance_y = dis_y
                        choosen_cell = cells_to_go[c]

        # se proprio non ha trovato nessuna cella buona si prende la cella 0

        # aggiorna la cella passata e vai nella nuova cella
        last_cell = (current_cell[0], current_cell[1])
        current_cell = (choosen_cell[0], choosen_cell[1])

        # se la cella è stata già visitata, sei in loop ed esci
        if current_cell in path:
            error = 1
            break

        # inserisci la cella nel path che stiamo percorrendo
        path.append(current_cell)

    return path, error


def dilate_path_squared(path):
    bigger_path = set()
    # escludo i primi pixel perché dilatandoli finisco per prendere sicuro pixel che non mi competono
    # range_ = range(2, len(path) - 1) if len(path) > 6 else range(1, len(path))
    range_ = range(1, len(path) - 1) if len(path) > 6 else range(1, len(path))
    for idx in range_:
        cell = path[idx]

        cells = get_matrix_cells(cell)
        for c in cells:
            bigger_path.add(c)

    return bigger_path


def dilate_path(path, diam):
    bigger_path = set()
    for c in range(0, len(path) - 1):
        current_cell = path[c]
        next_cell = path[c + 1]

        bigger_path.add(current_cell)

        # la cella successiva è a destra o a sinistra
        if current_cell[1] == next_cell[1]:
            # dilato sulla y
            for i in range(1, diam + 1):
                bigger_path.add((current_cell[0], current_cell[1] + i))
                bigger_path.add((current_cell[0], current_cell[1] - i))
        # la cella successiva è sopra o sotto
        elif current_cell[0] == next_cell[0]:
            for i in range(1, diam + 1):
                bigger_path.add((current_cell[0] + i, current_cell[1]))
                bigger_path.add((current_cell[0] - i, current_cell[1]))
        # la cella successiva è in una diagonale
        # diagonale di sopra
        elif (current_cell[1] + 1) == next_cell[1]:
            # screen[current_cell[0]][current_cell[1] + 1][high_class] = 255
            # controllo se è in diagonale a destra o a sinistra
            if (current_cell[0] + 1) == next_cell[0]:
                for i in range(1, diam):
                    bigger_path.add((current_cell[0] - i, current_cell[1] + i))
                    bigger_path.add((current_cell[0] + i, current_cell[1] - i))

                    bigger_path.add((current_cell[0] + 1 - i, current_cell[1] + i))
                    bigger_path.add((current_cell[0] + i, current_cell[1] + 1 - i))

                bigger_path.add((current_cell[0] + 1 - diam, current_cell[1] + diam))
                bigger_path.add((current_cell[0] + diam, current_cell[1] + 1 - diam))

                # screen[current_cell[0] + 1][current_cell[1]][high_class] = 255
            elif (current_cell[0] - 1) == next_cell[0]:
                # screen[current_cell[0] - 1][current_cell[1]][high_class] = 255
                for i in range(1, diam):
                    bigger_path.add((current_cell[0] + i, current_cell[1] + i))
                    bigger_path.add((current_cell[0] - i, current_cell[1] - i))

                    bigger_path.add((current_cell[0] - 1 + i, current_cell[1] + i))
                    bigger_path.add((current_cell[0] - i, current_cell[1] + 1 - i))

                bigger_path.add((current_cell[0] - 1 + diam, current_cell[1] + diam))
                bigger_path.add((current_cell[0] - diam, current_cell[1] + 1 - diam))
        # diagonale di sotto
        elif (current_cell[1] - 1) == next_cell[1]:
            # screen[current_cell[0]][current_cell[1] - 1][high_class] = 255
            # controllo se è in diagonale a destra o a sinistra
            if (current_cell[0] + 1) == next_cell[0]:
                # screen[current_cell[0] + 1][current_cell[1]][high_class] = 255
                for i in range(1, diam):
                    bigger_path.add((current_cell[0] + i, current_cell[1] + i))
                    bigger_path.add((current_cell[0] - i, current_cell[1] - i))

                    bigger_path.add((current_cell[0] + 1 - i, current_cell[1] - i))
                    bigger_path.add((current_cell[0] + i, current_cell[1] - 1 + i))

                bigger_path.add((current_cell[0] + 1 - diam, current_cell[1] - diam))
                bigger_path.add((current_cell[0] + diam, current_cell[1] - 1 + diam))
            elif (current_cell[0] - 1) == next_cell[0]:
                # screen[current_cell[0] - 1][current_cell[1]][high_class] = 255
                for i in range(1, diam):
                    bigger_path.add((current_cell[0] - i, current_cell[1] + i))
                    bigger_path.add((current_cell[0] + i, current_cell[1] - i))

                    bigger_path.add((current_cell[0] + i, current_cell[1] - 1 + i))
                    bigger_path.add((current_cell[0] - 1 + i, current_cell[1] - i))

                bigger_path.add((current_cell[0] + diam, current_cell[1] - 1 + diam))
                bigger_path.add((current_cell[0] - 1 + diam, current_cell[1] - diam))

    # if len(path) > 0:
    last_cell = path[-1]
    bigger_path.add(last_cell)

    return bigger_path


def get_simple_path(current_cell, destination_cell):
    # il contenitore dello scheletro del path tra i due vertici
    path = []
    path.append(current_cell)

    # cerchiamo di capire se andare verso destra o verso sinistra
    x_direction = 1
    if destination_cell[0] < current_cell[0]:
        x_direction = -1

    # cerchiamo di capire se andare verso sopra o verso sotto
    y_direction = 1
    if destination_cell[1] < current_cell[1]:
        y_direction = -1

    # determiniamo di quante celle muoversi lungo x e y ad ogni step
    x_tick = 1
    y_tick = 1
    distance_x = abs(destination_cell[0] - current_cell[0])
    distance_y = abs(destination_cell[1] - current_cell[1])
    if distance_y != 0:
        x_tick = int(round(distance_x / distance_y))
        if x_tick < 1:
            x_tick = 1

    if distance_x != 0:
        y_tick = int(round(distance_y / distance_x))
        if y_tick < 1:
            y_tick = 1

    while current_cell != destination_cell:
        x_current = current_cell[0]
        y_current = current_cell[1]
        # andiamo lungo y se la distanza è ancora > 0 su y
        if abs(destination_cell[1] - current_cell[1]) > 0:
            for i in range(y_tick):
                y_current += y_direction
                current_cell = (x_current, y_current)
                path.append(current_cell)
                if current_cell[1] == destination_cell[1]:
                    break

        # andiamo lungo x se la distanza è ancora > 0 su x
        if abs(destination_cell[0] - current_cell[0]) > 0:
            for i in range(x_tick):
                x_current += x_direction
                current_cell = (x_current, y_current)
                path.append(current_cell)
                if current_cell[0] == destination_cell[0]:
                    break

    return path


def get_only_vessel_cells(cells, gray_image):
    vessel_cells = []
    for cell in cells:
        x = cell[0]
        y = cell[1]

        if x < 0 or y < 0 or x >= gray_image.shape[0] or y >= gray_image.shape[1] or gray_image[x][y] == 0:
            continue

        vessel_cells.append((x, y))

    return vessel_cells


def resolve_small_edges_subset_of_bigger_edges(edges, skeleton_img, black_white_img):
    def get_all_included_edges(edge, edges):
        edges_included = []
        for e in edges.values():
            if edge.ID != e.ID:
                if edge.include_other_edge(e):
                    edges_included.append(e)

        return edges_included

    def erase_cells(cells, skel, edge):
        skel_copy = skel
        # cancello tutte queste coordinate
        for cell in cells:
            skel_copy[cell[0]][cell[1]] = 0

        # show_image(skel_copy)

        # disegno per sicurezza inizio e fine
        skel_copy[edge.n1_coord[0]][edge.n1_coord[1]] = 1
        skel_copy[edge.n2_coord[0]][edge.n2_coord[1]] = 1

    # controlla che nessun arco ha le sue coordinate interamente contenute in quelle di un altro arco
    for edge in edges.values():
        # conserva tutti gli archi quasi completamente inclusi in questo
        edges_included = get_all_included_edges(edge, edges)

        if edges_included:
            skel_copy = np.copy(skeleton_img)
            # show_image(skel_copy)

            total_included_coordinates = []
            for e_included in edges_included:
                # lascio stare la parte iniziale e finale dell'arco più piccolo
                total_included_coordinates += e_included.skel_coord_pixels[5:-5]

            erase_cells(total_included_coordinates, skel_copy, edge)

            # ricalcolo il path forzando a non usare parte delle celle usate dagli altri archi
            # show_image(skel_copy)
            edge.coord_pixels = get_coord_pixels_for_single_edge(edge, skel_copy, black_white_img)

            # show_cells(edge.skel_coord_pixels)

            # se non sono riuscito ad arrivare all'altro punto, riprovo ma eliminando tutti i pixels questa volta
            if set(edge.skel_coord_pixels) == set(get_simple_path(edge.n1_coord, edge.n2_coord)):
                total_included_coordinates = []
                for e_included in edges_included:
                    total_included_coordinates += e_included.skel_coord_pixels

                erase_cells(total_included_coordinates, skel_copy, edge)

                # ricalcolo il path forzando a non usare nessuna delle celle usate dagli altri archi
                # show_image(skel_copy)
                edge.coord_pixels = get_coord_pixels_for_single_edge(edge, skel_copy, black_white_img)

                # show_cells(edge.skel_coord_pixels)

            # se non sono riuscito ad arrivare all'altro punto, riprovo ma eliminando meno pixels questa volta
            if set(edge.skel_coord_pixels) == set(get_simple_path(edge.n1_coord, edge.n2_coord)):
                for cell in total_included_coordinates:
                    skel_copy[cell[0]][cell[1]] = 1

                total_included_coordinates = []
                for e_included in edges_included:
                    total_included_coordinates += e_included.skel_coord_pixels[9:-9]

                erase_cells(total_included_coordinates, skel_copy, edge)

                # ricalcolo il path forzando a non usare nessuna delle celle usate dagli altri archi
                # show_image(skel_copy)
                edge.coord_pixels = get_coord_pixels_for_single_edge(edge, skel_copy, black_white_img)

                # show_cells(edge.skel_coord_pixels)

            # se non sono riuscito ad arrivare all'altro punto, riprovo
            if set(edge.skel_coord_pixels) == set(get_simple_path(edge.n1_coord, edge.n2_coord)):
                for cell in total_included_coordinates:
                    skel_copy[cell[0]][cell[1]] = 1

                total_included_coordinates = []
                if len(edges_included) == 2:
                    total_included_coordinates += edges_included[0].skel_coord_pixels[8:-5]
                    total_included_coordinates += edges_included[1].skel_coord_pixels[5:-8]

                erase_cells(total_included_coordinates, skel_copy, edge)

                # ricalcolo il path forzando a non usare nessuna delle celle usate dagli altri archi
                # show_image(skel_copy)
                edge.coord_pixels = get_coord_pixels_for_single_edge(edge, skel_copy, black_white_img)

                # show_cells(edge.skel_coord_pixels)

