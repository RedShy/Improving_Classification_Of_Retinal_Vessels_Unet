import math

# angolo tra due linee
def angle_vectors(lineA, lineB):
    # prodotto scalare tra due vettori di 2 elementi
    def dot(vA, vB):
        return vA[0] * vB[0] + vA[1] * vB[1]

    # voglio che entrambe le linee siano nella forma [punto_comune, punto_non_comune]
    for point in lineA:
        if point in lineB:
            # prendo l'indice del punto non in comune
            other_idxA = (lineA.index(point) + 1) % 2
            other_idxB = (lineB.index(point) + 1) % 2

            lineA_tmp = [point, lineA[other_idxA]]
            lineB_tmp = [point, lineB[other_idxB]]

            lineA = lineA_tmp
            lineB = lineB_tmp
            break

    # trasforma le linee in vettori V = [x1-x2, y1-y2]
    vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
    vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]

    # fai prodotto scalare
    dot_prod = dot(vA, vB)

    # ottieni le magnitudine dei vettori
    magA = dot(vA, vA) ** 0.5
    magB = dot(vB, vB) ** 0.5

    # ottieni il valore del coseno
    cos_ = dot_prod / (magA * magB)

    # corregge problemi di rappresentazione decimali
    if cos_ > 1.0:
        cos_ = 1.0
    elif cos_ < -1.0:
        cos_ = -1.0

    # fai l'arcocoseno
    angle = math.acos(cos_)

    # faccio % 360 per avere un angolo tra 0 e 359
    ang_deg = math.degrees(angle) % 360

    # if ang_deg > 180:
    #     # ottengo un angolo tra 0 e 179
    #     return 360 - ang_deg
    # else:
    if math.isnan(ang_deg):
        ang_deg = 0
    return ang_deg


def euclide_distance(n1_coord, n2_coord):
    dis_x = pow(n1_coord[0] - n2_coord[0], 2)
    dis_y = pow(n1_coord[1] - n2_coord[1], 2)

    return math.sqrt(dis_x + dis_y)


def is_in_quadrilateral(coords_quadrilateral, coord):
    x1 = coords_quadrilateral[0][1]
    y1 = coords_quadrilateral[0][0]

    x2 = coords_quadrilateral[2][1]
    y2 = coords_quadrilateral[2][0]

    x = coord[1]
    y = coord[0]

    return x >= x1 and x <= x2 and y >= y1 and y <= y2
