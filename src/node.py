class Node(object):
    def __init__(self, ID, id_duplicate_node_4_degree=None, id_duplicate_node_3_degree=None):
        self.ID = ID
        self.id_duplicate_node_4_degree = id_duplicate_node_4_degree
        self.fake_cross_over = None
        self.easy_cross_over = False

        self.id_duplicate_node_3_degree = id_duplicate_node_3_degree

        self.periferical_triplet = False
