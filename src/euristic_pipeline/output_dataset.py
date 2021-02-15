import sys

sys.path.append("../")
from dataset import DatasetRetinalImagesRITE


class OutputDataset(DatasetRetinalImagesRITE):
    def __init__(self):
        pass

    def initialize(self, x_folder_path, x_images_format, y_folder_path, y_images_format, coord_disks_file):
        super().initialize(x_folder_path, x_images_format, y_folder_path, y_images_format, coord_disks_file)

        self.nn_outputs = {}
        self.graphs = {}
        self.skels = {}
        self.black_white_imgs = {}

    def set_nn_output(self, index, nn_output):
        self.nn_outputs[index] = nn_output

    def set_graph(self, index, graph):
        self.graphs[index] = graph

    def set_skel(self, index, skel):
        self.skels[index] = skel

    def set_black_white_img(self, index, black_white_img):
        self.black_white_imgs[index] = black_white_img

    def __getitem__(self, index):
        nn_output = self.nn_outputs[index]
        graph = self.graphs[index]
        skel = self.skels[index]
        black_white_img = self.black_white_imgs[index]

        upper_item = super().__getitem__(index)

        return (*upper_item, nn_output, graph, skel, black_white_img)

