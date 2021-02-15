import pickle
import numpy as np
from PIL import Image
import glob
import re
from torch.utils.data import Dataset


class DatasetRetinalImagesRITE(Dataset):
    def __init__(self):
        pass
        # if start != -1 and end != -1:
        #     new_x_data = np.zeros(shape=(end - start, self.x_data.shape[1], self.x_data.shape[2], self.x_data.shape[3]), dtype=self.x_data.dtype)
        #     new_y_data = np.zeros(shape=(end - start, self.y_data.shape[1], self.y_data.shape[2], self.y_data.shape[3]), dtype=self.y_data.dtype)
        #     new_coord_disks = {}

        #     for idx, i in enumerate(range(start, end)):
        #         new_x_data[idx] = self.x_data[i]
        #         new_y_data[idx] = self.y_data[i]
        #         new_coord_disks[idx] = self.coord_disks[i]

        #     self.x_data = new_x_data
        #     self.y_data = new_y_data
        #     self.coord_disks = new_coord_disks

    def initialize(self, x_folder_path, x_images_format, y_folder_path, y_images_format, coord_disks_file):
        self.load_input_images(x_folder_path, x_images_format)
        self.load_labels(y_folder_path, y_images_format)
        self.load_coord_disks(coord_disks_file)
        assert (
            self.x_data[0].shape[0] == self.y_data[0].shape[0] and self.x_data[0].shape[1] == self.y_data[0].shape[1]
        ), "Error: shape of input images are different of label images shapes"

    def load_input_images(self, images_path, images_format):
        images = self.load_images_in_folder(images_path, images_format)
        self.x_data = images.astype("float32")

    def load_labels(self, labels_path, labels_format):
        def get_labels_from_images(images):
            labels = np.zeros(shape=(images.shape[0], images.shape[1], images.shape[2], 1), dtype="uint8")
            for img in range(images.shape[0]):
                print("Generating label image number... ", img)
                for i in range(images.shape[1]):
                    for j in range(images.shape[2]):
                        r = images[img][i][j][0]
                        g = images[img][i][j][1]
                        b = images[img][i][j][2]
                        if r == 255 and g == 0 and b == 0:
                            labels[img][i][j][0] = 1
                        elif r == 0 and g == 255 and b == 0:
                            labels[img][i][j][0] = 1
                        elif r == 0 and g == 0 and b == 255:
                            labels[img][i][j][0] = 2
                        elif r == 0 and g == 0 and b == 0:
                            labels[img][i][j][0] = 0
                        else:
                            labels[img][i][j][0] = 0
            return labels

        with open(labels_path + "labels.pkl", "rb") as fp:
            labels = pickle.load(fp)
            self.y_data = labels

        # labels_images = self.load_images_in_folder(labels_path, labels_format)
        # self.y_data = get_labels_from_images(labels_images)

        # with open(labels_path + "labels.pkl", "wb") as fp:
        #     pickle.dump(self.y_data, fp)

    def load_images_in_folder(self, images_path, images_format):
        # prendo i nomi dei file su disco
        images_files = []
        for img in sorted(glob.glob(images_path + "*." + images_format)):
            images_files.append(img)
        assert len(images_files) > 0, "Error: no images in the folder"

        # carico effettivamente le immagini dai nomi dei file
        shape_images = np.array(Image.open(images_files[0])).shape
        images = np.zeros(shape=(len(images_files), shape_images[0], shape_images[1], shape_images[2]), dtype="uint8")
        for i, img_file in enumerate(images_files):
            print("getting input image: ", img_file)
            img = Image.open(img_file)
            images[i] = np.array(img)

        return images

    def load_coord_disks(self, coord_disks_file):
        f = open(coord_disks_file, "r")
        self.coord_disks = {}

        lines = f.read().splitlines()
        for idx, line in enumerate(lines):
            match = re.match("(\d+) (\d+) (\d+)", line)
            assert match, f"Errore nel caricamento delle coordinate del disco ottico, file: {coord_disks_file}"

            r = int(match.group(1))
            c = int(match.group(2))
            radius = int(match.group(3))

            self.coord_disks[idx] = (r, c, radius)

            # print(dict_coord[idx])

        f.close()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        coord_disk = self.coord_disks[index]
        return x, y, coord_disk

    def set_label(self, index, y):
        self.y_data[index] = y

    def get_subset(self, start, end, start2=None, end2=None):
        assert start >= 0 and end > 0, "start e end < 0!!!!"
        assert start < end, "start >= end!!!!!"

        length = end - start
        if start2 is not None:
            length += end2 - start2

        subset = DatasetRetinalImagesRITE()

        subset.x_data = np.zeros(shape=(length, self.x_data.shape[1], self.x_data.shape[2], self.x_data.shape[3]), dtype=self.x_data.dtype)
        subset.y_data = np.zeros(shape=(length, self.y_data.shape[1], self.y_data.shape[2], self.y_data.shape[3]), dtype=self.y_data.dtype)
        subset.coord_disks = {}
        idx = 0
        for i in range(start, end):
            subset.x_data[idx] = self.x_data[i]
            subset.y_data[idx] = self.y_data[i]
            subset.coord_disks[idx] = self.coord_disks[i]

            idx += 1

        if start2 is not None:
            for i in range(start2, end2):
                subset.x_data[idx] = self.x_data[i]
                subset.y_data[idx] = self.y_data[i]
                subset.coord_disks[idx] = self.coord_disks[i]

                idx += 1

        return subset

    def merge(self, other_dataset):
        merged = DatasetRetinalImagesRITE()
        length = self.__len__() + other_dataset.__len__()

        merged.x_data = np.zeros(shape=(length, self.x_data.shape[1], self.x_data.shape[2], self.x_data.shape[3]), dtype=self.x_data.dtype)
        merged.y_data = np.zeros(shape=(length, self.y_data.shape[1], self.y_data.shape[2], self.y_data.shape[3]), dtype=self.y_data.dtype)
        merged.coord_disks = {}
        idx = 0
        for i in range(self.__len__()):
            merged.x_data[idx] = self.x_data[i]
            merged.y_data[idx] = self.y_data[i]
            merged.coord_disks[idx] = self.coord_disks[i]

            idx += 1

        for i in range(other_dataset.__len__()):
            merged.x_data[idx] = other_dataset.x_data[i]
            merged.y_data[idx] = other_dataset.y_data[i]
            merged.coord_disks[idx] = other_dataset.coord_disks[i]

            idx += 1

        return merged

    def get_mean_channels(self):
        return np.mean(self.x_data, axis=(0, 1, 2), keepdims=True)

    def get_std_channels(self):
        return np.std(self.x_data, axis=(0, 1, 2), keepdims=True)

    def get_max_channels(self):
        return np.max(self.x_data, axis=(0, 1, 2), keepdims=True)
