import constants
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


def print_confusion_matrix(img_predicted, img_truth):
    labels_predicted = get_labels_from_img(img_predicted)
    labels_truth = get_labels_from_img(img_truth)

    print(metrics.confusion_matrix(labels_truth, labels_predicted))
    print(metrics.classification_report(labels_truth, labels_predicted, digits=3))


def get_labels_from_img(img):
    labels = []
    img = np.reshape(img, (-1, 3))
    # TODO fare reshape -1,3 per vedere impatto performance
    for i in range(img.shape[0]):
        rgb_color = []
        for c in range(3):
            rgb_color.append(img[i][c])
        labels.append(get_class_from_rgb_color(rgb_color))
    return labels


def get_class_from_rgb_color(rgb_color):
    if rgb_color == constants.ARTERY_RGB_COLOR:
        return constants.ARTERY_CLASS
    elif rgb_color == constants.VEIN_RGB_COLOR:
        return constants.VEIN_CLASS
    elif rgb_color == constants.BACKGROUND_RGB_COLOR:
        return constants.BACKGROUND_CLASS
    elif rgb_color == constants.UNCERTAINTY_RGB_COLOR:
        return constants.UNCERTAINTY_CLASS

    assert False, f"Non c'è una classe per il colore {rgb_color}"


def get_metrics(y_true, y_target):
    y_true = y_true.reshape(-1)
    y_target = y_target.reshape(-1)

    # uso solo le classi arteria e vena
    y_true_arteir_vein_only = []
    y_target_artery_vein_only = []
    for i in range(len(y_true)):
        if y_true[i] == constants.ARTERY_CLASS or y_true[i] == constants.VEIN_CLASS:
            y_true_arteir_vein_only.append(y_true[i])
            y_target_artery_vein_only.append(y_target[i])

    y_true = y_true_arteir_vein_only
    y_target = y_target_artery_vein_only

    accuracy = accuracy_score(y_true, y_target)
    precision = precision_score(y_true, y_target, average="weighted")
    recall = recall_score(y_true, y_target, average="weighted")
    f1 = f1_score(y_true, y_target, average="weighted")

    return accuracy, precision, recall, f1
