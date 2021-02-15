import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from IPython.display import display
from scipy.stats import wilcoxon, ttest_rel
import os
import random
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import time

sys.path.append("../")
from dataset import DatasetRetinalImagesRITE
from unet import UNet
from training_net import (
    train_net,
    train_net_with_corrections,
    show_metrics_on_TD,
    train_net_with_corrections_plus_LD,
)

from utils_image import (
    get_colored_img_from_labels,
    show_images_single_row,
    generate_black_white_img_from_probs,
    show_big_plots_images,
)
import constants


def show_dataset(dataset):
    # train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for images, labels in train_loader:
    #     print("IMAGES")
    #     print(images)
    #     print(images.shape)
    #     for i in range(images.shape[1]):
    #         for j in range(images.shape[2]):
    #             for k in range(images.shape[3]):
    #                 print(images[0][i][j][k].item())

    #     print("\n\n\nLABELS")
    #     print(labels)
    #     print(labels.shape)

    for input_img, labels in dataset:
        print(input_img.shape)
        print(input_img)
        show_images_single_row([input_img.astype("uint8"), labels])
        # label_img = get_colored_img_from_labels(labels)
        # show_images_single_row([input_img, label_img])


def show_final_table(metrics_before, metrics_after):
    def mean_std_metrics(metrics):
        means = []
        stds = []
        for m in metrics:
            values = np.array(metrics[m])
            mean = np.mean(values)
            std = np.std(values)

            means.append(mean)
            stds.append(std)

        return means, stds

    means_before, stds_before = mean_std_metrics(metrics_before)
    means_after, stds_after = mean_std_metrics(metrics_after)

    before = means_before
    after = means_after
    diff = []
    percentage = []
    for i in range(len(after)):
        diff.append(after[i] - before[i])
        percentage.append(((after[i] / before[i]) - 1) * 100)

    data = list(
        zip(means_before, stds_before, means_after, stds_after, diff, percentage)
    )
    name_columns = [
        "Mean before training CD",
        "std",
        "Mean after training CD",
        "std",
        "Difference",
        "Difference %",
    ]
    df = pd.DataFrame(
        data=data, index=list(metrics_before.keys()), columns=name_columns
    )
    display(df)

    return df


def merge_data_K_fold():
    # caricamento dei data_fold
    dict_metrics_all = []
    k_fold = 3
    for k in range(k_fold):
        with open(f"./results/dict_metrics_K{k+1}.pkl", "rb") as f:
            dict_metrics = pickle.load(f)

        dict_metrics_all.append(dict_metrics)

    dict_metrics_final = {}
    # per ogni test
    for test in dict_metrics_all[0]:
        dict_metrics_final[test] = {}
        # per ogni misura
        for m in dict_metrics_all[0][test]:
            dict_metrics_final[test][m] = []
            # per ogni immagine
            for img in range(len(dict_metrics_all[0][test][m])):
                values = [k[test][m][img] for k in dict_metrics_all]

                dict_metrics_final[test][m].append(np.mean(values))

    f = open("./results/dict_metrics_K_fold_final.pkl", "wb")
    pickle.dump(dict_metrics_final, f)
    f.close()


def test_wilcoxon(metrics_before, metrics_after):
    # eseguo il test solo per la F1
    m = "f1"
    values_before = metrics_before[m]
    values_after = metrics_after[m]
    diff = []
    for i in range(len(values_before)):
        diff.append(values_after[i] - values_before[i])

    data = {
        m + " before": values_before,
        m + " after": values_after,
        "difference": diff,
    }
    df = pd.DataFrame(data=data)
    display(df)

    w, p = wilcoxon(
        x=values_after, y=values_before, alternative="two-sided")
    alpha = 0.05
    result = "significant different" if p < alpha else "not significant different"
    print(f"statistic: {w} p-value: {p} result: {result}")


def multiple_test_wilcoxon(dict_metrics):
    alpha = 0.05

    column_w = []
    column_p = []
    column_result = []

    c_t_w = []
    c_t_p = []
    c_t_r = []

    m = "f1"
    for i, test in enumerate(dict_metrics):
        if i == 0:
            values_before = dict_metrics[test][m]
            base_test = test
            continue

        values_after = dict_metrics[test][m]

        w, p = wilcoxon(
            x=values_after, y=values_before, alternative="two-sided")
        result = "yes" if p < alpha else "no"

        print(f"eseguito wilcoxon con {test} p-value: {p} result {result} z {w}")

        column_w.append(w)
        column_p.append(p)
        column_result.append(result)

        t, pt = ttest_rel(values_after, values_before)
        c_t_w.append(t)
        c_t_p.append(pt)
        c_t_r.append("yes" if pt < alpha else "no")

        # time.sleep(5)

    data = list(zip(column_p, column_result))
    name_columns = ["p-value", f"significant different from {base_test}?"]
    # name_rows = ["CD no correction", "CD correction", "CD no correction + LD", "CD correction + LD", "LD30"]

    name_rows = list(dict_metrics.keys())
    name_rows.pop(0)

    df = pd.DataFrame(data=data, index=name_rows, columns=name_columns)
    display(df)

    print("T-test two-sided")
    data = list(zip(c_t_p, c_t_r))
    df2 = pd.DataFrame(data=data, index=name_rows, columns=name_columns)
    display(df2)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_net(file_net):
    net = UNet(num_classes=4)
    net.load_state_dict(torch.load(file_net))
    net = net.to(device)

    return net


def dojo_pipeline():
    def show_final_table(dict_metrics):
        def get_row(dict_metrics, m):
            row = []
            for c in dict_metrics:
                mean = np.mean(dict_metrics[c][m])
                std = np.std(dict_metrics[c][m])

                row.append(mean)
                row.append(std)

            return row

        accuracy_row = get_row(dict_metrics, "accuracy")
        precision_row = get_row(dict_metrics, "precision")
        recall_row = get_row(dict_metrics, "recall")
        f1_row = get_row(dict_metrics, "f1")

        data = [accuracy_row, precision_row, recall_row, f1_row]

        name_columns = []
        for test in dict_metrics:
            name_columns.append(test)
            name_columns.append("std")

        df1 = pd.DataFrame(
            data=data,
            index=list(dict_metrics[name_columns[0]].keys()),
            columns=name_columns,
        )
        display(df1)

        # diff_accuracy = [accuracy_row[0]]
        # diff_precision = [precision_row[0]]
        # diff_recall = [recall_row[0]]
        # diff_f1 = [f1_row[0]]
        diff_accuracy = []
        diff_precision = []
        diff_recall = []
        diff_f1 = []
        for i in range(2, len(accuracy_row), 2):
            diff_accuracy.append(accuracy_row[i] - accuracy_row[0])
            diff_accuracy.append(((accuracy_row[i] / accuracy_row[0]) - 1) * 100)

            diff_precision.append(precision_row[i] - precision_row[0])
            diff_precision.append(((precision_row[i] / precision_row[0]) - 1) * 100)

            diff_recall.append(recall_row[i] - recall_row[0])
            diff_recall.append(((recall_row[i] / recall_row[0]) - 1) * 100)

            diff_f1.append(f1_row[i] - f1_row[0])
            diff_f1.append(((f1_row[i] / f1_row[0]) - 1) * 100)

        data = [diff_accuracy, diff_precision, diff_recall, diff_f1]

        name_columns = []
        for i, test in enumerate(dict_metrics):
            if i == 0:
                continue

            name_columns.append(test)
            name_columns.append("%")

        df2 = pd.DataFrame(
            data=data,
            index=list(dict_metrics[name_columns[0]].keys()),
            columns=name_columns,
        )
        display(df2)

        return df1, df2

    type_folder = "all"
    all_dataset = DatasetRetinalImagesRITE()
    all_dataset.initialize(
        f"../../data/{type_folder}/images/",
        "png",
        f"../../data/{type_folder}/av/",
        "png",
        f"../../data/{type_folder}/coord_disks.txt",
    )

    start_index_training = 20
    end_index_training = 30

    start_index_validation = 30
    end_index_validation = 40

    start_index_TD = 10
    end_index_TD = 20

    start_index_CD = 0
    end_index_CD = 10

    training_dataset = all_dataset.get_subset(
        start=start_index_training, end=end_index_training
    )
    mean_channels = training_dataset.get_mean_channels()
    std_channels = training_dataset.get_std_channels()
    max_channels = training_dataset.get_max_channels()
    norm_parameters = [mean_channels, std_channels, max_channels]

    validation_set = all_dataset.get_subset(
        start=start_index_validation, end=end_index_validation
    )

    TD = all_dataset.get_subset(start=start_index_TD, end=end_index_TD)

    net_ld20 = train_net(
        training_dataset, validation_set, norm_parameters, epochs=200, shuffle=True
    )

    ld20_file = "./models/ld20_1.pt"
    torch.save(net_ld20.state_dict(), ld20_file)

    dict_metrics = {}

    print("Test di base senza niente")
    imgs_LD, metrics_LD = show_metrics_on_TD(net_ld20, TD, norm_parameters)
    dict_metrics["LD20"] = metrics_LD
    print()

    print("Training su CD senza correzioni")
    CD = all_dataset.get_subset(start=start_index_CD, end=end_index_CD)
    net_ld20 = train_net_with_corrections(
        CD, net_ld20, norm_parameters, corrections=False
    )
    print("Test su TD")
    imgs_CD_no_corrections, metrics_CD_no_corrections = show_metrics_on_TD(
        net_ld20, TD, norm_parameters
    )
    dict_metrics["CD no correction"] = metrics_CD_no_corrections
    print()

    print("Training su CD con correzioni")
    net_ld20 = load_net(ld20_file)
    CD = all_dataset.get_subset(start=start_index_CD, end=end_index_CD)
    net_ld20 = train_net_with_corrections(
        CD, net_ld20, norm_parameters, corrections=True
    )
    print("Test su TD")
    imgs_CD_corrections, metrics_CD_corrections = show_metrics_on_TD(
        net_ld20, TD, norm_parameters
    )
    dict_metrics["CD correction"] = metrics_CD_corrections
    print()

    print("Training su CD senza correzioni + LD")
    net_ld20 = load_net(ld20_file)
    CD = all_dataset.get_subset(start=start_index_CD, end=end_index_CD)
    net_ld20 = train_net_with_corrections_plus_LD(
        CD,
        net_ld20,
        norm_parameters,
        training_dataset,
        validation_set,
        corrections=False,
    )
    print("Test su TD")
    (
        imgs_CD_plus_LD_no_correction,
        metrics_CD_plus_LD_no_correction,
    ) = show_metrics_on_TD(net_ld20, TD, norm_parameters)
    dict_metrics["CD no correction + LD"] = metrics_CD_plus_LD_no_correction
    print()

    print("Training su CD con correzioni + LD")
    net_ld20 = load_net(ld20_file)
    CD = all_dataset.get_subset(start=start_index_CD, end=end_index_CD)
    net_ld20 = train_net_with_corrections_plus_LD(
        CD,
        net_ld20,
        norm_parameters,
        training_dataset,
        validation_set,
        corrections=True,
    )
    print("Test su TD")
    imgs_CD_plus_LD, metrics_CD_plus_LD = show_metrics_on_TD(
        net_ld20, TD, norm_parameters
    )
    dict_metrics["CD correction + LD"] = metrics_CD_plus_LD
    print()

    list_of_metrics = [
        metrics_LD,
        metrics_CD_no_corrections,
        metrics_CD_corrections,
        metrics_CD_plus_LD_no_correction,
        metrics_CD_plus_LD,
    ]

    f = open("./results/dict_metrics.pkl", "wb")
    pickle.dump(dict_metrics, f)
    f.close()

    show_final_table(dict_metrics)

    print("Visualizzazione boxplot")
    # per ogni misura, produci l'intero boxplot
    for m in ["accuracy", "precision", "recall", "f1"]:
        # voglio un dataframe in cui su ogni colonna ho il test e sulle righe i valori per le immagini di quella misura
        data = {}
        for test in dict_metrics:
            data[test] = dict_metrics[test][m]

        df = pd.DataFrame(data=data)

        fig, axs = plt.subplots(1, 1, figsize=(20, 15))
        sns.boxplot(data=df, ax=axs)
        axs.set_ylabel(m)
        # axs.set_xlabel("Configurations")

        max_value = df.max().max()
        # print(f"MAX VALUE: {max_value} percentage y {0.005/max_value} percentage h {0.000625/max_value}")
        x1 = 0
        mean_x1 = np.mean(data[list(data.keys())[x1]])
        # notazione statistica per ogni box contro il primo
        for x2 in range(1, len(list(data.keys()))):
            mean_x2 = np.mean(data[list(data.keys())[x2]])

            # y = max_value + 0.005 * x2
            # h = 0.000625
            y = max_value + (max_value * 0.0063084) * x2
            h = max_value * 0.0007885
            col = "k"

            if abs(mean_x2 - mean_x1) < 0.001:
                asterisks = "***"
            elif abs(mean_x2 - mean_x1) < 0.01:
                asterisks = "**"
            elif abs(mean_x2 - mean_x1) < 0.05:
                asterisks = "*"

            plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            plt.text(
                (x1 + x2) * 0.5, y + h, asterisks, ha="center", va="bottom", color=col
            )

        fig.tight_layout()
        plt.show()

    print("Eseguo Test di wilcoxon")
    multiple_test_wilcoxon(dict_metrics)

    print("wilcoxon tra CD no correction + LD e CD correction + LD")
    test_wilcoxon(
        metrics_before=dict_metrics["CD no correction + LD"],
        metrics_after=dict_metrics["CD correction + LD"],
    )


string_seed = "TesiBella"
seed = sum([ord(c) for c in list(string_seed)])
print(f"SEED: {seed}")
set_seed(seed)

if "google.colab" in sys.modules:
    print("Running on CoLab")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    print("Not running on CoLab")
    device = "cpu"

dojo_pipeline()

