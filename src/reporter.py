import pandas as pd
from IPython.display import display
from utils_image import show_multiple_edges, show_edges, show_images_single_row, show_directed_graph, show_edges_good_bad, get_colored_img_from_labels, get_labels_from_colored_img
from constants import TypeClassification
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import torch
import pickle
import constants


class Reporter(object):
    def __init__(self, graph_builder, num_images):
        self.data_edges = None
        self.data_nodes = None
        self.graph_builder = graph_builder

        self.labels = []
        self.colored_imgs_label = []

        self.data_over_images = {}
        for i in range(num_images):
            self.data_over_images[i] = {}

    def get_DataFrame_from_data(self):
        # print(self.data_over_images)
        # data_dict = {}
        # # devo prendere una colonna per ogni attributo
        # for key in self.data_over_images.keys():
        #     data_column = []
        #     for idx in len(self.data_over_images.keys())

        #     data_dict[key] = []

        df = pd.DataFrame(data=self.data_over_images)
        return df

    def show_DataFrame(self):
        df = self.get_DataFrame_from_data()
        # df.loc[:, "Row_Total"] = df.sum(numeric_only=True, axis=1)
        # print(df)
        display(df)
        display(df.agg(["sum", "mean", "std"], axis="columns"))
        # display(df.agg(["min", "max", "median", "mean", "std", "sum"], axis="columns"))

    def get_data_for_img(self, idx_img):
        return self.data_over_images[idx_img]

    def print_data_for_img(self, idx_img):
        data = self.get_data_for_img(idx_img)
        print(f"Dati immagine {idx_img}")
        for key in data.keys():
            print(f"{key}: {data[key]}")
        print()

    def generate_data_nodes_4(self, idx_img):
        nodes_4_degree = self.graph_builder.get_nodes_with_degree(4)
        nodes_4_degree_corrected = self.get_nodes_4_corrected(nodes_4_degree)
        nodes_4_degree_corrected_good, nodes_4_degree_corrected_bad = self.get_nodes_4_corrected_good_and_bad(nodes_4_degree_corrected)

        nodes_4_degree_wrong_classification = self.get_nodes_4_wrong_classification(nodes_4_degree)

        self.data_over_images[idx_img]["#nodes_4_degree"] = len(nodes_4_degree)
        # self.data_over_images[idx_img]["#nodes_4_degree_wrong_classification"] = len(nodes_4_degree_wrong_classification)
        self.data_over_images[idx_img]["#nodes_4_degree_corrected"] = len(nodes_4_degree_corrected)
        self.data_over_images[idx_img]["#nodes_4_degree_corrected_good"] = len(nodes_4_degree_corrected_good)
        self.data_over_images[idx_img]["#nodes_4_degree_corrected_bad"] = len(nodes_4_degree_corrected_bad)

    def get_nodes_4_corrected(self, nodes_4_degree):
        # nodo grado 4 corretto significa che c'è una differenza tra EU e NN in uno qualunque dei suoi archi
        nodes_4_corrected = set()
        # for node in nodes_4_degree:
        #     # ottengo l'ID degli edge di questo nodo
        #     id_edges = self.graph_builder.get_ID_edges_connected_to_node(node)

        #     corrected = False
        #     for idx in id_edges:
        #         if self.data_edges[idx].class_nn != self.data_edges[idx].class_EU:
        #             corrected = True
        #             break

        #     if corrected:
        #         nodes_4_corrected.add(node)

        # # TODO: gestire meglio metadati
        for node in nodes_4_degree:
            if self.data_nodes[node].easy_cross_over:
                nodes_4_corrected.add(node)

        return nodes_4_corrected

    def get_nodes_4_corrected_good_and_bad(self, nodes_4_corrected):
        # un nodo di grado 4 è corretto male se almeno un arco ha la classe EU diversa da GT
        nodes_4_corrected_good = set()
        nodes_4_corrected_bad = set()
        for node in nodes_4_corrected:
            id_edges = self.graph_builder.get_ID_edges_connected_to_node(node)

            bad_corrected = False
            for idx in id_edges:
                if self.data_edges[idx].class_EU != self.data_edges[idx].class_GT:
                    bad_corrected = True
                    break

            if bad_corrected:
                nodes_4_corrected_bad.add(node)
            else:
                nodes_4_corrected_good.add(node)

        return nodes_4_corrected_good, nodes_4_corrected_bad

    def get_nodes_4_wrong_classification(self, nodes_4_degree):
        # un nodo di grado 4 ha una classificazione sbagliata se nn è diverso da GT
        nodes_4_wrong = set()
        for node in nodes_4_degree:
            wrong = False
            for idx in self.graph_builder.get_ID_edges_connected_to_node(node):
                if self.data_edges[idx].class_nn != self.data_edges[idx].class_GT:
                    wrong = True
                    break

            if wrong:
                nodes_4_wrong.add(node)

        return nodes_4_wrong

    def generate_data_good_classification(self, idx_img):
        distances = range(2, 16)
        for dis in distances:
            nodes = self.graph_builder.get_nodes_with_distance_from_any_source_nodes(dis)
            good_classification, total_edges = self.calculate_fraction_of_good_classifications_of_nodes(nodes)
            # self.data_over_images[idx_img][f"#edges with distance {dis}"] = total_edges
            # self.data_over_images[idx_img][f"#edges correct"] = good_classification
            if total_edges != 0:
                self.data_over_images[idx_img][f"fraction correct distance {dis}"] = round(good_classification / total_edges, 2)
            else:
                self.data_over_images[idx_img][f"fraction correct distance {dis}"] = None

    def calculate_fraction_of_good_classifications_of_nodes(self, nodes):
        edges = set()
        # prendo ogni arco di ogni nodo
        for node in nodes:
            edges = edges.union(self.graph_builder.get_ID_edges_connected_to_node(node))

        # TODO eliminare
        d_edges = []
        for edge in edges:
            d_edges.append(self.data_edges[edge])
        # show_multiple_edges(d_edges)

        # calcolo quanti di questi archi sono stati classificati correttamente dalla rete
        good_classification = 0
        for edge in edges:
            if self.data_edges[edge].class_nn == self.data_edges[edge].class_GT:
                good_classification += 1

        return good_classification, len(edges)

    def generate_data_nodes_3(self, idx_img):
        nodes_3_degree = self.graph_builder.get_nodes_with_degree(3)

        nodes_3_degree_artery_vein = []
        # trovami tutti i nodi che nel ground truth hanno due archi uguali e uno diverso
        for node in nodes_3_degree:
            id_edges = list(self.graph_builder.get_ID_edges_connected_to_node(node))

            if self.data_edges[id_edges[0]].class_GT != self.data_edges[id_edges[1]].class_GT or self.data_edges[id_edges[0]].class_GT != self.data_edges[id_edges[2]].class_GT:
                nodes_3_degree_artery_vein.append(node)

        # self.data_over_images[idx_img]["#nodes_artery_vein_bifurcation"] = len(nodes_3_degree_artery_vein)

        # possiamo fidarci della rete per l'identificazione di questo tipo di nodi
        # numero di questi nodi correttamente classificato dalla rete
        num_nodes_correct_classification = 0
        for node in nodes_3_degree_artery_vein:
            wrong = False
            for idx in self.graph_builder.get_ID_edges_connected_to_node(node):
                if self.data_edges[idx].class_nn != self.data_edges[idx].class_GT:
                    wrong = True
                    break

            if not wrong:
                num_nodes_correct_classification += 1

        # self.data_over_images[idx_img][f"#nodes_artery_vein_correct_classification"] = num_nodes_correct_classification

        for node in nodes_3_degree_artery_vein:
            # trovo quali sono i due uguali e quello diverso
            id_edges = list(self.graph_builder.get_ID_edges_connected_to_node(node))
            if self.data_edges[id_edges[0]].class_GT == self.data_edges[id_edges[1]].class_GT:
                edges_equal = [self.data_edges[id_edges[0]], self.data_edges[id_edges[1]]]
                edge_different = self.data_edges[id_edges[2]]
            elif self.data_edges[id_edges[0]].class_GT == self.data_edges[id_edges[2]].class_GT:
                edges_equal = [self.data_edges[id_edges[0]], self.data_edges[id_edges[2]]]
                edge_different = self.data_edges[id_edges[1]]
            else:
                edges_equal = [self.data_edges[id_edges[1]], self.data_edges[id_edges[2]]]
                edge_different = self.data_edges[id_edges[0]]

            # calcolo differenza in valore assoluto tra la media del diametro dei due uguali e quello diverso
            mean_diameter_equal_edges = (edges_equal[0].get_diameter() + edges_equal[1].get_diameter()) / 2
            diff = abs(mean_diameter_equal_edges - edge_different.get_diameter())
            # self.data_over_images[idx_img][f"node{node}_diff_diameter_equal_edges"] = diff

            # calcolo angolo tra i due uguali
            # self.data_over_images[idx_img][f"node{node}_angle_equal_edges"] = edges_equal[0].get_angle_with(edges_equal[1])

            # calcolo angolo tra i due uguali e quello diverso
            # self.data_over_images[idx_img][f"node{node}_angle_equal_vs_different_1"] = edges_equal[0].get_angle_with(edge_different)
            # self.data_over_images[idx_img][f"node{node}_angle_equal_vs_different_2"] = edges_equal[1].get_angle_with(edge_different)

            # show_edges([edges_equal[0], edges_equal[1], edge_different])

        nodes_with_cycle = []
        for node in nodes_3_degree:
            # è presente un ciclo partendo da uno degli archi?
            adjacent_nodes = self.graph_builder.get_nodes_adjacent_to_node(node)
            for adj_node in adjacent_nodes:
                if self.graph_builder.get_min_distance_from_node_tmp(adj_node, node) != -1:
                    nodes_with_cycle.append(node)
                    break
        # self.data_over_images[idx_img][f"#nodes_3_degree_with_cycle"] = len(nodes_with_cycle)

        # for node in nodes_with_cycle:
        #     edges = []
        #     for idx in self.graph_builder.get_ID_edges_connected_to_node(node):
        #         edges.append(self.data_edges[idx])

        #     show_edges(edges)

        num_nodes_3_degree_problematic_with_cycle = 0
        num_nodes_3_degree_no_problematic_with_cycle = 0
        for node in nodes_with_cycle:
            if node in nodes_3_degree_artery_vein:
                num_nodes_3_degree_problematic_with_cycle += 1
            else:
                num_nodes_3_degree_no_problematic_with_cycle += 1

        num_nodes_3_degree_problematic_no_cycle = 0
        for node in nodes_3_degree_artery_vein:
            if node not in nodes_with_cycle:
                num_nodes_3_degree_problematic_no_cycle += 1

        self.data_over_images[idx_img][f"#nodes_3_degree_with_cycle_artery_vein"] = num_nodes_3_degree_problematic_with_cycle
        self.data_over_images[idx_img][f"#nodes_3_degree_no_cycle_artery_vein"] = num_nodes_3_degree_problematic_no_cycle
        self.data_over_images[idx_img][f"#nodes_3_degree_no_artery_vein_with_cycle"] = num_nodes_3_degree_no_problematic_with_cycle

    def generate_data_split_nodes_3(self, idx_img):
        # numero di nodi di grado 3 splittati
        # numero nodi di grado 3 splittati correttamente
        # numero nodi di grado 3 che dovevano essere splittati ma non sono stati splittati
        DG = self.graph_builder.get_DG_from_graph(self.data_edges)
        # i nodi generati da quelli splittati hanno degree 1
        num_splits = 0
        num_correct_splits = 0
        num_wrong = 0
        nodes_degree_1 = self.graph_builder.get_nodes_with_degree(1)
        good_edges = []
        bad_edges = []
        for node in nodes_degree_1:
            # skippo quelli che non sono effettivamente generati dallo split
            if self.data_nodes[node].id_duplicate_node_3_degree is None:
                continue

            num_splits += 1

            other_node = self.data_nodes[node].id_duplicate_node_3_degree
            assert self.data_nodes[other_node].id_duplicate_node_3_degree == node, "i due nodi non combaciano"
            assert len(list(DG.out_edges(other_node))) == 1, "altro nodo non ha 1 arco uscita"

            # controllo che lo split è corretto
            # lo split è corretto se effettivamente i due archi hanno classe diversa
            edge_1 = self.data_edges[list(self.graph_builder.get_ID_edges_connected_to_node(node))[0]]
            edge_2 = self.data_edges[self.graph_builder.get_ID_from_edge(list(DG.out_edges(other_node))[0])]

            if edge_1.class_GT != edge_2.class_GT:
                num_correct_splits += 1
                edge_1.highlight = True
                edge_2.highlight = True
                good_edges.append(edge_1)
                good_edges.append(edge_2)
            else:
                edge_1.highlight = True
                edge_2.highlight = True
                num_wrong += 1
                bad_edges.append(edge_1)
                bad_edges.append(edge_2)
                # show_edges([edge_1, edge_2])

        show_images_single_row(
            [
                show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.EU, show=False),
                # show_edges(self.data_edges.values(), type_class=TypeClassification.EU, show=False),
                show_edges_good_bad(self.data_edges.values(), good_edges, bad_edges, type_class=TypeClassification.EU, show=False),
                # show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.GT, show=False),
                self.colored_imgs_label[idx_img],
            ]
        )

        self.data_over_images[idx_img][f"#splits_3"] = num_splits
        self.data_over_images[idx_img][f"#splits_3_good"] = num_correct_splits
        self.data_over_images[idx_img][f"#splits_3_wrong"] = num_wrong

    def generate_data_periferical_triplets(self, idx_img):
        DG = self.graph_builder.get_DG_from_graph(self.data_edges)
        num_edges = 0
        num_correct = 0
        num_wrong = 0

        good_edges = []
        bad_edges = []
        for node in DG.nodes():
            if not self.data_nodes[node].periferical_triplet:
                continue

            in_edges = list(DG.in_edges(node))
            out_edges = list(DG.out_edges(node))

            # il nodo deve avere esattamente 1 entrante e 0 uscenti
            assert len(in_edges) == 1 and len(out_edges) == 0, "non ha 1 entrante e 0 uscenti"

            num_edges += 1
            id_edge = self.graph_builder.get_ID_from_edge(in_edges[0])
            if self.data_edges[id_edge].class_EU == self.data_edges[id_edge].class_GT:
                num_correct += 1
                good_edges.append(self.data_edges[id_edge])
            else:
                num_wrong += 1
                bad_edges.append(self.data_edges[id_edge])

        show_images_single_row(
            [
                show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.EU, show=False),
                # show_edges(self.data_edges.values(), type_class=TypeClassification.EU, show=False),
                show_edges_good_bad(self.data_edges.values(), good_edges, bad_edges, type_class=TypeClassification.EU, show=False),
                # show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.GT, show=False),
                self.colored_imgs_label[idx_img],
            ]
        )

        self.data_over_images[idx_img][f"#splits_3"] = num_edges
        self.data_over_images[idx_img][f"#splits_3_good"] = num_correct
        self.data_over_images[idx_img][f"#splits_3_wrong"] = num_wrong

    def generate_data_corrected_edges(self, idx_img):
        num_modified_edges = 0
        num_correct = 0
        num_wrong = 0
        good_edges = []
        bad_edges = []
        for edge in self.data_edges.values():
            if edge.class_EU != edge.class_nn:
                num_modified_edges += 1
                if edge.class_EU == edge.class_GT:
                    num_correct += 1
                    good_edges.append(edge)
                else:
                    num_wrong += 1
                    bad_edges.append(edge)

        show_images_single_row(
            [
                show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.EU, show=False),
                # show_edges(self.data_edges.values(), type_class=TypeClassification.EU, show=False),
                show_edges_good_bad(self.data_edges.values(), good_edges, bad_edges, type_class=TypeClassification.EU, show=False),
                # show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.GT, show=False),
                self.colored_imgs_label[idx_img],
            ]
        )

        self.data_over_images[idx_img][f"#corrections"] = num_modified_edges
        self.data_over_images[idx_img][f"#good_corrections"] = num_correct
        self.data_over_images[idx_img][f"#wrong_corrections"] = num_wrong

    def show_highlight_errors_EU_GT(self):
        for edge in self.data_edges.values():
            if edge.class_EU != edge.class_nn and edge.class_EU != edge.class_GT:
                edge.highlight = True

        show_images_single_row(
            [
                show_edges(self.data_edges.values(), type_class=TypeClassification.EU, show=False),
                show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.GT, show=False),
            ]
        )

    def show_dataframe_split_nodes_3(self):
        df = self.get_DataFrame_from_data()
        print(df)
        display(df)

        df_agg = df.agg(["sum", "min", "max", "median", "mean", "std"], axis="columns")
        percentage = []
        for index, row in df_agg.iterrows():
            if index == 0:
                percentage.append(1)
            else:
                percentage.append(row["sum"] * 100 / df_agg["sum"][0])

        df_agg["%"] = percentage

        # df_agg = df_agg[["sum", "%", "min", "max", "median", "mean", "std"]].round(2)
        df_agg = df_agg[["mean", "std"]].round(6)
        display(df_agg)

    def show_dataframe_edges_EU(self):
        df = self.get_DataFrame_from_data()
        df = df.loc[["#corrections", "#good_corrections", "#wrong_corrections"]]

        df_agg = df.agg(["sum", "min", "max", "median", "mean", "std"], axis="columns")
        percentage = []
        for index, row in df_agg.iterrows():
            if index == 0:
                percentage.append(1)
            else:
                percentage.append(row["sum"] * 100 / df_agg["sum"][0])

        df_agg["%"] = percentage

        df_agg = df_agg[["sum", "%", "min", "max", "median", "mean", "std"]].round(2)
        df_only_sum = df_agg[["sum", "%"]]
        display(df_agg)
        display(df_only_sum)

    def show_dataframe_fake_4_crossover(self):
        df = self.get_DataFrame_from_data()
        display(df)

        df_agg = df.agg(["sum", "min", "max", "median", "mean", "std"], axis="columns")
        percentage = []
        for index, row in df_agg.iterrows():
            if index == 0:
                percentage.append("-")
            else:
                percentage.append(row["sum"] * 100 / df_agg["sum"][0])

        df_agg["%"] = percentage

        df_agg = df_agg[["sum", "%", "min", "max", "median", "mean", "std"]].round(2)
        df_only_sum = df_agg[["sum", "%"]]
        display(df_agg)
        display(df_only_sum)

    def generate_data_4_cross_over(self, idx_img):
        DG = self.graph_builder.get_DG_from_graph(self.data_edges)
        num_modified = 0
        num_good = 0
        num_wrong = 0
        visited = set()
        for node in DG.nodes():
            if not self.data_nodes[node].easy_cross_over:
                continue

            if node in visited:
                continue

            other_node = self.data_nodes[node].id_duplicate_node_4_degree
            if other_node is None:
                continue

            assert self.data_nodes[other_node].id_duplicate_node_4_degree == node, "nodi non combaciano"

            visited.add(node)
            visited.add(other_node)

            # assert len(DG.out_edges(node)) == 1, "piu di un arco uscente per il cross node"
            # assert len(DG.in_edges(node)) == 1, "piu di un arco entrante per il cross node"

            # # if idx_img == 5:
            # #     print(node)
            # #     print(other_node)
            # #     print(DG.out_edges(other_node))
            # #     print("intorno a node: ", self.graph_builder.get_nodes_adjacent_to_node(node))
            # #     print("intorno a other_node: ", self.graph_builder.get_nodes_adjacent_to_node(other_node))
            # #     show_directed_graph(DG, True)
            # assert len(DG.out_edges(other_node)) == 1, "piu di un arco uscente per il duplicate node"
            # assert len(DG.in_edges(other_node)) == 1, "piu di un arco entrante per il duplicate node"

            # idx_in_edge = self.graph_builder.get_ID_from_edge(list(DG.in_edges(node))[0])
            # idx_out_edge = self.graph_builder.get_ID_from_edge(list(DG.out_edges(node))[0])

            # idx_in_other_edge = self.graph_builder.get_ID_from_edge(list(DG.in_edges(other_node))[0])
            # idx_out_other_edge = self.graph_builder.get_ID_from_edge(list(DG.out_edges(other_node))[0])

            # in_edge = self.data_edges[idx_in_edge]
            # out_edge = self.data_edges[idx_out_edge]
            # in_other_edge = self.data_edges[idx_in_other_edge]
            # out_other_edge = self.data_edges[idx_out_other_edge]

            # if (
            #     in_edge.class_EU == in_edge.class_GT
            #     and out_edge.class_EU == out_edge.class_GT
            #     and in_other_edge.class_EU == in_other_edge.class_GT
            #     and out_other_edge.class_EU == out_other_edge.class_GT
            # ):
            #     num_good += 1
            # else:
            #     num_wrong += 1

            first_couple = list(self.graph_builder.get_ID_edges_connected_to_node(node))
            second_couple = list(self.graph_builder.get_ID_edges_connected_to_node(other_node))

            assert len(first_couple) == 2, "piu di due per il primo nodo"
            assert len(second_couple) == 2, "piu di due per il secondo nodo"

            # se gli archi hanno tutti la stessa classe di nn, vuol dire che è stato modificato solo strutturalmente
            if (
                self.data_edges[first_couple[0]].class_EU == self.data_edges[first_couple[0]].class_nn
                and self.data_edges[first_couple[1]].class_EU == self.data_edges[first_couple[1]].class_nn
                and self.data_edges[second_couple[0]].class_EU == self.data_edges[second_couple[0]].class_nn
                and self.data_edges[second_couple[1]].class_EU == self.data_edges[second_couple[1]].class_nn
            ):
                continue

            num_modified += 1

            if (
                self.data_edges[first_couple[0]].class_EU == self.data_edges[first_couple[0]].class_GT
                and self.data_edges[first_couple[1]].class_EU == self.data_edges[first_couple[1]].class_GT
                and self.data_edges[second_couple[0]].class_EU == self.data_edges[second_couple[0]].class_GT
                and self.data_edges[second_couple[1]].class_EU == self.data_edges[second_couple[1]].class_GT
            ):
                num_good += 1
            else:
                num_wrong += 1

            self.data_edges[first_couple[0]].highlight = True
            self.data_edges[first_couple[1]].highlight = True
            self.data_edges[second_couple[0]].highlight = True
            self.data_edges[second_couple[1]].highlight = True

            # correct = True
            # for e in [in_edge, out_edge, in_other_edge, out_other_edge]:
            #     if e.class_EU != e.class_GT:
            #         correct = False
            #         break

        # show_images_single_row(
        #     [
        #         show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.EU, show=False),
        #         show_edges(self.data_edges.values(), type_class=TypeClassification.EU, show=False),
        #         # show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.GT, show=False),
        #         self.colored_imgs_label[idx_img],
        #     ]
        # )

        self.data_over_images[idx_img][f"#nodes_4_modified"] = num_modified
        self.data_over_images[idx_img][f"#nodes_4_correct"] = num_good
        self.data_over_images[idx_img][f"#nodes_4_wrong"] = num_wrong

        print(f"Nodi 4 modificati {num_modified} bene {num_good} male {num_wrong}")

    def generate_data_fake_4_cross_over(self, idx_img):
        DG = self.graph_builder.get_DG_from_graph(self.data_edges)
        num_identified = 0
        num_good = 0
        num_wrong = 0
        visited = set()
        for node in DG.nodes():
            if not self.data_nodes[node].fake_cross_over:
                continue

            num_identified += 1

            adj_edges = list(self.graph_builder.get_ID_edges_connected_to_node(node))
            assert len(adj_edges) == 4, "fake crossover senza 4 archi!!!"

            correct = True
            for e in adj_edges:
                if self.data_edges[adj_edges[0]].class_GT != self.data_edges[e].class_GT:
                    correct = False
                    break
            if correct:
                num_good += 1
            else:
                num_wrong += 1

            for e in adj_edges:
                self.data_edges[e].highlight = True

        # show_images_single_row(
        #     [
        #         show_edges(self.data_edges.values(), type_class=TypeClassification.GT, show=False),
        #         show_edges(self.data_edges.values(), highlight=False, type_class=TypeClassification.GT, show=False),
        #     ]
        # )

        self.data_over_images[idx_img][f"#identified_fake_crossover"] = num_identified
        self.data_over_images[idx_img][f"#fake_crossover_good"] = num_good
        self.data_over_images[idx_img][f"#fake_crossover_wrong"] = num_wrong

    def generate_metrics_single_img(self, idx_img, nn_probs):
        def get_only_artery_vein(array):
            artery_vein = []
            for i in range(len(array)):
                if array[i] == constants.ARTERY_CLASS or array[i] == constants.VEIN_CLASS:
                    artery_vein.append(array[i])

            return artery_vein

        y_true = self.labels[idx_img].reshape(-1)

        # creo l'immagine fornita dalla rete
        labels_nn = torch.argmax(nn_probs, dim=1).permute(1, 2, 0).cpu().numpy()
        colored_img_nn = get_colored_img_from_labels(labels_nn)

        # ricoloro questa immagine con EU e NN
        img_EU = show_edges(self.data_edges.values(), nodes=False, structure=False, highlight=False, type_class=TypeClassification.EU, show=False, img=colored_img_nn.copy())
        img_NN = show_edges(self.data_edges.values(), nodes=False, structure=False, highlight=False, type_class=TypeClassification.NN, show=False, img=colored_img_nn.copy())

        # show_images_single_row([img_NN, img_EU, self.colored_imgs_label[idx_img]])

        # ricavo le etichette
        labels_EU = get_labels_from_colored_img(img_EU).reshape(-1)
        labels_NN = get_labels_from_colored_img(img_NN).reshape(-1)

        # uso solo le classi arteria e vena
        y_true_arteir_vein_only = []
        labels_EU_artery_vein_only = []
        labels_NN_artery_vein_only = []
        for i in range(len(y_true)):
            if y_true[i] != constants.BACKGROUND_CLASS and y_true[i] != constants.UNCERTAINTY_CLASS:
                y_true_arteir_vein_only.append(y_true[i])
                labels_EU_artery_vein_only.append(labels_EU[i])
                labels_NN_artery_vein_only.append(labels_NN[i])

        y_true = y_true_arteir_vein_only
        labels_EU = labels_EU_artery_vein_only
        labels_NN = labels_NN_artery_vein_only

        # calcolo numero pixel diversi
        # num_different_pixels = 0
        # for i in range(len(labels_EU)):
        #     if labels_EU[i] != labels_NN[i]:
        #         num_different_pixels += 1
        # print(f"numero pixel diversi {num_different_pixels}")
        # print(f"numero pixel sui vasi {len(y_true)}")
        # # self.data_over_images[idx_img]["total pixels on vessel"] = len(y_true)
        # # self.data_over_images[idx_img]["pixels modified"] = num_different_pixels

        # calcolo precisione rispetto alla ground truth per entrambe
        precision_NN = precision_score(y_true, labels_NN, average="weighted")
        precision_EU = precision_score(y_true, labels_EU, average="weighted")
        print(f"Precisione NN {precision_NN} EU {precision_EU}")
        self.data_over_images[idx_img]["P_EU - P_NN"] = precision_EU - precision_NN

        accuracy_NN = accuracy_score(y_true, labels_NN)
        accuracy_EU = accuracy_score(y_true, labels_EU)
        print(f"Accuratezza NN {accuracy_NN} EU {accuracy_EU}")
        self.data_over_images[idx_img]["ACC_EU - ACC_NN"] = accuracy_EU - accuracy_NN

        recall_NN = recall_score(y_true, labels_NN, average="weighted")
        recall_EU = recall_score(y_true, labels_EU, average="weighted")
        print(f"Recall NN {recall_NN} EU {recall_EU}")
        self.data_over_images[idx_img]["REC_EU - REC_NN"] = recall_EU - recall_NN

        f1_NN = f1_score(y_true, labels_NN, average="weighted")
        f1_EU = f1_score(y_true, labels_EU, average="weighted")
        print(f"F1 NN {f1_NN} EU {f1_EU}")
        self.data_over_images[idx_img]["F1_EU - F1_NN"] = f1_EU - f1_NN

        self.data_over_images[idx_img]["F1_EU"] = f1_EU
        self.data_over_images[idx_img]["F1_NN"] = f1_NN

    def generate_metrics_fake_4_cross_over(self, idx_img):
        # per ogni nodo di grado 4 assegno la vera etichetta: fake o incrocio
        # per ogni nodo di grado 4 assegno l'etichetta fornita dal classificatore: fake o incrocio
        # calcolo le metriche che mi interessano

        y_true = list()
        labels = list()
        # prendi tutti i nodi di grado 4 escludendo quelli non toccati dall'analisi di incrocio (es: nodi source)
        for node in self.graph_builder.graph.nodes:
            if self.data_nodes[node].fake_cross_over is None:
                continue

            # ottengo la vera etichetta
            if self.data_nodes[node].fake_cross_over:
                adj_edges = list(self.graph_builder.get_ID_edges_connected_to_node(node))
            else:
                adj_edges = list(self.graph_builder.get_ID_edges_connected_to_node(node))

                id_duplicate_node = self.data_nodes[node].id_duplicate_node_4_degree
                adj_edges += list(self.graph_builder.get_ID_edges_connected_to_node(id_duplicate_node))

            assert len(adj_edges) == 4, "Non sono 4 archi!!!!"

            fake = 1
            for e in adj_edges:
                if self.data_edges[adj_edges[0]].class_GT != self.data_edges[e].class_GT:
                    fake = 0
                    break
            y_true.append(fake)

            # ottengo l'etichetta fornita dal classificatore
            labels.append(1 if self.data_nodes[node].fake_cross_over else 0)

        # salvo le etichette per questa immagine, in modo da calcolare le metriche alla fine di tutte le immagini
        self.data_over_images[idx_img]["y_true_fake"] = y_true
        self.data_over_images[idx_img]["labels_fake"] = labels

    def report_all_images_metrics_fake_4_cross_over(self):
        y_true = list()
        labels = list()

        for key in self.data_over_images.keys():
            y_true += self.data_over_images[key]["y_true_fake"]
            labels += self.data_over_images[key]["labels_fake"]

        # calcolo le metriche
        precision = precision_score(y_true, labels, average="weighted")
        accuracy = accuracy_score(y_true, labels)
        recall = recall_score(y_true, labels, average="weighted")
        f1 = f1_score(y_true, labels, average="weighted")

        tn, fp, fn, tp = confusion_matrix(y_true, labels).ravel()
        print((tn, fp, fn, tp))
        print(confusion_matrix(y_true, labels))

        print(classification_report(y_true, labels, target_names=["Incrocio", "Fake Incrocio"]))

    def save_run_on_file(self):
        # calcolo media e deviazione standard di questa run
        df = self.get_DataFrame_from_data()
        display(df)

        df_agg = df.agg(["mean", "std"], axis="columns")
        display(df_agg)

        # carica la matrice
        try:
            runs = pickle.load(open("metrics.pickle", "rb"))
        except:
            runs = []

        # inserisco media e deviazione di questa run
        p_mean = df_agg["mean"][0]
        p_std = df_agg["std"][0]

        a_mean = df_agg["mean"][1]
        a_std = df_agg["std"][1]

        r_mean = df_agg["mean"][2]
        r_std = df_agg["std"][2]

        f1_mean = df_agg["mean"][3]
        f1_std = df_agg["std"][3]

        runs.append([p_mean, p_std, a_mean, a_std, r_mean, r_std, f1_mean, f1_std])
        print("\n")
        for i in range(len(runs)):
            print(f"Run {i} ", end="")
            for j in range(len(runs[i])):
                print(round(runs[i][j], 5), end=" ")
            print()

        # salvo su file il totale delle runs
        pickle.dump(runs, open("metrics.pickle", "wb"))

    def show_all_runs(self):
        # carico tutte le run
        runs = pickle.load(open("metrics.pickle", "rb"))

        # runs_good = []
        # for i in range(len(runs)):
        #     if i == 4 or i == 11:
        #         continue

        #     runs_good.append(runs[i])

        # order = [0, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 11]
        # runs_ordered = [runs_good[i] for i in order]

        # le metto in un data frame
        name_rows = [
            "C",
            "C+FT",
            "C+FA",
            "C+FO",
            "C+FS",
            "S",
            "T",
            "C+S+T",
            "C+FO+S+T",
            "C+FA+S+T",
            "C+FT+S+T",
            "C+FS+S+T",
        ]
        df = pd.DataFrame(
            data=runs,
            columns=["Diff Precision mean", "Diff Precision std", "Diff Accuracy mean", "Diff Accuracy std", "Diff Recall mean", "Diff Recall std", "Diff F1 mean", "Diff F1 std",],
            index=name_rows,
        )
        display(df)

        legends = []
        name_legends = ["C", "S", "T", "FT", "FA", "FO", "FS"]
        legends.append("Crossover 4")
        legends.append("Split nodes 3")
        legends.append("Periferical Triplets")
        legends.append("Fake Crossover (FC) 2 terminals")
        legends.append("FC angle <= 90°")
        legends.append("FC 1 in edge and 3 out edges")
        legends.append("FC 1 couple can reach source 1 couple don't")

        legend = pd.DataFrame(data=legends, index=name_legends, columns=["Meaning"])
        display(legend)

