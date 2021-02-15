import clingo
import re
from edge import Edge


class LogicProgram(object):
    def __init__(self, graph):
        self.logic_program = None
        self.graph = graph

    def assign_ASP_class_to_edges(self, edges):
        input_facts = self.get_input_facts(edges)

        facts = self.get_facts()
        rules = self.get_rules()

        self.logic_program = input_facts + facts + rules
        # print(self.logic_program)

        answer_sets = self.get_answer_sets()
        assert answer_sets, "Nessun answer set generato!"

        # print(f"Numero AS generati {len(answer_sets)}\n")
        assert len(answer_sets) == 1, "Generati più di 1 answer set!!!!!"

        edges = self.get_classification_from_AS(answer_sets[0], edges)

        return edges

    def get_facts(self):
        facts = ""

        return facts

    def get_rules(self):
        rules = ""
        rules += "arco(ID,Y,X,C) :- arco(ID,X,Y,C).\n"

        # rules += "adiacente(X,Y) :- adiacente(Y,X).\n"

        # rules += "num_adiacenti(ID,N) :- adiacente(ID,_), N = #count{ID2 : adiacente(ID,ID2)}.\n"
        # rules += "correzione(Y, C1) :- m(X,C1), m(Y,C2), m(Z,C3), X!=Y, X!=Z, Y!=Z, adiacente(X,Y), adiacente(Y,Z), C1 = C3, C2 != C1, num_adiacenti(Y,2).\n"

        rules += "num_archi_nodo(N, NUM) :- NUM = #count{ID : arco(ID, N, _, _)}, nodo(N).\n"

        rules += "correzione(ID3, C1) :- arco(ID1,N,N1,C1), arco(ID2,N,N2,C2), arco(ID3,N,N3,C3), C1 = C2, C1 != C3, num_archi_nodo(N,3), num_archi_nodo(N3,1), nodo(N), nodo(N1), nodo(N2), nodo(N3), N!=N1, N!=N2, N!=N3, N1!=N2, N1!=N3, N2!=N3.\n"

        rules += "asp(ID, C) :- correzione(ID,C).\n"
        rules += "asp(ID, C) :- not correzione(ID,_), arco(ID,_,_,C).\n"

        return rules

    def get_classification_from_AS(self, answer_set, edges):
        asp_atoms = []
        for atom in answer_set:
            match = re.search("asp\((\d+),(\d)\)", atom)
            if match is not None:
                asp_atoms.append((int(match.group(1)), int(match.group(2))))

        assert len(asp_atoms) == len(edges), f"Numero diverso di archi e atomi asp. {len(edges)} archi vs {len(asp_atoms)} asps"

        for id_edge, asp_class in sorted(asp_atoms):
            edges[id_edge].class_ASP = asp_class

        return edges

    def get_input_facts(self, edges):
        facts_edges = self.get_input_facts_edges(edges)
        facts_nodes = self.get_input_facts_nodes()
        facts_angle_edges = self.get_input_facts_angle_edges(edges)
        facts_diameter_edges = self.get_input_facts_diameter_edges(edges)

        return facts_edges + facts_nodes + facts_angle_edges + facts_diameter_edges

    def get_input_facts_edges(self, edges):
        facts = ""
        for edge in edges.values():
            facts += f"arco({edge.ID},{edge.n1},{edge.n2},{edge.class_nn}).\n"

        return facts

    def get_input_facts_nodes(self):
        facts = ""
        for node in list(self.graph.nodes):
            facts += f"nodo({node}).\n"

        return facts

    def get_input_facts_angle_edges(self, edges):
        facts = ""
        # per ogni arco, calcola l'angolo con i suoi adiacenti e crea il fatto
        for edge in edges.values():
            for edge2 in edges.values():
                if edge.ID != edge2.ID and edge.is_adjacent_with(edge2):
                    angle = edge.get_angle_with(edge2)
                    angle = int(round(angle))
                    facts += f"angolo({edge.ID},{edge2.ID},{angle}).\n"

        return facts

    def get_input_facts_diameter_edges(self, edges):
        facts = ""
        for edge in edges.values():
            diameter = edge.get_diameter()
            diameter = int(round(diameter * 10))
            facts += f"diametro({edge.ID},{diameter}).\n"

        return facts

    def get_answer_sets(self):
        clingo_control = clingo.Control(["--warn=none", 0])
        try:
            clingo_control.add("base", [], self.logic_program)
        except:
            print("Errore nel programma logico")
            print(self.logic_program)
            exit(0)

        clingo_control.ground([("base", [])])

        answer_sets = []
        clingo_control.solve(None, lambda model: answer_sets.append(model.symbols(atoms=True)))

        models_result = [[str(atom) for atom in model] for model in answer_sets]

        return models_result
