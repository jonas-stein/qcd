import neighborhoodConnectivity as nC
import networkx as nx
import graphs
from sklearn.metrics.cluster import normalized_mutual_info_score
import copy
import math
import os
import plotting
path = os.path.abspath(os.path.dirname(__file__))


def modularity_deviation(number_of_runs=50, iterations=50, save_plot_to_csv=True):
    """
    Creates a plot that compares the results for real world data sets to their best known modularity values.

    Parameters
    ----------
    number_of_runs : int
        Determines how often each graph will be analysed.
    iterations : int
        The number of simulated annealing iterations.
    save_plot_to_csv : bool
        Determines whether the data used to create the plot should be saved as a csv file to be recreated.

    Returns
    -------
    NOTHING
    """
    save_plot = True
    type = "mod_dev"

    # best known solutions
    karate_max = 0.41979  # number of comm = 4
    dolphins_max = 0.52852  # number of comm = 5
    miserables_max = 0.56001  # number of comm = 8
    protein_max = 0.6491  # number of comm = 7 by mod-max, 10 is ground-truth
    books_max = 0.52724  # number of comm = 4?
    # jazz_max = 0.4452  # number of comm = 4
    # elegans_max = 0.4342  # number of comm = 12

    bestMod = [karate_max, dolphins_max, miserables_max, protein_max, books_max]
    deviation = [[], [], [], [], []]

    labels = ["karate", "dolphins", "miserables", "protein", "books"]

    i = 0
    for label in labels:
        for j in range(number_of_runs):
            print("Analyzing graph " + label + ", run number " + str(j) + ":\n")

            G = nx.read_gml(os.path.join(path, 'graphs', label + '.gml'))
            G = nx.relabel.convert_node_labels_to_integers(G)

            node_classification = nC.run_separation_node_classification(G, d=2, a=0.5, iterations=iterations,
                                                                        draw=False)
            communities = nC.greedy_separation_node_classification(G, node_classification)
            node_assignments = graphs.assign_nodes_to_coms(G, communities)
            modularity = graphs.calculate_modularity(G, node_assignments)

            print("Best known Modularity for " + label + " is: " + str(bestMod[i]))
            print("Calculated Modularity is: " + str(modularity))

            deviation[i].append(modularity / bestMod[i])

            print("Deviation: \n" + str(deviation[i]))

        i += 1

    plotting.create_box_plot(deviation, labels, save_plot, type, y_axis="Deviation in %")

    # save_plot_to_csv check
    if save_plot_to_csv:
        _ = plotting.create_plot_csv_file(type, deviation, labels)


def nmi_deviation(amount_of_graphs=50, iterations=50, save_seeds=True, save_plot_to_csv=True):
    """
    Creates a plot that compares the results for three different complexity of lfr graphs with their ground truth.

    Parameters
    ----------
    amount_of_graphs : int
        Amount of graphs analyzed for each tier.
    iterations : int
        The number of simulated annealing iterations.
    save_seeds : bool
        Determines whether the seeds and the results should get saved into an extra file.
    save_plot_to_csv : bool
        Determines whether the data used to create the plot should be saved as a csv file to be recreated.

    Returns
    -------
    NOTHING
    """
    save_plot = True
    labels = ["easy", "medium", "hard"]
    deviation = [[], [], []]
    type = "nmi_dev"

    # save_seeds check
    seedfile = None
    if save_seeds:
        seedfile = plotting.create_seed_file(type)

    i = 0
    for label in labels:

        if not (seedfile is None):
            plotting.write_seed_file(seedfile, label, ["normalized mutual information score"])

        for j in range(amount_of_graphs):

            print("Analysing " + label + " graph number " + str(j + 1) + ":\n")

            G = graphs.create_lfr_graph(label, seedfile)
            com_index_dict = graphs.assign_com_index_ground_truth(G)
            com_index_list = list(com_index_dict.values())

            node_classification = nC.run_separation_node_classification(G, d=2, a=0.5, iterations=iterations,
                                                                        draw=False)
            communities = nC.greedy_separation_node_classification(G, node_classification)
            node_assignments = graphs.assign_nodes_to_coms(G, communities)
            node_assignments_list = list(node_assignments.values())

            # print("com_index: " + str(com_index_dict))
            print("ass: " + str(node_assignments))

            nmi_score = normalized_mutual_info_score(com_index_list, node_assignments_list)
            print("Normalized mutual info score for " + str(label) + " graph is: " + str(nmi_score) + "\n")

            deviation[i].append(nmi_score)

            print("NMI scores: \n" + str(deviation[i]) + "\n")

            # save results to seeds file
            if not (seedfile is None):
                with open(seedfile, "a") as f:
                    f.write(f"{str(deviation[i][j]):20}\n")

        i += 1

    plotting.create_box_plot(deviation, labels, save_plot, type, y_axis="NMI score")

    # save_plot_to_csv check
    if save_plot_to_csv:
        _ = plotting.create_plot_csv_file(type, deviation, labels)


def com_deviation(amount_of_graphs=50, iterations=50, save_seeds=True, save_plot_to_csv=True):
    """
    Creates a plot that compares the number of communities found by the algorithm to the actual ground truth number of
    communities.

    Parameters
    ----------
    amount_of_graphs : int
        Amount of graphs analyzed for each tier.
    iterations : int
        The number of simulated annealing iterations.
    save_seeds : bool
        Determines whether the seeds and the results should get saved into an extra file.
    save_plot_to_csv : bool
        Determines whether the data used to create the plot should be saved as a csv file to be recreated.

    Returns
    -------
    NOTHING
    """
    save_plot = True
    deviation = [[], [], []]
    labels = ["easy", "medium", "hard"]
    type = "com_dev"

    # save_seeds check
    seedfile = None
    if save_seeds:
        seedfile = plotting.create_seed_file(type)

    i = 0
    for label in labels:

        if not (seedfile is None):
            plotting.write_seed_file(seedfile, label, ["coms found", "coms ground truth", "deviation"])

        for j in range(amount_of_graphs):

            print("Analysing " + label + " graph number " + str(j + 1) + ":\n")

            G = graphs.create_lfr_graph(label, seedfile)

            node_classification = nC.run_separation_node_classification(G, d=2, a=0.5, iterations=iterations,
                                                                        draw=False)
            communities = nC.greedy_separation_node_classification(G, node_classification)
            # node_assignments = graphs.assign_nodes_to_coms(G, communities)

            coms_found = len(communities.keys())
            coms_gt = graphs.count_lfr_coms(G)
            deviation[i].append(coms_found / coms_gt)

            print("Deviation: \n" + str(deviation[i]) + "\n")

            # save results to seeds file
            if not (seedfile is None):
                with open(seedfile, "a") as f:
                    f.write(f"{str(coms_found):20}{str(coms_gt):20}{str(deviation[i][j]):20}\n")

        i += 1

    plotting.create_box_plot(deviation, labels, save_plot, type, y_axis="Deviation")

    # save_plot_to_csv check
    if save_plot_to_csv:
        _ = plotting.create_plot_csv_file(type, deviation, labels)


def separation_node_set_deviation(amount_of_graphs=50, iterations=50, save_seeds=True, save_plot_to_csv=True):
    """
    Creates a plot that compares the number of separation nodes found by the algorithm to the minimum set of optimal
    separation nodes.

    Parameters
    ----------
    amount_of_graphs : int
        Amount of graphs analyzed for each tier.
    iterations : int
        The number of simulated annealing iterations.
    save_seeds : bool
        Determines whether the seeds and the results should get saved into an extra file.
    save_plot_to_csv : bool
        Determines whether the data used to create the plot should be saved as a csv file to be recreated.

    Returns
    -------
    NOTHING
    """
    save_plot = True
    deviation = [[], [], []]
    labels = ["easy", "medium", "hard"]
    type = "sep_nodes_dev"

    # save_seeds check
    seedfile = None
    if save_seeds:
        seedfile = plotting.create_seed_file(type)

    i = 0
    for label in labels:

        if not (seedfile is None):
            plotting.write_seed_file(seedfile, label, ["sep nodes found", "sep nodes gt", "deviation"])

        for j in range(amount_of_graphs):

            print("Analysing " + label + " graph number " + str(j + 1) + ":\n")

            G = graphs.create_lfr_graph(label, seedfile)

            separation_nodes_amount_gt = len(set(graphs.find_separation_node_set(G)))
            # num_sn = len(find_optimal_separation_node_set(G))

            while separation_nodes_amount_gt == 0:
                separation_nodes_amount_gt = len(set(graphs.find_separation_node_set(G)))
                print("Trying to find separation node set.", end="\r")
            else:
                print("\nThe minimum amount of separation nodes for the graph " + label + " is " + str(
                    separation_nodes_amount_gt))

            node_classification = nC.run_separation_node_classification(G, d=2, a=0.5, iterations=iterations,
                                                                        draw=False)
            # communities = nC.greedy_separation_node_classification(G, node_classification)
            # node_assignments = graphs.assign_nodes_to_coms(G, communities)

            separation_nodes = [i for i in node_classification.keys() if node_classification[i] == 0]
            separation_nodes_count = len(separation_nodes)

            print("Amount of separation nodes found: " + str(separation_nodes_count))

            deviation[i].append(separation_nodes_count / separation_nodes_amount_gt)

            print("\nDeviation: \n" + str(deviation[i]) + "\n")

            # save results to seeds file
            if not (seedfile is None):
                with open(seedfile, "a") as f:
                    f.write(
                        f"{str(separation_nodes_count):20}"
                        f"{str(separation_nodes_amount_gt):20}{str(deviation[i][j]):20}\n")

        i += 1

    plotting.create_box_plot(deviation, labels, save_plot, type, y_axis="Deviation")

    # save_plot_to_csv check
    if save_plot_to_csv:
        _ = plotting.create_plot_csv_file(type, deviation, labels)


def prediction_deviation(amount_of_graphs=50, iterations=50, save_seeds=True, save_plot_to_csv=True):
    """
    Creates a plot that compares the prediction for nodes/edges to whether they are actually separating
    (0 is random guessing and 1 would always be correct)

    Parameters
    ----------
    amount_of_graphs : int
        Amount of graphs analyzed for each tier.
    iterations : int
        The number of simulated annealing iterations.
    save_seeds : bool
        Determines whether the seeds and the results should get saved into an extra file.
    save_plot_to_csv : bool
        Determines whether the data used to create the plot should be saved as a csv file to be recreated.

    Returns
    -------
    NOTHING
    """
    save_plot = True
    node_deviation = [[], [], []]
    edge_deviation = [[], [], []]
    deviation = [[], [], []]
    labels = ["easy", "medium", "hard"]
    type = "pred_dev"

    # save_seeds check
    seedfile = None
    if save_seeds:
        seedfile = plotting.create_seed_file(type)

    i = 0
    for label in labels:

        if not (seedfile is None):
            plotting.write_seed_file(seedfile, label, ["node pred dev", "edge pred dev", "# of sep nodes", "deviation"])

        for j in range(amount_of_graphs):

            print("Analysing " + label + " graph number " + str(j + 1) + ":\n")

            G = graphs.create_lfr_graph(label, seedfile)

            separation_nodes_gt, separation_edges_gt = get_all_separation_nodes_edges_lfr(G)

            node_classification = nC.run_separation_node_classification(G, d=2, a=0.5, iterations=iterations,
                                                                        draw=False)
            communities = nC.greedy_separation_node_classification(G, node_classification)
            node_assignments = graphs.assign_nodes_to_coms(G, communities)

            separation_nodes = [i for i in node_classification.keys() if node_classification[i] == 0]

            print("\nSeparation nodes found: " + str(separation_nodes))
            print("Separation nodes Truth: " + str(separation_nodes_gt) + "\n")
            print(node_assignments)

            correct_edge_predictions = 0
            correct_node_predictions = 0
            separation_edges_found = []
            for node in separation_nodes:
                for neighbor in G.neighbors(node):
                    if node_assignments[node] != node_assignments[neighbor] and (
                            ((node, neighbor) in separation_edges_gt) or ((neighbor, node) in separation_edges_gt)):
                        if (node, neighbor) not in separation_edges_found:
                            separation_edges_found.append((node, neighbor))
                            separation_edges_found.append((neighbor, node))
                            # print(str((node,neighbor)) + "is a seperation edge.")
                            correct_edge_predictions += 1

                if node in separation_nodes_gt:
                    correct_node_predictions += 1

            node_deviation[i].append(correct_node_predictions / len(separation_nodes))
            edge_deviation[i].append(correct_edge_predictions / len(separation_nodes))
            deviation[i].append((node_deviation[i][j] + edge_deviation[i][j]) / 2)

            print("Deviation: \n" + str(deviation[i]) + "\n")

            # save results to seeds file
            if not (seedfile is None):
                with open(seedfile, "a") as f:
                    f.write(
                        f"{str(node_deviation[i][j]):20}{str(edge_deviation[i][j]):20}"
                        f"{str(len(separation_nodes)):20}{str(deviation[i][j]):20}\n")

        i += 1

    plotting.create_box_plot(deviation, labels, save_plot, type)

    # save_plot_to_csv check
    if save_plot_to_csv:
        _ = plotting.create_plot_csv_file(type, deviation, labels)


def check_surjective_and_injective_correctness(amount_of_graphs=50, iterations=50, save_seeds=True):
    """
    Checks a number of randomly generated graphs whether their surjective or injective correctness gets violated

    Parameters
    ----------
    amount_of_graphs : int
        Amount of graphs analyzed for each tier.
    iterations : int
        The number of simulated annealing iterations.
    save_seeds : bool
        Determines whether the seeds and the results should get saved into an extra file.

    Returns
    -------
    NOTHING
    """

    labels = ["easy", "medium", "hard"]
    type = "sur_inj_corr"

    # save_seeds check
    seedfile = None
    if save_seeds:
        seedfile = plotting.create_seed_file(type)

    # i = 0
    for label in labels:

        if not (seedfile is None):
            plotting.write_seed_file(seedfile, label, ["inj violations", "sur violations", "sep set violations"])

        for j in range(amount_of_graphs):

            print("Analysing " + label + " graph number " + str(j + 1) + ":\n")

            G = graphs.create_lfr_graph(label, seedfile)

            node_classification = nC.run_separation_node_classification(G, d=2, a=0.5, iterations=iterations,
                                                                        draw=False)
            # communities = nC.greedy_separation_node_classification(G, node_classification)
            # node_assignments = graphs.assign_nodes_to_coms(G, communities)

            separation_nodes = [i for i in node_classification.keys() if node_classification[i] == 0]

            H = copy.deepcopy(G)
            for node in separation_nodes:
                H.remove_node(node)
            connected_components = [set(cc) for cc in nx.connected_components(H)]

            gt_communities = graphs.get_com_list_from_lfr(G)

            separation_set_constraint_violations = 0
            for cc in connected_components:
                intersection_sizes = []
                for com in gt_communities:
                    number_of_intersecting_nodes = len(cc.intersection(set(com)))
                    if number_of_intersecting_nodes > 0:
                        intersection_sizes.append(number_of_intersecting_nodes)
                if len(intersection_sizes) > 1:
                    intersection_sizes.sort()
                    intersection_sizes.pop()
                    separation_set_constraint_violations += sum(intersection_sizes)

            surjectivity_violations = 0
            injectivity_violations = 0
            for com in gt_communities:
                intersection_sizes = []
                for cc in connected_components:
                    intersection_size = len(set(com).intersection(cc))
                    if intersection_size > 0:
                        intersection_sizes.append(len(cc))
                if len(intersection_sizes) == 0:
                    surjectivity_violations += 1
                elif len(intersection_sizes) > 1:
                    intersection_sizes.sort()
                    intersection_sizes.pop()
                    injectivity_violations += sum(intersection_sizes)

            print("\nSeparation node set violations: " + str(separation_set_constraint_violations))
            print("Surjective violations: " + str(surjectivity_violations))
            print("Injective violations: " + str(injectivity_violations) + "\n")

            # save results to seeds file
            if not (seedfile is None):
                with open(seedfile, "a") as f:
                    f.write(
                        f"{str(injectivity_violations):20}{str(surjectivity_violations):20}"
                        f"{str(separation_set_constraint_violations):20}\n")


def intra_probability_SBM_plots(amount_of_graphs=50, simulated_annealing_iterations=1000, save_plot=True,
                                save_plot_to_csv=True):
    """
    Creates a plot that compares the results for three different complexity of SBM graphs with their ground truth.

    Parameters
    ----------
    amount_of_graphs : int
        Amount of graphs analyzed for each tier.
    simulated_annealing_iterations : int
        The number of simulated annealing iterations.
    save_plot : bool
        Determines whether the resulting plot should be saved or not.
    save_plot_to_csv : bool
        Determines whether the data used to create the plot should be saved as a csv file to be recreated.

    Returns
    -------
    NOTHING
    """
    labels = ["NMI Score", "com size", "bijective viols"]

    intra_probs = [0.3, 0.4, 0.5]
    n = 250
    c = 7

    seeds = graphs.create_seed_list(133742072, len(intra_probs) * amount_of_graphs)
    filenames = []

    for i in range(len(intra_probs)):
        nmi_score = []
        nodes_in_cc = []
        bi_sn_set = []

        results = [nmi_score, nodes_in_cc, bi_sn_set]

        for j in range(amount_of_graphs):
            print("Run: " + str(j + 1) + ", intra prob: " + str(intra_probs[i]) + ", seed: " + str(
                seeds[i * amount_of_graphs + j]))
            try:
                G = graphs.create_sbm_graph(n=n, c=c, intra_prob=intra_probs[i], seed=seeds[i * amount_of_graphs + j])
                best, best_eval, best_sep_set_viols, best_sur_viols, best_inj_viols = \
                    nC.simulated_annealing(G, warmstart=None, n_iterations=simulated_annealing_iterations, temp=100)
                communities = nC.greedy_separation_node_classification(G, {k: best[k] for k in range(len(best))})

                # nmi score
                node_assignments = {u: None for u in G.nodes()}
                for u in G.nodes():
                    for j, l in communities.items():  # TODO @Domi check j (already defined in outer for loop)
                        if u in l:
                            node_assignments[u] = j

                nmi_score.append(normalized_mutual_info_score(list(node_assignments.values()),
                                                              list(nx.get_node_attributes(G, "block").values())))

                # nodes in connected components
                cc_nodelist = {cc: set() for cc in communities.keys()}
                for node in G.nodes():
                    if best[node]:
                        cc_nodelist[node_assignments[node]].add(node)

                average_cc_length = 0
                for cc in cc_nodelist:
                    average_cc_length += len(cc_nodelist[cc]) ** 2

                average_cc_length = math.sqrt(average_cc_length / len(cc_nodelist.keys()))
                # print("Average: " + str(average_cc_length))

                variance = 0
                for cc in cc_nodelist:
                    variance += (len(cc_nodelist[cc]) - average_cc_length) ** 2

                variance = variance / len(cc_nodelist.keys())

                standard_deviation = math.sqrt(variance)

                nodes_in_cc.append(
                    1 - standard_deviation / average_cc_length)  # normalized by soll size of the communities int(n/c)

                # separation set violations
                # to quantify injectivity violations: counts the number of nodes besides the biggest intersection with
                # a community (in absolute node numbers)
                # to quantify surjectivty: simply count up the mnumber of communities not found
                bi_sn_set.append(best_inj_viols + best_sur_viols)

            except KeyError:
                print("KeyError occurred")

        plotting.create_box_plot(results, labels, save_plot, "intra_prob: " + str(intra_probs[i]))

        if save_plot_to_csv:
            filename = plotting.create_plot_csv_file("intra_prob_" + str(intra_probs[i]), results, labels)
            filenames.append(filename)

    plotting.combine_box_plots(filenames, save_plot=True)
