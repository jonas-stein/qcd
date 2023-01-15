import os
import networkx as nx
import numpy as np
from networkx.generators.community import LFR_benchmark_graph
import random
import datetime
from collections import Counter
import ast
import copy
import neighborhoodConnectivity

path = os.path.abspath(os.path.dirname(__file__))


def create_lfr_graph(diff="default", filename=None):
    """
    Creates an lfr graph with the desired complexity. 
    Depending on the parameters this might take a while.
    The graph will always be connected.

    Parameters
    ----------
    diff : String
        Determines the complexity of the created graph. "easy", "medium" or "hard". 
        Otherwise a default graph will be created.
    filename : String
        If a filename is given the seed will be saved in the file.

    Returns
    -------
    G : networkx.classes.graph.Graph
        The created lfr graph.
    """
    G = None
    seed = None
    connected = False
    i = 0
    print("Trying to create a graph of the complexity " + diff + ".\nTry number:")
    while G is None or not connected:
        seed = create_random_seed()
        i += 1
        print(f"{str(i):10}", end="\r")  # + "Seed: " + str(seed))
        try:
            if diff == "easy" or diff == "lfr_easy":
                G = create_lfr_benchmark_graph(n=50, tau1=4, tau2=3, mu=0.05, max_degree=20, min_community=10,
                                               max_community=15, seed=seed, file_name=path + '/graphs/lfr_easy.gml')
            elif diff == "medium" or diff == "lfr_medium":
                G = create_lfr_benchmark_graph(n=65, tau1=5, tau2=3, mu=0.065, max_degree=20, min_community=8,
                                               max_community=13, seed=seed, file_name=path + '/graphs/lfr_medium.gml')
            elif diff == "hard" or diff == "lfr_hard":
                G = create_lfr_benchmark_graph(n=80, tau1=6.5, tau2=1.5, mu=0.08, max_degree=20, min_community=7,
                                               max_community=11, seed=seed, file_name=path + '/graphs/lfr_hard".gml')
            elif diff == "test":
                G = create_lfr_benchmark_graph(n=80, tau1=6.5, tau2=1.5, mu=0.08, max_degree=20, min_community=7,
                                               max_community=11, seed=seed, file_name=path + '/graphs/lfr_test.gml')
            else:
                print("No correct benchmark given!")

        except nx.exception.ExceededMaxIterations:
            pass

        if not (G is None):
            connected = nx.is_connected(G)
            if not connected:
                print("Graph is not connected. Creating another graph.") 
                G = None

    if not (filename is None):
        with open(filename, "a") as f:
            f.write(f"{str(seed):15}")

    return G


def create_sbm_graph(n=400, c=10, intra_prob=0.375, seed=None):
    inter_prob = 1 - intra_prob
    sizes = [int(n/c)] * c  # equally sized communities
    # edge connection probabilities for sbm, 1/2 is for nullifying double counting from symmetry of the matrix:
    probs = [[1/2 * inter_prob/(c-1) if i != j else intra_prob for j in range(c)] for i in range(c)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return G


def create_lfr_benchmark_graph(n=100, tau1=3, tau2=2, mu=0.05, average_degree=10, max_degree=30, min_community=20,
                               max_community=100, seed=None, file_name=None):
    """Creates a sample LFR benchmark graph. Saves it as a .gml file if a file_name is specified.

    Parameters
    ----------
    n : int, optional
        Number of nodes in the graph.
    tau1 : float, optional
        Power law exponent for the degree distribution of the created
        graph. This value must be strictly greater than one.
    tau2 : float, optional
        Power law exponent for the community size distribution in the
        created graph. This value must be strictly greater than one.
    mu : float, optional
        Fraction of inter-community edges incident to each node. This
        value must be in the interval [0, 1].
    average_degree : float, optional
        Desired average degree of nodes in the created graph. This value
        must be in the interval [0, *n*]. Exactly one of this and
        ``min_degree`` must be specified, otherwise a
        :exc:`NetworkXError` is raised.
    max_degree : int, optional
        Maximum degree of nodes in the created graph. If not specified,
        this is set to ``n``, the total number of nodes in the graph.
    min_community : int, optional
        Minimum size of communities in the graph. If not specified, this
        is set to ``min_degree``.
    max_community : int, optional
        Maximum size of communities in the graph. If not specified, this
        is set to ``n``, the total number of nodes in the graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
    file_name : str, optional
        If specified, a .gml of the generated LFR benchmark graph will be saved under this file name.

    Returns
    -------
    networkx.classes.graph.Graph
        The LFR benchmark graph as a networkx graph object. The community structure gets encoded as a node attribute
        under the name of "ground_truth_community".
    """
    G = LFR_benchmark_graph(n, tau1, tau2, mu, max_degree=max_degree, average_degree=average_degree,
                            min_community=min_community, max_community=max_community, seed=seed)
    G.remove_edges_from(nx.selfloop_edges(G))
    nx.set_node_attributes(G, {v: str(G.nodes[v]["community"]) for v in G.nodes()}, "ground_truth_community")
    for v in G.nodes():
        del G.nodes[v]["community"]
    if file_name is not None:
        nx.write_gml(G, file_name)
    print('G.nodes: ' + str(G.nodes))
    return G


def create_random_seed(low_end=100000, high_end=999999):
    random.seed(datetime.datetime.now())
    rand_seed = random.randint(low_end, high_end)
    return rand_seed


def create_seed_list(seed=694201337, amount_of_seeds=50, low_end=100000, high_end=999999):
    random.seed(seed)
    seeds = []
    for i in range(amount_of_seeds):
        seeds.append(random.randrange(low_end, high_end))
    return seeds


def assign_com_index_ground_truth(G):
    """ 
    Analyzes the ground truth of an lfr graph and assigns each node a community index.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to be analysed. Has to be an lfr graph with ground truth.
    
    Returns
    -------
    dict
        The assignments of nodes to community indices.
    """
    ground_truth = nx.get_node_attributes(G, "ground_truth_community")
    communities = []
    assignment_index = {}

    for key in ground_truth:

        if ground_truth[key] in communities:
            index = communities.index(ground_truth[key])
            assignment_index.update({key: index})

        else:
            communities.append(ground_truth[key])
            index = communities.index(ground_truth[key])
            assignment_index.update({key: index})
        
    return assignment_index


def count_lfr_coms(G):
    """
    Counts the number of communities for an lfr graph with ground truth.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to be analysed. Has to be an lfr graph with ground truth.
    
    Returns
    -------
    int
        The number of communities in the graph.
    """
    ground_truth = nx.get_node_attributes(G, "ground_truth_community")
    coms = Counter(ground_truth.values())
    com_amounts = len(coms)
    return com_amounts


def get_com_list_from_lfr(G):
    """ 
    Analyzes the ground truth of an lfr graph and returns a list of all communities as lists.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to be analysed. Has to be an lfr graph with ground truth.
    
    Returns
    -------
    list
        A list of all communities.
    """

    ground_truth = nx.get_node_attributes(G, "ground_truth_community")
    communities = []
    for key in ground_truth:
        if list(eval(ground_truth[key])) not in communities:
            communities.append(list(eval(ground_truth[key])))

    return communities


def assign_nodes_to_coms(G, communities):
    """
    Calculates the modularity of the graph g using the assignments of all
    nodes to their community.
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to be analysed.
    communities : dict
        All communities in the graph keyed by their index.
    
    Returns
    -------
    node_assignments : dict
        A dictionary keyed by every node with their community index as their value
    """
    node_assignments = {i: None for i in G.nodes()}
    for i in G.nodes():
        for j, c in communities.items():
            if i in c:
                node_assignments[i] = j
    return node_assignments


def calculate_modularity(G, node_assignments):
    """
    Calculates the modularity of the graph g using the assignments of all
    nodes to their community.
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to be analysed.
    node_assignments : dict
        The community assignments of the nodes keyed by the node index.
        Form: {node: community_assignment}
    
    Returns
    -------
    float
        The modularity of graph g with its nodes assigned to communities as
        described in node_assignments.
    """
    community_buckets = {community_index: set() for community_index in set(node_assignments.values())}
    for node, community_index in node_assignments.items():
        community_buckets[community_index].add(node)
    communities = list(community_buckets.values())
    return nx.algorithms.community.modularity(G, communities)


def find_separation_node_set(G, iterations=1, file_name=None):
    """
    Searches for a set of separator-nodes for the given input graph.
    Returns the empty set if no such set was found.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to analyse. Has to include node attributes named "ground_truth_community" specifying the community of
        each node via the set all nodes in that nodes community.
    iterations : int, optional
        Number of solutions to be fetched from the QUBO solver.
    file_name : str, optional
        If a suitable set of nodes was found, and a file name was specified, the resulting graph (with node attributes
        flagging the computed separation-node set) gets stored.

    Returns
    -------
    set
        A set of separator-nodes. Set is empty if no adequate set was found. An adequate set is defined as a bijective
        separator-node set.
    """
    n = G.number_of_nodes()
    q = np.zeros((n, n))
    communities = {v: G.nodes[v]["ground_truth_community"] for v in G.nodes()}
    node_list = list(G.nodes)
    numbering = {node_list[i]: i for i in range(len(node_list))}
    for i in G.nodes:
        q[numbering[i]][numbering[i]] = -1
    for (i, j) in G.edges:
        q[numbering[i]][numbering[j]] = 0 if communities[i] == communities[j] else 20
    solution = neighborhoodConnectivity.solve(q, iterations)
    node_list = list(G.nodes)
    reverse_numbering = {i: node_list[i] for i in range(n)}  # was range(len(node_list)) instead of n
    node_assignments = {reverse_numbering[index]: classification for index, classification in solution.items()}
    identified_nodes = [v for v in node_assignments if node_assignments[v] == 0]

    # check if the identified nodes actually form a set of separator-nodes.
    H = copy.deepcopy(G)
    H.remove_nodes_from(identified_nodes)
    connected_components = [list(cc) for cc in list(nx.algorithms.components.connected.connected_components(H))]

    gt_communities = []
    for com in set(communities.values()):
        gt_communities.append({node for node in ast.literal_eval(com)})
    connected_components = [set(cc) for cc in connected_components]
    # check separation-node set property of the connected components
    # in other words: check if each connected component is subset of exactly one ground truth community

    for cc in connected_components:
        cc = {int(x) for x in cc}
        alright = False
        for com in gt_communities:
            if cc.issubset(com):
                if not alright:
                    alright = True
                    break
                else:
                    print('The calculated set is not a set of separation-nodes!'
                          '\nA node was part of multiple communities.')
                    print("Calculated set:\n" + str(identified_nodes))
                    return set()
        if not alright:
            print('The calculated set is not a set of separation-nodes!')
            print("Calculated set: " + str(identified_nodes))
            return set()
    # check bijectivity of the corresponding refinement map
    # in other words: check if each ground truth community is superset of exactly one connected component
    for com in gt_communities:
        alright = False
        for cc in connected_components:
            cc = {int(x) for x in cc}
            if com.issuperset(cc):
                if not alright:
                    alright = True
                else:
                    print('The calculated set is not a bijective set of separation-nodes!2')
                    print("Calculated set: " + str(identified_nodes))
                    return set()
        if not alright:
            print('The calculated set is not a bijective set of separation-nodes!1')
            print("Calculated set: " + str(identified_nodes))
            return set()

    identified_nodes = set(identified_nodes)
    print('\nFollowing set of separator-nodes was found: ' + str(identified_nodes))
    # set node attributes for G
    attrs = {node: {'separation_node': (node in identified_nodes)} for node in G.nodes}
    nx.set_node_attributes(G, attrs)
    if file_name is not None:
        nx.write_gml(G, file_name)
    calculate_core_nodes(H)
    return identified_nodes


def calculate_core_nodes(G):
    """
    Calculates the set of core-nodes. (Also prints the number of core-nodes.)

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to analyse. Has to include node attributes named "ground_truth_community" specifying the community of
        each node via the set all nodes in that nodes community.

    Returns
    -------
    set
        The set of core-nodes.
    """
    communities = {v: G.nodes[v]["ground_truth_community"] for v in G.nodes()}
    number_of_core_nodes = 0

    for v in G.nodes():
        is_core_node = True
        for w in G.adj[v]:
            if communities[v] != communities[w]:
                is_core_node = False
                break
        if is_core_node:
            number_of_core_nodes += 1
    print('number_of_core_nodes: ' + str(number_of_core_nodes))
    print('number_of_border_nodes: ' + str(G.number_of_nodes() - number_of_core_nodes))


def find_constraint_violations(G, x: list):
    """
    Finds separation node constraints in a given set of separation nodes.
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to be analysed.
    x : list
        The separation node classification of every node.
        0 corresponds to a separation node, all other nodes are assigned with 1.
    
    Returns
    -------
    separation_set_constraint_violations : int
        The amount of separation nodes set constraint violations.
    injectivity_violations : int
        The amount of injectivity constraint violations.
    surjectivity_violations : int
        The amount of surjectivity constraint violations.
    """
    separation_node_set = [i for i in G.nodes() if x[i] == 0]
    H = copy.deepcopy(G)
    H.remove_nodes_from(separation_node_set)
    connected_components = [set(cc) for cc in nx.connected_components(H)]
        
    gt_communities = []
    for com in set(nx.get_node_attributes(G, "block").values()):
        gt_communities.append({i for i in G.nodes() if G.nodes[i]["block"] == com})
    
    # check separation-node set property of the connected components
    # in other words: check if each connected component is subset of exactly one ground truth community
    # in order to quantify violations, count the number of nodes that are part of a connected component lying in a
    # different than the biggest community intersection (in absolute node count)
    separation_set_constraint_violations = 0    
    for cc in connected_components:
        intersection_sizes = []
        for com in gt_communities:
            number_of_intersecting_nodes = len(cc.intersection(com))
            if number_of_intersecting_nodes > 0:
                intersection_sizes.append(number_of_intersecting_nodes)
        if len(intersection_sizes) > 1:
            intersection_sizes.sort()
            intersection_sizes.pop()
            # the number of all nodes not belonging to the community with the biggest intersection:
            separation_set_constraint_violations += sum(intersection_sizes)
    # check bijectivity of the corresponding refinement map
    # in other words: check if each ground truth community is superset of exactly one connected component
    # to quantify injectivity violations: counts the number of nodes besides the biggest intersection with a community
    # (in absolute node numbers)
    # to quantify surjectivty: simply count up the number of communities not found
    surjectivity_violations = 0
    injectivity_violations = 0
    for com in gt_communities:
        intersection_sizes = []
        for cc in connected_components:
            intersection_size = len(com.intersection(cc))
            if intersection_size > 0:
                intersection_sizes.append(len(cc))
        if len(intersection_sizes) == 0:
            surjectivity_violations += 1
        elif len(intersection_sizes) > 1:
            intersection_sizes.sort()
            intersection_sizes.pop()
            injectivity_violations += sum(intersection_sizes)
    return separation_set_constraint_violations, injectivity_violations, surjectivity_violations
