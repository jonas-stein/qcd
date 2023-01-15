import networkx as nx
import numpy as np
import math
import dimod
import graphs
from random import randrange
from numpy.random import rand


def bfs(G, v, w, d=2):
    """Executes a breadth-first search starting a the nodes v and w and omitting the edge (v, w)
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The networkx graph containing the edge to be analysed.
    v : hashable
        A node in the graph G, must be adjacent to w.
    w : hashable
        A node in the graph G, must be adjacent to v.
    d : int
        The depth of the neighborhood to explore.
    
    Returns
    -------
    node_layer_depth : dict
        Dict mapping each node to its layer depth, -1 if node was not explored.
    nodes_in_layers : dict
        Dict mapping each layer to all nodes contained in that layer, keyed by the layer's depth.
        d(v) = d(w) = 0
    edge_layer_depth : dict
        Dict mapping each edge to its layer depth, -1 if edge was not explored.
        The edge layer depth of an edge d(v, w) is the average of the node layer depths (d(v)+d(w))/2.
    edges_in_layers : dict
        Dict mapping each layer depth to all edges contained in that layer.
    sub_tree_root : dict
        Dict mapping each node to its respective subtree root(s), None if node was not explored.
        The possible subtree roots are v and w and {v,w} if the respective node has parents of different subtree roots.
    """
    # delete and later reintroduce the edge between v & w
    G.remove_edge(v, w)

    edge_layer_depth = {edge: -1 for edge in G.edges()}
    edge_layer_depth[(v, w)] = 0
    edge_layer_depth[(w, v)] = 0

    sub_tree_root = {node: None for node in G.nodes()}
    sub_tree_root[v] = v
    sub_tree_root[w] = w

    node_layer_depth = {node: -1 for node in G.nodes()}
    node_layer_depth[v] = 0
    node_layer_depth[w] = 0

    nodes_in_layers = {i: set() for i in range(d + 1)}
    nodes_in_layers[0] = {v, w}

    edges_in_layers = {i: set() for i in np.arange(0.5, d, 0.5)}

    for i in range(0, d):
        for node in nodes_in_layers[i]:
            for neighbor in G.neighbors(node):  # typical bfs iterations
                if node_layer_depth[neighbor] == -1:  # previously unseen
                    nodes_in_layers[i+1].add(neighbor)
                    node_layer_depth[neighbor] = i + 1
                    edges_in_layers[i+0.5].add((node, neighbor))
                    edge_layer_depth[(node, neighbor)] = i + 0.5
                    edge_layer_depth[(neighbor, node)] = i + 0.5
                    sub_tree_root[neighbor] = sub_tree_root[node]
                # neighbor is in same layer as node, and has already been explored
                elif node_layer_depth[neighbor] == node_layer_depth[node]:
                    # make sure the edge wasn't considered before from the opposite direction
                    if (neighbor, node) not in edges_in_layers[i]:
                        edges_in_layers[i].add((node, neighbor))
                        edge_layer_depth[(node, neighbor)] = i
                        edge_layer_depth[(neighbor, node)] = i
                # neighbor is in layer below node, and has already been explored
                elif node_layer_depth[neighbor] == node_layer_depth[node] + 1:
                    edges_in_layers[i+0.5].add((node, neighbor))
                    edge_layer_depth[(node, neighbor)] = i + 0.5
                    edge_layer_depth[(neighbor, node)] = i + 0.5
                    # if new parental subroot appears, mark as an intersection node
                    if sub_tree_root[neighbor] != sub_tree_root[node]:
                        sub_tree_root[neighbor] = {v, w}

    G.add_edge(v, w)

    return node_layer_depth, nodes_in_layers, edges_in_layers, edge_layer_depth, sub_tree_root


def neighborhood_connectivity(G, v, w, edges_in_layers, sub_tree_root, nodes_in_layers, a=0.5):
    """Executes a breadth-first search starting a the nodes v and w and omitting the edge (v, w)
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The networkx graph containing the edge to be analysed.
    v : hashable
        A node in the graph G, must be adjacent to w.
    w : hashable
        A node in the graph G, must be adjacent to v.
    edges_in_layers : dict
        Dict mapping each layer depth to all edges contained in that layer.
    sub_tree_root : dict
        Dict mapping each node to its respective subtree root(s), None if node was not explored.
        The possible subtree roots are v and w and {v,w} if the respective node has parents of different subtree roots.
    a : number
        Determines the influence of direct 2-path connections between v and w vs the influence of 3-path connections
        between v and w. a = 1 only values 2-path connections and a = 0 only values 3-path connections.
    nodes_in_layers : dict
    
    Returns
    -------
    edge_nc : dict
        Dict mapping each edge to its layer depth, -1 if edge is not relevant for the nc computation.
        The edge layer depth of an edge d(v, w) is the average of the node layer depths (d(v)+d(w))/2.
        This variable is only relevant for plotting the respectively used edges for nc computation.
    nc : number
        The neighborhood connectivity between v and w approximating \delta_{c(v)c(w)}.
    """
    d = len(edges_in_layers)

    nc_edges = {k: set() for k in np.arange(0.5, math.ceil(d/2), 0.5)}
    edge_nc = {(i, j): -1 for (i, j) in G.edges()}
    edge_nc[(v, w)] = 0
    edge_nc[(w, v)] = 0

    for k in np.arange(0.5, math.ceil(d/2), 0.5):  # iterate over all previously explored edges
        for (i, j) in edges_in_layers[k]:
            if k == int(k):
                if sub_tree_root[i] != sub_tree_root[j] or sub_tree_root[i] == {v, w}:
                    nc_edges[k].add((i, j))
                    edge_nc[(i, j)] = k
                    edge_nc[(j, i)] = k
            else:
                if sub_tree_root[i] == {v, w} or sub_tree_root[j] == {v, w}:
                    nc_edges[k].add((i, j))
                    edge_nc[(i, j)] = k
                    edge_nc[(j, i)] = k

    # calculate neighborhood connectivity while circumventing possible divisions by zero
    def fun(x, y): return x * y / (2*G.number_of_edges())

    free_stubs_v = {i: sum([getattr(G, 'degree')[node] -
                            sum([1 for (a, b) in edges_in_layers[i-0.5] if a == node or b == node])
                            for node in nodes_in_layers[i] if sub_tree_root[node] == v]) for i in range(1, d)}
    free_stubs_v[0] = getattr(G, 'degree')[v] - 1
    free_stubs_w = {i: sum([getattr(G, 'degree')[node] -
                            sum([1 for (a, b) in edges_in_layers[i-0.5] if a == node or b == node])
                            for node in nodes_in_layers[i] if sub_tree_root[node] == w]) for i in range(1, d)}
    free_stubs_w[0] = getattr(G, 'degree')[w] - 1
    free_stubs_intersection = {i: sum([getattr(G, 'degree')[node] -
                                       sum([1 for (a, b) in edges_in_layers[i-0.5] if a == node or b == node])
                                       for node in nodes_in_layers[i] if sub_tree_root[node] == {v, w}])
                               for i in range(1, d)}
    free_stubs_intersection[0] = 0
    estimated_connections = {k: fun(free_stubs_v[k], free_stubs_w[k]) + fun(free_stubs_intersection[k], free_stubs_v[k]
                                                                            + free_stubs_w[k] + free_stubs_intersection[k]) if k == int(k) else fun(free_stubs_v[k-0.5] + free_stubs_w[k-0.5] + free_stubs_intersection[k-0.5], sum([getattr(G, 'degree')[node] for node in nodes_in_layers[k+0.5]])) for k in np.arange(0.5, math.ceil(d/2), 0.5)}

    nc = 0
    if d > 1 and len(edges_in_layers[1]) != 0:
        nc = a * (len(nc_edges[0.5]) - estimated_connections[0.5]) / len(edges_in_layers[0.5]) + \
             (1 - a) * (len(nc_edges[1]) - estimated_connections[1]) / len(edges_in_layers[1])
    elif len(edges_in_layers[0.5]) != 0:
        nc = (len(nc_edges[0.5]) - estimated_connections[0.5]) / len(edges_in_layers[0.5])
    nc = (nc+1)/2
    return edge_nc, nc


def solve(q, iterations=1):
    """
    Uses Simulated Annealing to find a best possible solution to the passed qubo q.
    
    Parameters
    ----------
    q : np.array
        The qubo matrix, a 2d-np.array.
    iterations : int
        The number of times the QUBO should get solved using SA.
    
    Returns
    -------
    dict
        The qubit indices and their assigned values of the best solution found for all iterations.
    """
    sampleset = dimod.SimulatedAnnealingSampler().sample_qubo(q, num_reads=iterations)
    solution = (sampleset.samples())[0]

    return {key: value for (key, value) in solution.items()}


def calculate_qubo_nc(G, estimations):
    """Calculates the QUBO matrix for finding an optimal set of separation nodes based on a given separation
    edge estimation.
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to be analysed.
    estimations : dict
        A dict with the separation edge estimations, form: {(v, w): nc_estimation}.
    Returns
    -------
    np.array
        The corresponding QUBO matrix Q.
    """
    n = G.number_of_nodes()
    Q = np.zeros((n, n))

    for i in G.nodes():
        Q[i][i] = -1

    for (i, j) in estimations:
        Q[i][j] = 2 * (1 - estimations[(i, j)])

    return Q


def run_separation_node_classification(G, d=2, a=0.5, iterations=20, draw=True):
    """Calculates the QUBO matrix for finding an optimal set of separation nodes based on a given separation edge
    estimation. Uses neighborhood centrality as the hard coded method of separation edge estimation.
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to be analysed.
    d : int
        The depth until which the breadth-first search of the neighborhood connectivity should be executed.
        Currently does not change anything about the result for d > 2 (and does not work for d > 2).
    a :
        Determines the influence of direct 2-path connections between v and w vs the influence of 3-path connections
        between v and w. a = 1 only values 2-path connections and a = 0 only values 3-path connections.
    iterations : int
        The number of times the QUBO should get solved using the solver specified below (currently Simulated Annealing).
    draw : bool
        If set to True, the result is drawn.
    
    Returns
    -------
    dict
        The separation node classification of every node.
        0 corresponds to a separation node, all other nodes are assigned with 1.
    """
    estimations = dict()

    for (i, j) in G.edges():
        node_layer_depth, nodes_in_layers, edges_in_layers, edge_layer_depth, sub_tree_root = bfs(G, i, j, d)
        edge_nc, nc = neighborhood_connectivity(G, i, j, edges_in_layers, sub_tree_root, nodes_in_layers, a=a)
        estimations[(i, j)] = nc

    Q = calculate_qubo_nc(G, estimations)
    solution = solve(Q, iterations=iterations)

    if draw:
        node_color = []
        def map_to_colors(x): return 'r' if x == 0 else 'b'

        for i in G.nodes():
            node_color.append(map_to_colors(solution[i]))

        nx.draw(G, node_color=node_color, with_labels=True)

    return solution


def greedy_separation_node_classification(H, classification):
    """
    Greedy approach to determine communities through separation nodes.
    
    Parameters
    ----------
    H : networkx.classes.graph.Graph
        The graph to be analysed.
    classification : dict 
        The separation node classification of every node.
        0 corresponds to a separation node, all other nodes are assigned with 1.    
    
    Returns
    -------
    communities : dict
        All found communities keyed by their index.
    """
    # H is the original graph
    # G is the subgraph formed by all non-separation nodes (later excluding isolated nodes)
    G = nx.induced_subgraph(H, [i for i in H.nodes() if classification[i] == 1]).copy()

    isolated_nodes = [i for i in G.nodes() if G.degree[i] == 0]

    G.remove_nodes_from(isolated_nodes)
    for i in isolated_nodes:
        classification[i] = 0

    separation_node_set = [i for i in H.nodes() if classification[i] == 0]

    initial_separation_node_set = separation_node_set.copy()

    communities = {i: set(c) for i, c in enumerate(nx.connected_components(G))}

    most_certain_node_connectedness = -1
    most_certain_node = 0
    most_certain_community = 0

    connection_summary = {i: {j: 0 for j in range(len(communities))} for i in separation_node_set}

    for i in separation_node_set:
        for j in H.neighbors(i):
            for k, c in communities.items():
                if j in c:
                    connection_summary[i][k] += 1
                    if connection_summary[i][k] > most_certain_node_connectedness:
                        most_certain_node_connectedness = connection_summary[i][k]
                        most_certain_node = i
                        most_certain_community = k

    for a in range(len(initial_separation_node_set)):
        communities[most_certain_community].add(most_certain_node)
        separation_node_set.remove(most_certain_node)
        most_certain_node_connectedness = -1

        for i in separation_node_set:
            for j in H.neighbors(i):
                for k, c in communities.items():
                    if j in c:
                        connection_summary[i][k] += 1
                        if connection_summary[i][k] > most_certain_node_connectedness:
                            most_certain_node_connectedness = connection_summary[i][k]
                            most_certain_node = i
                            most_certain_community = k

    return communities


def simulated_annealing(G, warmstart, n_iterations, temp):
    """
    Simulated annealing approach using background knowledge.
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to be analysed.
    warmstart :
        Bitstring representing an initial solution
    n_iterations : int
        The amount of simulated annealing iterations.
    temp : int
        Starting temperature.
    
    Returns
    -------
    best found results throughout all runs.
    """

    n = G.number_of_nodes()

    Q = np.zeros((n, n))
    np.fill_diagonal(Q, -1)
    for (i, j) in G.edges():
        if G.nodes[i]["block"] != G.nodes[j]["block"]:
            Q[i][j] = 2

    def f(solution): return solution @ Q @ solution
    # generate an initial point
    if warmstart is None:
        x = np.zeros(n, dtype=np.bool8)
    else:
        x = warmstart
    # init best solution found so far
    curr, curr_eval = x, f(x)
    curr_sep_set_viols, curr_inj_viols, curr_sur_viols = graphs.find_constraint_violations(G, list(x))

    best, best_eval = curr, curr_eval
    best_sep_set_viols, best_inj_viols, best_sur_viols = curr_sep_set_viols, curr_inj_viols, curr_sur_viols

    for i in range(n_iterations):
        print('curr_eval: ' + str(curr_eval) + '\t curr_sep_set_viols: ' + str(curr_sep_set_viols) +
              '\t curr_sur_viols: ' + str(curr_sur_viols) + '\t curr_inj_viols: ' + str(curr_inj_viols), end='\r')
        candidate = np.copy(curr)
        bit_flip_index = randrange(n)
        candidate[bit_flip_index] = not curr[bit_flip_index]
        # check if constraints are satisfied
        cand_sep_set_viols, cand_inj_viols, cand_sur_viols = graphs.find_constraint_violations(G, list(candidate))
        # evaluate candidate point
        candidate_eval = f(candidate)
        if candidate_eval <= best_eval and cand_sep_set_viols <= best_sep_set_viols and \
           cand_sur_viols <= best_sur_viols and cand_inj_viols <= best_inj_viols:
            best, best_eval, best_sep_set_viols, best_sur_viols, best_inj_viols = candidate, candidate_eval, \
                                                                                  cand_sep_set_viols, cand_sur_viols, \
                                                                                  cand_inj_viols
            # print('>%d f(x) = %.5f' % (i, best_eval))
            print('best_eval: ' + str(best_eval) + '\t best_sep_set_viols: ' + str(best_sep_set_viols) +
                  '\t best_sur_viols: ' + str(best_sur_viols) + '\t best_inj_viols: ' + str(best_inj_viols), end='\r')
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis_eval = math.exp(-diff / t)
        constraints_diff = (cand_sep_set_viols + cand_sur_viols + cand_inj_viols) - (curr_sep_set_viols + curr_inj_viols
                                                                                     + curr_sur_viols)

        # check if we should keep the new point
        if constraints_diff <= 0:
            if diff <= 0 or rand() < metropolis_eval:
                curr, curr_eval, curr_sep_set_viols, curr_inj_viols, curr_sur_viols = candidate, candidate_eval, \
                                                                                      cand_sep_set_viols, \
                                                                                      cand_inj_viols, cand_sur_viols
        """
        # metropolis_constraints = math.exp(-constraints_diff / t)
        else:
            if rand() < metropolis_constraints and cand_sep_set_viols <= curr_sep_set_viols:  # dice roll is required 
            to not care about bijectivity
                if diff <= 0 or rand() < metropolis_eval:
                    curr, curr_eval, curr_sep_set_viols, curr_inj_viols, curr_sur_viols = candidate, candidate_eval, 
                    cand_sep_set_viols, cand_inj_viols, cand_sur_viols"""
    return best, best_eval, best_sep_set_viols, best_sur_viols, best_inj_viols
