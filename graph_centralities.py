from collections import deque
from graph import Graph
import numpy


def adjacency_matrix(graph: Graph) -> tuple:
    """
    Generates an adjacency matrix from a graph representation.

    Args:
        graph (Graph): The graph object representing the graph.

    Returns:
        tuple: A tuple containing the adjacency matrix and a list of node names.
    """
    num_nodes = len(graph.nodes)
    node_names = list(graph.nodes.keys())
    adjacency_matrix = [[0 for _ in range(num_nodes)]
                        for _ in range(num_nodes)]

    for node in graph.nodes.values():
        row_index = node_names.index(node.name)
        for neighbor, weight in node.neighbours:
            try:
                col_index = node_names.index(neighbor)
                adjacency_matrix[row_index][col_index] = 1
            except ValueError:
                # Handle case where neighbor is not found in node_names
                print(f"Error: Neighbor '{neighbor}' not found in node list.")

    return adjacency_matrix, node_names


def degree_centrality(graph: Graph) -> tuple:
    """
    Calculates the degree centrality for each node in the graph.

    Args:
        graph (Graph): The graph object representing the graph.

    Returns:
        tuple: A tuple containing a dictionary of node centrality values and a list of nodes with the highest centrality.
    """
    centrality_scores = {}
    total_nodes = len(graph.nodes)

    # Calculate centrality scores for each node
    for node_name, node in graph.nodes.items():
        centrality_scores[node_name] = len(node.neighbours) / total_nodes

    # Find nodes with the highest centrality
    max_centrality = max(centrality_scores.values())
    top_centrality_nodes = [node for node, centrality in centrality_scores.items(
    ) if centrality == max_centrality]

    return centrality_scores, top_centrality_nodes


def closeness_centrality(graph: Graph) -> tuple:
    """
    Computes the closeness centrality for each node in the graph.

    Args:
        graph (romania.Graph): The graph object representing the graph.

    Returns:
        tuple: A tuple containing a list of closeness scores for all nodes and a list of nodes with the highest closeness centrality scores.
    """
    node_names = [node_name for node_name in graph.nodes.keys()]
    closeness_scores = [0 for _ in range(len(node_names))]

    top_ranked = []
    max_closeness = 0

    for i in range(len(node_names)):
        distances = {node: float('inf') for node in graph.nodes}
        distances[node_names[i]] = 0

        queue = deque([node_names[i]])
        while queue:
            curr_node = queue.popleft()
            for neighbor, cost in graph.nodes[curr_node].neighbours:
                total_dist = distances[curr_node] + cost
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = total_dist
                    queue.append(neighbor)
                if distances[neighbor] > total_dist:
                    distances[neighbor] = total_dist

        sum_distances = sum(distances.values())
        closeness_scores[i] = (len(node_names) - 1) / sum_distances
        if closeness_scores[i] > max_closeness:
            max_closeness = closeness_scores[i]
            top_ranked = [(node_names[i], max_closeness)]
        elif closeness_scores[i] == max_closeness:
            top_ranked.append((node_names[i], max_closeness))

    return dict(zip(node_names, closeness_scores)), top_ranked


def eigenvector_centrality(graph: Graph, max_iterations: int = 100, tolerance: float = 1e-6) -> tuple:
    """
    Computes the eigenvector centrality for each node in the graph using the power iteration method.

    Args:
        graph (romania.Graph): The graph object representing the graph.
        max_iterations (int): Maximum number of iterations for power iteration.
        tolerance (float): Convergence criterion for the difference between successive centrality vectors.

    Returns:
        tuple: A tuple containing a list of centrality values for all nodes and a list of nodes with the highest centrality values.
    """
    adj_matrix, node_names = adjacency_matrix(graph)
    centrality_values = numpy.ones(len(adj_matrix))
    centrality_values /= numpy.linalg.norm(centrality_values)

    for _ in range(max_iterations):
        new_centrality = numpy.dot(adj_matrix, centrality_values)
        new_centrality /= numpy.linalg.norm(new_centrality)

        if numpy.linalg.norm(new_centrality - centrality_values, 2) < tolerance:
            break

        centrality_values = new_centrality

    max_centrality = max(centrality_values)
    top_centrality_nodes = [node_names[i] for i, centrality in enumerate(
        centrality_values) if centrality == max_centrality]

    return dict(zip(node_names, centrality_values)), top_centrality_nodes


def pagerank_centrality(graph: Graph, damping_factor: float = 0.85, max_iterations: int = 100, tolerance: float = 1e-6) -> tuple:
    """
    Computes the PageRank centrality for each node in the graph using the PageRank algorithm.

    Args:
        graph (Graph): The graph object representing the graph.
        damping_factor (float): Damping factor for the PageRank algorithm.
        max_iterations (int): Maximum number of iterations for convergence.
        tolerance (float): Convergence criterion for the difference between successive PageRank vectors.

    Returns:
        tuple: A tuple containing a list of PageRank centrality values for all nodes and a list of nodes with the highest PageRank centrality values.
    """
    adj_matrix, node_names = adjacency_matrix(graph)
    adj_matrix = numpy.array(adj_matrix)
    num_nodes = len(adj_matrix)
    pagerank_scores = numpy.ones(num_nodes) / num_nodes

    for _ in range(max_iterations):
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        transition_matrix = numpy.where(numpy.logical_and(row_sums != 0, ~numpy.isnan(
            row_sums), ~numpy.isnan(adj_matrix)), adj_matrix / row_sums, 1 / num_nodes)
        new_pagerank_scores = (1 - damping_factor) / num_nodes + \
            damping_factor * numpy.dot(transition_matrix.T, pagerank_scores)
        if numpy.linalg.norm(new_pagerank_scores - pagerank_scores, 2) < tolerance:
            break
        pagerank_scores = new_pagerank_scores

    max_centrality = max(pagerank_scores)
    top_centrality_nodes = [node_names[i] for i in range(
        len(pagerank_scores)) if pagerank_scores[i] == max_centrality]

    return dict(zip(node_names, pagerank_scores)), top_centrality_nodes


def katz_centrality(graph: Graph, alpha: float = 0.1, beta: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> tuple:
    """
    Computes the Katz centrality for each node in the graph using the Katz centrality equation.

    Args:
        graph (romania.Graph): The graph object representing the graph.
        alpha (float): Damping parameter for the influence of immediate neighbors.
        beta (float): Scaling factor for the initial centrality values.
        max_iter (int): Maximum number of iterations for convergence.
        tol (float): Convergence criterion for the difference between successive centrality vectors.

    Returns:
        tuple: A tuple containing a list of centrality values for all nodes and a list of nodes with the highest centrality values.
    """
    adj_matrix, node_names = adjacency_matrix(graph)
    n = len(adj_matrix)

    ketz_centrality = numpy.zeros(n)
    beta_array = numpy.full(n, beta)

    for _ in range(max_iter):
        new_ketz_centrality = alpha * \
            numpy.dot(adj_matrix, ketz_centrality) + beta_array

        if numpy.linalg.norm(new_ketz_centrality - ketz_centrality, 2) < tol:
            break

        ketz_centrality = new_ketz_centrality

    ketz_centrality /= numpy.linalg.norm(ketz_centrality)

    max_centrality = max(ketz_centrality)
    top_centrality_nodes = [(node_names[i], ketz_centrality[i])
                            for i in range(n) if ketz_centrality[i] == max_centrality]

    return dict(zip(node_names, ketz_centrality)), top_centrality_nodes


def betweenness_centrality(graph: Graph) -> tuple:
    """
    Computes the betweenness centrality for each node in the graph.

    Args:
        graph (Graph): The graph object representing the graph.

    Returns:
        tuple: A tuple containing a list of betweenness centrality values for all nodes and a list of nodes with the highest betweenness centrality values.
    """
    node_names = list(graph.nodes.keys())
    n = len(node_names)
    betweenness = numpy.zeros(n)

    for s in range(n):
        stack = []
        predecessors = [[] for _ in range(n)]
        num_shortest_paths = numpy.zeros(n)
        distance = numpy.full(n, -1)
        distance[s] = 0
        num_shortest_paths[s] = 1

        queue = [s]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for neighbor, _ in graph.nodes[node_names[v]].neighbours:
                w = node_names.index(neighbor)
                if distance[w] < 0:
                    queue.append(w)
                    distance[w] = distance[v] + 1
                if distance[w] == distance[v] + 1:
                    num_shortest_paths[w] += num_shortest_paths[v]
                    predecessors[w].append(v)

        delta = numpy.zeros(n)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (num_shortest_paths[v] /
                             num_shortest_paths[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    max_betweenness = numpy.max(betweenness)
    if max_betweenness > 0:
        betweenness /= max_betweenness

    top_betweenness_centrality_nodes = [node_names[i] for i in range(
        n) if betweenness[i] == numpy.max(betweenness)]

    return dict(zip(node_names, betweenness)), top_betweenness_centrality_nodes
