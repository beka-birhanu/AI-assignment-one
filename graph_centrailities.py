import numpy


def adjacency_matrix(graph):
    node_names = list(graph.nodes.keys())
    adj_matrix = numpy.zeros((len(node_names), len(node_names)))

    # Construct the adjacency matrix
    for i, node_name in enumerate(node_names):
        node = graph.nodes[node_name]
        for neighbor_name, weight in node.neighbours:
            j = node_names.index(neighbor_name)
            adj_matrix[i, j] = weight

    return adj_matrix


def degree_centrality(graph):
    degree_centralities = {}
    number_of_nodes = len(graph.nodes)

    # Calculate degree centrality for each node
    for node_name, node in graph.nodes.items():
        degree_centralities[node_name] = len(node.neighbours) / number_of_nodes

    # Find nodes with highest degree centrality
    max_centrality = max(degree_centralities.values())
    top_degree_centralities = [node_name for node_name, centrality in degree_centralities.items(
    ) if centrality == max_centrality]

    return degree_centralities, top_degree_centralities


def floyd_warshall(adj_matrix):
    n = len(adj_matrix)
    dist = [[float('inf') if i != j and adj_matrix[i][j] ==
             0 else adj_matrix[i][j] for j in range(n)] for i in range(n)]

    # Apply Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def closeness_centrality(graph):
    # Calculate the adjacency matrix
    adj_matrix = adjacency_matrix(graph)
    nodes = list(graph.nodes.keys())

    # Compute the shortest path distances using Floyd-Warshall algorithm
    dist_matrix = floyd_warshall(adj_matrix)
    n = len(adj_matrix)
    centralities = []

    # Calculate closeness centrality for each node
    for i in range(n):
        reachable_nodes = sum(
            1 for d in dist_matrix[i] if d != float('inf') and d != 0)
        total_distance = sum(
            d for d in dist_matrix[i] if d != float('inf') and d != 0)

        if total_distance == 0:
            centralities.append(0.0)
        else:
            centralities.append(reachable_nodes / total_distance)

    # Find nodes with highest closeness centrality
    top_closeness_centralities = []
    top = max(centralities)

    for i in range(len(centralities)):
        if centralities[i] == top:
            top_closeness_centralities.append(nodes[i])

    return centralities, top_closeness_centralities


def eigenvector_centrality(graph, max_iter=200, tolerance=1e-6):
    # Get the list of nodes and the adjacency matrix
    nodes = list(graph.nodes.keys())
    adj_matrix = adjacency_matrix(graph)

    # Initialize centrality scores
    centrality_scores = numpy.ones(len(nodes))
    centrality_scores /= numpy.linalg.norm(centrality_scores)

    # Perform power iteration method
    for i in range(max_iter):
        new_centrality_scores = numpy.dot(adj_matrix, centrality_scores)
        new_centrality_scores /= numpy.linalg.norm(new_centrality_scores)

        # Check for convergence
        if numpy.linalg.norm(new_centrality_scores - centrality_scores, 2) < tolerance:
            break

        centrality_scores = new_centrality_scores

    # Find nodes with highest eigenvector centrality
    top_eigenvector_centralities = []
    top = max(centrality_scores)

    for i in range(len(centrality_scores)):
        if centrality_scores[i] == top:
            top_eigenvector_centralities.append(nodes[i])

    return centrality_scores, top_eigenvector_centralities


def pagerank_centrality(graph, d=0.85, max_iter=100, tolerance=1e-6):
    # Get the list of nodes and the adjacency matrix
    nodes = list(graph.nodes.keys())
    adj_matrix = numpy.array(adjacency_matrix(graph))
    length = len(nodes)

    # Initialize centrality scores
    centrality_scores = numpy.ones(length) / length

    # Perform PageRank iteration
    for i in range(max_iter):
        sums = adj_matrix.sum(keepdims=True, axis=1)
        transition = numpy.where(numpy.logical_and(sums != 0, numpy.isnan(
            sums), ~numpy.isnan(adj_matrix)), adj_matrix / sums, 1 / length)

        new_centrality_scores = (1 - d) / length + \
            d * numpy.dot(transition.T, centrality_scores)

        # Check for convergence
        if numpy.linalg.norm(new_centrality_scores - centrality_scores, 2) < tolerance:
            break

        centrality_scores = new_centrality_scores

    # Find nodes with highest PageRank centrality
    top_pagerank_centralities = []
    top = max(centrality_scores)

    for i in range(len(centrality_scores)):
        if centrality_scores[i] == top:
            top_pagerank_centralities.append(nodes[i])

    return centrality_scores, top_pagerank_centralities


def katz_centrality(graph, a=0.1, b=1.0, max_iter=100, tol=1e-6):
    # Get the adjacency matrix and list of nodes
    adj_matrix = adjacency_matrix(graph)
    nodes = list(graph.nodes.keys())
    n = len(nodes)

    # Initialize Katz centrality scores
    katz_centrality = numpy.zeros(n)
    b = numpy.full(n, b)

    # Perform Katz centrality iteration
    for _ in range(max_iter):
        new_centrality = a * numpy.dot(adj_matrix, katz_centrality) + b

        # Check for convergence
        if numpy.linalg.norm(new_centrality - katz_centrality, 2) < tol:
            break

        katz_centrality = new_centrality

    # Normalize Katz centrality scores
    katz_centrality /= numpy.linalg.norm(katz_centrality)

    # Find nodes with highest Katz centrality
    top = max(katz_centrality)
    top_katz_centralities = [nodes[i] for i, centrality in enumerate(
        katz_centrality) if centrality == top]

    return katz_centrality, top_katz_centralities


def betweenness_centrality(graph):
    # Get the adjacency matrix and list of vertices
    adj_matrix = adjacency_matrix(graph)
    vertices = list(graph.nodes.keys())
    n = len(vertices)

    # Initialize betweenness centrality scores
    betweenness = numpy.zeros(n)

    # Iterate over all vertices
    for i in range(n):
        stack = []  # Stack for DFS
        predecessors = [[] for _ in range(n)]
        num_shortest_paths = numpy.zeros(n)
        distance = numpy.full(n, -1)
        distance[i] = 0
        num_shortest_paths[i] = 1

        queue = [i]
        # Perform BFS to find shortest paths and predecessors
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in range(n):
                if adj_matrix[v][w] != 0:
                    if distance[w] < 0:
                        queue.append(w)
                        distance[w] = distance[v] + 1
                    if distance[w] == distance[v] + 1:
                        num_shortest_paths[w] += num_shortest_paths[v]
                        predecessors[w].append(v)

        # Calculate delta values for each vertex
        delta = numpy.zeros(n)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (num_shortest_paths[v] /
                             num_shortest_paths[w]) * (1 + delta[w])
            if w != i:
                betweenness[w] += delta[w]

    # Normalize betweenness centrality scores
    max_betweenness = numpy.max(betweenness)
    if max_betweenness > 0:
        betweenness /= max_betweenness

    # Find vertices with highest betweenness centrality
    top_betweenness_centrality_nodes = [vertices[i] for i in range(
        n) if betweenness[i] == numpy.max(betweenness)]

    return betweenness, top_betweenness_centrality_nodes
