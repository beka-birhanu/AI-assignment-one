import utils
import networkx as nx


from collections import deque
import numpy
class GraphCentrality:
   
    def adjacency_matrix(self,graph):
        node_names = list(graph.nodes.keys())
        adj_matrix = numpy.zeros((len(node_names), len(node_names)))
        
        # Construct the adjacency matrix
        for i, node_name in enumerate(node_names):
            node = graph.nodes[node_name]
            for neighbor_name, weight in node.neighbours:
                j = node_names.index(neighbor_name)
                adj_matrix[i, j] = weight
            
        return adj_matrix
    def degree_centrality(self,graph):
        degree_centralities = {}
        number_of_nodes = len(graph.nodes)
        
        
        # Iterate through a hash map that contains node and its neighbours
        for node in graph.nodes:
            degree_centralities[node] = len(graph.nodes[node].neighbours)/number_of_nodes
            
        #Initialized list that holds nodes with degree centrality
        top_degree_centralities = []
        top = max(degree_centralities.values())

        for centrality in degree_centralities:
            if degree_centralities[centrality] == top:
                top_degree_centralities.append(centrality)
        

        print("The Vertices with Highest degree centrality are", *top_degree_centralities)
        

        # Final degree Centrality
        return degree_centralities

    def floyd_warshall(self,adj_matrix):
        n = len(adj_matrix)
        dist = [[float('inf') if i != j and adj_matrix[i][j] == 0 else adj_matrix[i][j] for j in range(n)] for i in range(n)]
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        return dist

    def closeness_centrality(self,graph):
        adj_matrix = self.adjacency_matrix(graph)
        nodes = list(graph.keys())
        dist_matrix = self.floyd_warshall(adj_matrix)
        n = len(adj_matrix)
        centralities = []
        
        for i in range(n):
            reachable_nodes = sum(1 for d in dist_matrix[i] if d != float('inf') and d != 0)
            total_distance = sum(d for d in dist_matrix[i] if d != float('inf') and d != 0)
            
            if total_distance == 0:
                centralities.append(0.0)
            else:
                centralities.append(reachable_nodes / total_distance)
        
        topcloseness_centralities = []
        top = max(centralities)

        for i in range(len(centralities)):
            if centralities[i]== top:
                topcloseness_centralities.append(nodes[i])
        

        print("The Vertices with Highest closeness centrality are", *topcloseness_centralities)
        
        return centralities

    def eigenvector_centrality(self,graph):
        adj_matrix = self.adjacency_matrix(graph)
        eigenvalues, eigenvectors = numpy.linalg.eig(adj_matrix)
        
        # Find the eigenvector corresponding to the largest eigenvalue
        max_eigenvalue_index = numpy.argmax(eigenvalues)
        max_eigenvector = numpy.abs(eigenvectors[:, max_eigenvalue_index])
        
        # Normalize the eigenvector
        max_eigenvector /= max_eigenvector.sum()
        
        # Create a dictionary mapping node names to eigenvector centrality scores
        node_names = list(graph.nodes.keys())
        centrality_scores = {node_names[i]: max_eigenvector[i] for i in range(len(node_names))}

        top_eigenvector_centralities = []
        top = max(centrality_scores.values())

        for centrality in centrality_scores:
            if centrality_scores[centrality] == top:
                top_eigenvector_centralities.append(centrality)
        

        print("The Vertices with Highest eiganvector centrality are", *top_eigenvector_centralities)

        
        return centrality_scores

    def pagerank(self,graph, d=0.85, max_iter=100, tolerance=1e-6):
        nodes = list(graph.nodes.keys())
        # Convert the graph into an adjacency matrix
        adjacency_matrix = self.adjacency_matrix(graph)

        # Convert the adjacency matrix to a NumPy array
        adjacency_matrix = numpy.array(adjacency_matrix)

        # Get the number of nodes in the graph
        N = len(adjacency_matrix)

        # Initialize PageRank scores with equal weights
        pagerank_scores = numpy.ones(N) / N

        for _ in range(max_iter):
            # Normalize the adjacency matrix to represent transition probabilities
            row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
            transition_matrix = numpy.where(row_sums != 0, adjacency_matrix / row_sums, 1 / N)

            # Calculate the next iteration of PageRank scores
            new_pagerank_scores = (1 - d) / N + d * numpy.dot(transition_matrix.T, pagerank_scores)

            # Check for convergence
            if numpy.linalg.norm(new_pagerank_scores - pagerank_scores, 2) < tolerance:
                break

            pagerank_scores = new_pagerank_scores

        top_pagerank_centralities = []
        top = max(pagerank_scores)

        for i in range(len(pagerank_scores)):
            if pagerank_scores[i] == top:
                top_pagerank_centralities.append(nodes[i])
        

        print("The Vertices with Highest page rank centrality are", *top_pagerank_centralities)

        return pagerank_scores

    def katz_centrality(self,graph, alpha=0.1, beta=1.0, max_iter=100, tol=1e-6):
        # Convert the graph into an adjacency matrix
        adjacency_matrix = self.adjacency_matrix(graph)
        nodes = list(graph.nodes.keys())

        n = len(adjacency_matrix)
        
        # Initialize centrality scores
        centrality = numpy.zeros(n)
        beta = numpy.full(n, beta)

        for _ in range(max_iter):
            # Update centrality scores using the Katz centrality equation
            new_centrality = alpha * numpy.dot(adjacency_matrix, centrality) + beta

            # Check for convergence
            if numpy.linalg.norm(new_centrality - centrality, 2) < tol:
                break

            centrality = new_centrality

        # Normalize centrality scores
        centrality /= numpy.linalg.norm(centrality)
        
        # Identify top-ranked nodes
        top_katz_centralities = []
        top = numpy.max(centrality)

        for i in range(len(centrality)):
            if centrality[i] == top:
                top_katz_centralities.append(nodes[i])
        
        print("The Vertices with Highest katz centrality are", *top_katz_centralities)
                
        return centrality

    def betweenness_centrality(self,graph):
        adj_matrix = self.adjacency_matrix(graph)
        vertices = list(graph.nodes.keys())
        n = len(vertices)
        
        betweenness = numpy.zeros(n)  # Initialize betweenness centrality scores
        
        for s in range(n):  # Iterate over all nodes as potential sources
            stack = []  # Stack for DFS
            predecessors = [[] for _ in range(n)]  # List of predecessors for each node
            num_shortest_paths = numpy.zeros(n)  # Number of shortest paths from s to each node
            distance = numpy.full(n, -1)  # Distance from s to each node
            distance[s] = 0  # Distance from s to s is 0
            num_shortest_paths[s] = 1  # There is one shortest path from s to s

            # BFS to find shortest paths and count them
            queue = [s]
            while queue:
                v = queue.pop(0)
                stack.append(v)
                for w in range(n):
                    if adj_matrix[v][w] != 0:  # There is an edge (v, w)
                        if distance[w] < 0:  # w is visited for the first time
                            queue.append(w)
                            distance[w] = distance[v] + 1
                        if distance[w] == distance[v] + 1:  # w is adjacent to v on a shortest path
                            num_shortest_paths[w] += num_shortest_paths[v]
                            predecessors[w].append(v)

            # Accumulate dependencies
            delta = numpy.zeros(n)  # Dependency of each node
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    delta[v] += (num_shortest_paths[v] / num_shortest_paths[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        
        # Normalize betweenness centrality scores
        max_betweenness = numpy.max(betweenness)
        if max_betweenness > 0:
            betweenness /= max_betweenness

            
        top_betweenness_centrality_nodes = [vertices[i] for i in range(n) if betweenness[i] == numpy.max(betweenness)]
        print("The Vertices with Highest betweenness centrality are", *top_betweenness_centrality_nodes)

        return betweenness

    
    