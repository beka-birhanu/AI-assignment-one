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
        

        

        # Final degree Centrality
        return degree_centralities,top_degree_centralities

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
        nodes = list(graph.nodes.keys())
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
        

        
        return centralities,topcloseness_centralities

    def eigenvector_centrality(self,graph,iter=200,tolerance=1e-6):
        nodes = list(graph.nodes.keys())
        adj_matrix = self.adjacency_matrix(graph)
        centrality_scores = numpy.ones(len(nodes))
        centrality_scores /= numpy.linalg.norm(centrality_scores)
        
        
        for i in range(iter):
            new_centrality = numpy.dot(adj_matrix, centrality_scores)

         
            new_centrality /= numpy.linalg.norm(new_centrality)

           
            if numpy.linalg.norm(new_centrality - centrality_scores, 2) < tolerance:
                break

            centrality_scores = new_centrality

        top_eigenvector_centralities = []
        top = max(centrality_scores)

        for i in range(len(centrality_scores)):
            if centrality_scores[i] == top:
                top_eigenvector_centralities.append(nodes[i])
        

        
        return centrality_scores,top_eigenvector_centralities


    def pagerank_centrality(self,graph, d=0.85, max_iter=100, tolerance=1e-6):
        nodes = list(graph.nodes.keys())
        
        adj_matrix = numpy.array(self.adjacency_matrix(graph))
        n = len(adj_matrix)

      
        centrality_scores = numpy.ones(n) / n

        for i in range(max_iter):
           
            sums = adj_matrix.sum(axis=1, keepdims=True)
            transition_matrix = numpy.where(sums != 0, adj_matrix / sums, 1 / n)

            
            new_centrality_scores = (1 - d) / n + d * numpy.dot(transition_matrix.T, centrality_scores)

           
            if numpy.linalg.norm(new_centrality_scores - centrality_scores, 2) < tolerance:
                break

            pagerank_scores = new_centrality_scores

        top_pagerank_centralities = []
        top = max(pagerank_scores)

        for i in range(len(pagerank_scores)):
            if pagerank_scores[i] == top:
                top_pagerank_centralities.append(nodes[i])
        

        return pagerank_scores,top_pagerank_centralities

    def katz_centrality(self,graph, a=0.1, b=1.0, max_iter=100, tol=1e-6):
        
        adj_matrix = self.adjacency_matrix(graph)
        nodes = list(graph.nodes.keys())
        n = len(nodes)
        
    
        katz_centrality = numpy.zeros(n)
        b = numpy.full(n, b)

        for _ in range(max_iter):
            
            new_centrality = a * numpy.dot(adj_matrix, katz_centrality) + b

            if numpy.linalg.norm(new_centrality - katz_centrality, 2) < tol:
                break

            katz_centrality = new_centrality

        katz_centrality /= numpy.linalg.norm(katz_centrality)
        

        top = max(katz_centrality)
        top_katz_centralities = []
        for i in range(len(katz_centrality)):
            if katz_centrality[i] == top:
                top_katz_centralities.append(nodes[i])
                
        return katz_centrality,top_katz_centralities

    def betweenness_centrality(self,graph):

        adj_matrix = self.adjacency_matrix(graph)
        vertices = list(graph.nodes.keys())
        n = len(vertices)
        
        betweenness = numpy.zeros(n)  # Initialize betweenness centrality scores
        
        for i in range(n):  # Iterate over all nodes as potential sources
            stack = []  # Stack for DFS
            predecessors = [[] for _ in range(n)]  # List of predecessors for each node
            num_shortest_paths = numpy.zeros(n)  # Number of shortest paths from s to each node
            distance = numpy.full(n, -1)  # Distance from s to each node
            distance[i] = 0  # Distance from s to s is 0
            num_shortest_paths[i] = 1  # There is one shortest path from s to s

            # BFS to find shortest paths and count them
            queue = [i]
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
                if w != i:
                    betweenness[w] += delta[w]

        
        # Normalize betweenness centrality scores
        max_betweenness = numpy.max(betweenness)
        if max_betweenness > 0:
            betweenness /= max_betweenness

    
        top_betweenness_centrality_nodes = [vertices[i] for i in range(n) if betweenness[i] == numpy.max(betweenness)]
        print("The Vertices with Highest betweenness centrality are", *top_betweenness_centrality_nodes)

        return betweenness,top_betweenness_centrality_nodes




    
    