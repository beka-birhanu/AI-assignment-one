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
        
        return centralities
    
    