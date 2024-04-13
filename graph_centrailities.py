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
    
    