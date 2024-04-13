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
    
    