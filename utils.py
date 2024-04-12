import random
import uuid
from graph import Graph


def generate_random_graph(num_nodes: int, edge_prob: float) -> Graph:
    """
    Generates a random graph with the given number of nodes and edge probability.

    Args:
        num_nodes (int): The number of nodes in the graph.
        edge_prob (float): The probability of an edge existing between any pair of nodes.

    Returns:
        Graph: The randomly generated graph.
    """
    nodes = [(str(uuid.uuid4()), (random.uniform(0, 100), random.uniform(0, 100)))
             for _ in range(num_nodes)]
    edges = []

    for i, (name1, _) in enumerate(nodes):
        for j, (name2, _) in enumerate(nodes):
            if i != j and random.random() < edge_prob:
                weight = random.uniform(1, 10)
                edges.append((name1, name2, weight))

    graph = Graph()
    for name, coordinates in nodes:
        graph.add_node(name, coordinates)

    for name1, name2, weight in edges:
        graph.add_edge(name1, name2, weight)

    return graph
