import math
import random
import uuid
from graph import Graph, Node


# Earth radius in kilometers
EARTH_RADIUS = 6371.0


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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points
    on the Earth's surface using the Haversine formula.
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * \
        math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = EARTH_RADIUS * c

    return distance


def compute_heuristic(node1: Node, node2: Node) -> float:
    """
    Compute the Haversine distance between two nodes.
    Assumes the coordinates of the nodes are latitude and longitude values.
    """
    if not hasattr(node1, 'coordinates') or not hasattr(node2, 'coordinates'):
        raise ValueError("Nodes must have 'coordinates' attribute")

    try:
        lat1, lon1 = node1.coordinates
        lat2, lon2 = node2.coordinates
    except (TypeError, ValueError):
        raise ValueError(
            "Coordinates must be iterable with at least two elements (latitude and longitude)")

    return haversine_distance(lat1, lon1, lat2, lon2)
