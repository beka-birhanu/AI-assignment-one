import random
from typing import Dict, List, Optional, Set, Tuple


class Node:
    """
    Represents a node in the graph.

    Attributes:
        name (str): The name/identifier of the node.
        coordinates (Tuple[int, int]): The geographic coordinates (latitude, longitude) of the node.
        neighbours (Set[Tuple[str, float]]): A set of tuples representing neighbouring nodes and their edge weights.
    """

    def __init__(self, name: str, coordinates: Tuple[int, int]) -> None:
        self.name = name
        self.coordinates = coordinates
        self.neighbours: Set[Tuple[str, float]] = set()


class Graph:
    """
    Represents a graph data structure.

    Attributes:
        nodes (Dict[str, Node]): A dictionary containing node names as keys and corresponding Node objects as values.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}

    def add_node(self, name: str, coordinates: Tuple[int, int]) -> None:
        """
        Adds a new node to the graph.

        Args:
            name (str): The name/identifier of the node.
            coordinates (Tuple[int, int]): The geographic coordinates (latitude, longitude) of the node.

        Raises:
            ValueError: If a node with the same name already exists in the graph.
        """
        if name in self.nodes:
            raise ValueError(f'A node with name "{name}" already exists')

        new_node = Node(name, coordinates)
        self.nodes[name] = new_node

    def delete_node(self, name: str) -> None:
        """
        Deletes a node from the graph.

        Args:
            name (str): The name/identifier of the node to be deleted.

        Raises:
            ValueError: If the node does not exist in the graph.
        """
        if name not in self.nodes:
            raise ValueError(f'A node with name "{name}" does not exist')

        node_to_be_deleted = self.nodes[name]

        for nbr, weight in node_to_be_deleted.neighbours:
            nbr_node = self.nodes[nbr]
            nbr_node.neighbours.remove((name, weight))

        del self.nodes[name]

    def add_edge(self, start: str, end: str, weight: float) -> None:
        """
        Adds a new edge between two nodes in the graph.

        Args:
            start (str): The name of the starting node.
            end (str): The name of the ending node.
            weight (float): The weight/cost of the edge.

        Raises:
            ValueError: If the start or end node does not exist in the graph, or if the weight is negative.
        """
        if start not in self.nodes or end not in self.nodes:
            raise ValueError('Nodes do not exist in the graph')

        if weight < 0:
            raise ValueError('Weight must be a non-negative number')

        start_node = self.nodes[start]
        end_node = self.nodes[end]

        start_node.neighbours.add((end_node.name, weight))
        end_node.neighbours.add((start_node.name, weight))

    def delete_edge(self, start: str, end: str, weight: float) -> None:
        """
        Deletes an edge between two nodes in the graph.

        Args:
            start (str): The name of the starting node.
            end (str): The name of the ending node.
            weight (float): The weight/cost of the edge.

        Raises:
            ValueError: If the start or end node does not exist in the graph, or if the specified edge does not exist.
        """
        if start not in self.nodes or end not in self.nodes:
            raise ValueError('Nodes do not exist in the graph')

        start_node = self.nodes[start]
        end_node = self.nodes[end]

        if (end_node.name, weight) not in start_node.neighbours or (start_node.name, weight) not in end_node.neighbours:
            raise ValueError('No such Edge')

        start_node.neighbours.remove((end_node.name, weight))
        end_node.neighbours.remove((start_node.name, weight))

    def select_random_nodes(self, num_nodes: int) -> List[Optional[Node]]:
        """
        Selects a random subset of nodes from the graph.

        Args:
            num_nodes (int): The number of nodes to select.

        Returns:
            List[Optional[Node]]: A list of randomly selected nodes.

        Raises:
            ValueError: If the number of nodes requested exceeds the total number of nodes in the graph.
        """
        if num_nodes > len(self.nodes):
            raise ValueError(
                'Number of nodes requested exceeds total number of nodes in the graph')

        return random.sample(list(self.nodes.values()), num_nodes)
