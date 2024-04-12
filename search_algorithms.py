from collections import deque
from typing import List
from graph import Graph


def breadth_first_search(graph: Graph, start: str, end: str) -> List[str]:
    """
    Perform breadth-first search traversal on a graph to find the shortest path
    between a given start and end node.

    Args:
        graph (Graph): The graph to be traversed.
        start (str): The name of the start node.
        end (str): The name of the end node.

    Returns:
        List[str]: The shortest path from start to end node.

    """
    # Queue for BFS traversal
    queue = deque([start])

    # Dictionary to store the parent node of each visited node
    parent_map = {start: None}

    while queue:
        curr_node_name = queue.popleft()

        # Check if the current node is the destination node
        if curr_node_name == end:
            break

        # Traverse the neighbors of the current node
        for neighbor_name, _ in graph.nodes[curr_node_name].neighbours:
            if neighbor_name not in parent_map:
                parent_map[neighbor_name] = curr_node_name
                queue.append(neighbor_name)

    # If the destination node was not found
    if end not in parent_map:
        return []

    # Reconstruct the path from start to end
    path = []
    curr_node_name = end
    while curr_node_name is not None:
        path.append(curr_node_name)
        curr_node_name = parent_map[curr_node_name]

    # Reverse the path to get it from start to end
    return path[::-1]
