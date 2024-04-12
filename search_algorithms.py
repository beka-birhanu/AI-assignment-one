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


def depth_first_search(graph: Graph, start: str, end: str, max_depth: Optional[int] = float('inf')) -> List[str]:
    """
    Perform depth-first search traversal on a graph to find a path between a given start and end node
    within a specified maximum depth.

    Args:
        graph (Graph): The graph to be traversed.
        start (str): The name of the start node.
        end (str): The name of the end node.
        max_depth (Optional[int]): The maximum depth to explore during traversal. Defaults to infinity.

    Returns:
        List[str]: The path from start to end node, if one exists within the maximum depth.
    """
    # Stack for DFS traversal
    stack = [(start, 0)]
    # Dictionary to store the parent node of each visited node
    parent_map = {start: None}

    while stack:
        curr_node_name, curr_depth = stack.pop()
        curr_node = graph.nodes[curr_node_name]

        if curr_depth > max_depth:
            continue  # Skip nodes beyond the maximum depth

        if curr_node_name == end:
            break  # Terminate the loop if the destination node is found

        # Traverse the neighbors of the current node
        for neighbor_name, _ in curr_node.neighbours:
            if neighbor_name not in parent_map:
                parent_map[neighbor_name] = curr_node_name
                stack.append((neighbor_name, curr_depth + 1))

    # If the destination node was not found
    if end not in parent_map:
        return []

    # Reconstruct the path from end to start
    path = []
    curr_node_name = end
    while curr_node_name:
        path.append(curr_node_name)
        curr_node_name = parent_map.get(curr_node_name)

    # Reverse the path to get it from start to end
    return path[::-1]
