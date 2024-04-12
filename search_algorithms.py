from collections import deque
from typing import Dict, List, Optional
from graph import Graph
from heapq import heappop as pop, heappush as push


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


def uniform_cost_search(graph: Graph, start: str, end: str) -> List[str]:
    """
    Perform uniform-cost search traversal on a graph to find the shortest path
    between a given start and end node.

    Args:
        graph (Graph): The graph to be traversed.
        start (str): The name of the start node.
        end (str): The name of the end node.

    Returns:
        List[str]: The shortest path from start to end node.

    """
    # Priority queue for UCS traversal
    priority_queue = [(0, start)]
    # Dictionary to store the parent node and path cost of each visited node
    parent_map: Dict[str, Optional[str]] = {start: None}
    path_cost_map: Dict[str, float] = {start: 0}

    while priority_queue:
        curr_path_cost, curr_node_name = pop(priority_queue)

        # Check if the current node is the destination node
        if curr_node_name == end:
            break

        # Traverse the neighbors of the current node
        for neighbor_name, cost in graph.nodes[curr_node_name].neighbours:
            total_cost = curr_path_cost + cost

            # Update parent and path cost if the neighbor is not visited or found a shorter path
            if neighbor_name not in parent_map or total_cost < path_cost_map[neighbor_name]:
                parent_map[neighbor_name] = curr_node_name
                path_cost_map[neighbor_name] = total_cost
                push(priority_queue, (total_cost, neighbor_name))

    # If the destination node was not found
    if end not in parent_map:
        return []

    # Reconstruct the path from end to start
    path = []
    curr_node_name = end
    while curr_node_name:
        path.append(curr_node_name)
        curr_node_name = parent_map[curr_node_name]

    # Reverse the path to get it from start to end
    return path[::-1]


def iterative_deepening_search(graph: Graph, start: str, end: str, max_depth: Optional[int] = 20) -> List[str]:
    """
    Perform iterative deepening depth-first search traversal on a graph to find a path
    between a given start and end node within a specified maximum depth.

    Args:
        graph (Graph): The graph to be traversed.
        start (str): The name of the start node.
        end (str): The name of the end node.
        max_depth (Optional[int]): The maximum depth to explore during each iteration. Defaults to 20.

    Returns:
        List[str]: The path from start to end node, if one exists within the specified maximum depth.
    """
    for current_depth in range(max_depth + 1):
        path = depth_first_search(graph, start, end, current_depth)

        if path:
            return path

    # If the destination node is unreachable within the specified maximum depth
    return []
