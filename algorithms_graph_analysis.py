import time

from matplotlib import pyplot as plt
from search_algorithms import *
from utils import compute_heuristic, generate_random_graph

search_algorithms = {
    'bfs': breadth_first_search,
    'dfs': depth_first_search,
    'ucs': uniform_cost_search,
    'iterative_deepening': iterative_deepening_search,
    'bidirectional_search': bidirectional_search,
    'greedy': greedy_search,
    'astar': a_star_search
}


def run_algorithms_and_plot(graph_generator, num_nodes_list, edge_prob_list, num_trials=5):
    results = {}  # Store results for each graph configuration

    # Iterate over different graph configurations
    for num_nodes in num_nodes_list:
        for edge_prob in edge_prob_list:
            avg_times = {}  # Store average time for each algorithm
            avg_path_lengths = {}  # Store average path length for each algorithm

            # Perform multiple trials for robustness
            for _ in range(num_trials):
                # Generate a random graph
                graph = graph_generator(num_nodes, edge_prob)

                # Randomly select 10 nodes for path finding
                nodes = graph.select_random_nodes(10)

                # Evaluate each algorithm for path finding
                for algorithm_name, algorithm_func in search_algorithms.items():
                    total_time = 0
                    total_path_length = 0

                    # Measure time and path length for each pair of nodes
                    for node_start in nodes:
                        for node_end in nodes:
                            if node_start != node_end:
                                # Call the appropriate algorithm function
                                if algorithm_name in ['bfs', 'dfs', 'ucs']:
                                    start_time = time.time()
                                    path = algorithm_func(
                                        graph, node_start.name, node_end.name)
                                    end_time = time.time()
                                elif algorithm_name == 'iterative_deepening':
                                    start_time = time.time()
                                    path = algorithm_func(
                                        graph, node_start.name, node_end.name, num_nodes)
                                    end_time = time.time()
                                else:
                                    start_time = time.time()
                                    path = algorithm_func(
                                        graph, node_start.name, node_end.name, compute_heuristic)
                                    end_time = time.time()

                                # Accumulate time and path length if a path is found
                                if path:
                                    total_time += end_time - start_time
                                    total_path_length += len(path)

                    # Calculate average time and path length for the current algorithm
                    avg_times[algorithm_name] = avg_times.get(
                        algorithm_name, 0) + total_time
                    avg_path_lengths[algorithm_name] = avg_path_lengths.get(
                        algorithm_name, 0) + total_path_length

            # Calculate average time and path length across all trials
            for algorithm_name in avg_times:
                avg_times[algorithm_name] /= num_trials
                avg_path_lengths[algorithm_name] /= num_trials

            # Store results for the current graph configuration
            results[(num_nodes, edge_prob)] = (avg_times, avg_path_lengths)

    # Define a list of colors for each algorithm
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    # Plotting results for time
    plt.figure(figsize=(10, 6))
    for i, algorithm_name in enumerate(search_algorithms):
        avg_times = []
        x_positions = range(len(results))  # Use range for discrete markers

        # Collect average times for the current algorithm across different graph configurations
        for key in results:
            avg_times.append(results[key][0][algorithm_name])

        # Plot average times for the current algorithm
        plt.plot(x_positions, avg_times,
                 label=f'{algorithm_name} - Time', color=colors[i], linestyle='-', marker='o')

    plt.xlabel('Graph Configurations (Number of Nodes, Connection Probability)')
    plt.ylabel('Average Time (s)')
    plt.title('Performance of Search Algorithms on Random Graphs (Time)')
    plt.xticks(range(len(num_nodes_list) * len(edge_prob_list)),
               [(n, p) for n in num_nodes_list for p in edge_prob_list], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting results for path length
    plt.figure(figsize=(10, 6))
    for i, algorithm_name in enumerate(search_algorithms):
        avg_path_lengths = []
        x_positions = range(len(results))  # Use range for discrete markers

        # Collect average path lengths for the current algorithm across different graph configurations
        for key in results:
            avg_path_lengths.append(results[key][1][algorithm_name])

        # Plot average path lengths for the current algorithm
        plt.plot(x_positions, avg_path_lengths,
                 label=f'{algorithm_name} - Path Length', color=colors[i], linestyle='-', marker='o')

    plt.xlabel('Graph Configurations (Number of Nodes, Connection Probability)')
    plt.ylabel('Average Path Length')
    plt.title('Performance of Search Algorithms on Random Graphs (Path Length)')
    plt.xticks(range(len(num_nodes_list) * len(edge_prob_list)),
               [(n, p) for n in num_nodes_list for p in edge_prob_list], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    run_algorithms_and_plot(generate_random_graph, [
                            10, 20, 30, 40], [0.2, 0.4, 0.6, 0.8])
