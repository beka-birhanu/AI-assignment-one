from graph import Graph
from graph_centralities import *
from search_algorithms import *
from utils import compute_heuristic

romania = Graph()
with open('cities.txt', 'r') as file:
    for line in file:
        city_name, lattitued, longtiude = line.strip().split()
        if city_name == "City":
            continue

        romania.add_node(city_name, (float(lattitued), float(longtiude)))

with open('edges.txt', 'r') as file:
    for line in file:
        city1, city2, weight = line.strip().split()
        romania.add_edge(city1, city2, float(weight))


if __name__ == "__main__":
    for node1 in romania.nodes.values():
        for node2 in romania.nodes.values():
            if node1.name == node2.name:
                continue

            print(
                f"Breadth First Search: {breadth_first_search(romania, node1.name, node2.name)}\n")
            print(
                f"Iterative Deepening Search: {iterative_deepening_search(romania, node1.name, node2.name, 30)}\n")
            print(
                f"Uniform Cost Search: {uniform_cost_search(romania, node1.name, node2.name)}\n")
            print(
                f"Bidirectional Search: {bidirectional_search(romania, node1.name, node2.name, compute_heuristic)}\n")
            print(
                f"Greedy Search: {greedy_search(romania, node1.name, node2.name, compute_heuristic)}\n")
            print(
                f"A* Search: {a_star_search(romania, node1.name, node2.name, compute_heuristic)}\n")
            print(
                f"Depth First Search: {depth_first_search(romania, node1.name, node2.name)}\n\n")

    centrality, top_degree_nodes = degree_centrality(romania)
    print(f"Degree Centrality: {centrality}, Top Nodes: {top_degree_nodes}\n")

    centrality, top_closeness_nodes = closeness_centrality(romania)
    print(
        f"Closeness Centrality: {centrality}, Top Nodes: {top_closeness_nodes}\n")

    centrality, top_eigenvector_nodes = eigenvector_centrality(romania)
    print(
        f"Eigenvector Centrality: {centrality}, Top Nodes: {top_eigenvector_nodes}\n")

    centrality, top_pagerank_nodes = pagerank_centrality(romania)
    print(
        f"Pagerank Centrality: {centrality}, Top Nodes: {top_pagerank_nodes}\n")

    centrality, top_katz_nodes = katz_centrality(romania)
    print(f"Katz Centrality: {centrality}, Top Nodes: {top_katz_nodes}\n")

    centrality, top_betweenness_nodes = betweenness_centrality(romania)
    print(
        f"Betweenness Centrality: {centrality}, Top Nodes: {top_betweenness_nodes}\n")
