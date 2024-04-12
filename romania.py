from graph import Graph
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
    g = romania
    for node1 in g.nodes.values():
        for node2 in g.nodes.values():
            print(breadth_first_search(g, node1.name, node2.name))
            print(iterative_deepening_search(g, node1.name, node2.name, 30))
            print(uniform_cost_search(g, node1.name, node2.name))
            print(bidirectional_search(
                g, node1.name, node2.name, compute_heuristic))
            print(greedy_search(g, node1.name, node2.name, compute_heuristic))
            print(a_star_search(g, node1.name, node2.name, compute_heuristic))
            print(depth_first_search(g, node1.name, node2.name), '\n')
