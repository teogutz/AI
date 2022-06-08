import heapq
import sys
import pickle
from explorable_graph import ExplorableGraph

def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    
    inf = sys.maxsize

    node_data = dict.fromkeys(graph.nodes)

    for key in node_data.keys():
        node_data[key] = {'cost':inf, 'pred':[]}
    
    node_data[start]['cost'] = 0
    visited = []
    temp = start
    min_heap = []

    for i in range(len(node_data)-1):
        if temp == goal:
            break
        if temp not in visited:
            visited.append(temp)
            #min_heap = []
            for j in graph[temp]:
                if j not in visited:
                    cost = node_data[temp]['cost'] + graph.get_edge_weight(temp, j)
                    if cost < node_data[j]['cost']:
                        node_data[j]['cost'] = cost
                        node_data[j]['pred'] = node_data[temp]['pred'] + list(temp)
                    heapq.heappush(min_heap,(node_data[j]['cost'],j))
        heapq.heapify(min_heap)
        temp = heapq.heappop(min_heap)[1]
        print(temp)

    return node_data[goal]['pred'] + list(goal)


def main():
    with open('romania_graph.pickle', 'rb') as rom:
        romania = pickle.load(rom)
    romania = ExplorableGraph(romania)
    romania.reset_search()

    graph = romania         

    start = 'a'
    goal = 'u'


    path = a_star(romania, start, goal)



    print(breadth_first_search(graph, start, goal))

main()