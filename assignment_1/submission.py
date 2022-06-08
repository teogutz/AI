# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math
import sys

from math import sqrt

class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """
    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        #raise NotImplementedError
        popped = heapq.heappop(self.queue)
        heapq.heapify(self.queue)
        return popped

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        #raise NotImplementedError
        """
        if isinstance(node, (list, dict, tuple)):
            entry = 0
            for item in self.queue:
                if item[0] == node[0] and item[1] >= entry:
                    entry = item[1] + 1
            temp = list(node)
            node = temp
            node.insert(1,entry)
        """
        #print(node[0])
        self.queue.append(node)
        heapq.heapify(self.queue)
        #print(self.queue)
        #print(self.queue.index("[node[0],'']"))
        #self.queue[0][1] = node[1]
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


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
    if start == goal:
        return []


    inf = sys.maxsize

    node_data = dict.fromkeys(graph.nodes)

    for key in graph.nodes:
        node_data[key] = {'cost':inf, 'pred':[]}

    node_data[start]['cost'] = 0
    visited = []
    temp = start

    for i in range(len(graph.nodes) - 1):
        if temp not in visited:
            visited.append(temp)
            min_heap = []
            for j in graph[temp]:
                if j not in visited:
                    cost = node_data[temp]['cost'] + 1
                    if cost < node_data[j]['cost']:
                        node_data[j]['cost'] = cost
                        node_data[j]['pred'] = node_data[temp]['pred'] + list(temp)
                    heapq.heappush(min_heap,(node_data[j]['cost'],j))
        if len(min_heap) > 0:
            heapq.heapify(min_heap)
            temp = min_heap[0][1]
    
    return node_data[goal]['pred'] + list(goal)


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []

    inf = sys.maxsize
    
    node_data = dict.fromkeys(graph.nodes)
    
    for node in node_data:
        node_data[node] = {'weight': inf, 'pred': []}

    node_data[start]['weight'] = 0
    visited = []
    temp = start
    stack = []

    while temp != goal:
        if temp not in visited:
            visited.append(temp)
            pq = PriorityQueue()
            for neighbor in graph[temp]:
                if neighbor not in visited:
                    weight = node_data[temp]['weight'] + graph.get_edge_weight(temp,neighbor)
                    if weight < node_data[neighbor]['weight']:
                        node_data[neighbor]['weight'] = weight
                        node_data[neighbor]['pred'] = node_data[temp]['pred'] + list(temp)
                    pq.append((node_data[neighbor]['weight'], neighbor))
        if len(pq.queue) > 0:
            temp = pq.pop()[1]
            if len(pq.queue) > 0:
                stack.append(pq)
        else:
            pq = stack.pop()
            temp = pq.pop()[1]

    return node_data[goal]['pred'] + list(goal)


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    coord_v = graph.nodes[v]['pos']
    coord_g = graph.nodes[goal]['pos']

    return sqrt((coord_v[0]-coord_g[0])**2 + (coord_v[1]-coord_g[1])**2)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []

    inf = float(sys.maxsize)
    
    node_data = dict.fromkeys(graph.nodes)
    
    for node in node_data:
        node_data[node] = {'weight': inf, 'pred': []}

    node_data[start]['weight'] = 0
    visited = []
    temp = start

    pq = PriorityQueue()

    while temp != goal:
        if temp not in visited:
            visited.append(temp)
            for neighbor in graph[temp]:
                if neighbor not in visited:
                    weight = node_data[temp]['weight'] + graph.get_edge_weight(temp,neighbor) 
                    if weight < node_data[neighbor]['weight']:
                        node_data[neighbor]['weight'] = weight
                        node_data[neighbor]['pred'] = node_data[temp]['pred'] + list(temp)
                    pq.append((node_data[neighbor]['weight']+ heuristic(graph,neighbor,goal), neighbor))
        if len(pq.queue) > 0:
            temp = pq.pop()[1]

    return node_data[goal]['pred'] + list(goal)


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []

    inf = sys.maxsize
    
    node_data = dict.fromkeys(graph.nodes)
    node_data_back = dict.fromkeys(graph.nodes)
    
    for node in node_data:
        node_data[node] = {'weight': inf, 'pred': []}
        node_data_back[node] = {'weight': inf, 'pred': []}


    node_data[start]['weight'] = 0
    node_data_back[goal]['weight'] = 0
    
    visited = []
    visited_back = []
    fwd = start
    back = goal

    pq_back = PriorityQueue()
    pq_fwd = PriorityQueue()

    intersection = []

    while intersection == []:
        #forward
        if fwd not in visited:
            visited.append(fwd)
            intersection = [node for node in visited if node in visited_back]
            if len(intersection) > 0:
                break
            for neighbor in graph[fwd]:
                if neighbor not in visited:
                    weight = node_data[fwd]['weight'] + graph.get_edge_weight(fwd,neighbor)
                    if weight < node_data[neighbor]['weight']:
                        node_data[neighbor]['weight'] = weight
                        node_data[neighbor]['pred'] = node_data[fwd]['pred'] + [fwd]
                    pq_fwd.append((node_data[neighbor]['weight'], neighbor))
        if len(pq_fwd.queue) > 0:
            fwd = pq_fwd.pop()[1]        

        #backwards
        if back not in visited_back:
            visited_back.append(back)
            intersection = [node for node in visited if node in visited_back]
            if len(intersection) > 0:
                break
            #pq_back = PriorityQueue()
            for neighbor in graph[back]:
                if neighbor not in visited_back:
                    weight = node_data_back[back]['weight'] + graph.get_edge_weight(back,neighbor)
                    if weight < node_data_back[neighbor]['weight']:
                        node_data_back[neighbor]['weight'] = weight
                        node_data_back[neighbor]['pred'] = [back] + node_data_back[back]['pred']
                    pq_back.append((node_data_back[neighbor]['weight'], neighbor))
        #print("BACKWARDS")
        #print(pq_back)
        if len(pq_back.queue) > 0:
            back = pq_back.pop()[1]

        intersection = [node for node in visited if node in visited_back]
    
    result = []

    if len(node_data[intersection[0]]['pred']) > 0:
        result = node_data[intersection[0]]['pred']

    result.append(intersection[0])

    if len(node_data_back[intersection[0]]['pred']) > 0:
        result += node_data_back[intersection[0]]['pred']
    
    return result


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []

    inf = sys.maxsize
    
    node_data = dict.fromkeys(graph.nodes)
    node_data_back = dict.fromkeys(graph.nodes)
    
    for node in node_data:
        node_data[node] = {'weight': inf, 'pred': []}
        node_data_back[node] = {'weight': inf, 'pred': []}


    node_data[start]['weight'] = 0
    node_data_back[goal]['weight'] = 0
    
    visited = []
    visited_back = []
    fwd = start
    back = goal

    pq_back = PriorityQueue()
    pq_fwd = PriorityQueue()

    intersection = []

    while intersection == []:
        #forward
        if fwd not in visited:
            visited.append(fwd)
            intersection = [node for node in visited if node in visited_back]
            if len(intersection) > 0:
                break
            for neighbor in graph[fwd]:
                if neighbor not in visited:
                    weight = node_data[fwd]['weight'] + graph.get_edge_weight(fwd,neighbor)
                    if weight < node_data[neighbor]['weight']:
                        node_data[neighbor]['weight'] = weight
                        node_data[neighbor]['pred'] = node_data[fwd]['pred'] + [fwd]
                    pq_fwd.append((node_data[neighbor]['weight'] + heuristic(graph,neighbor,goal), neighbor))

        if len(pq_fwd.queue) > 0:
            fwd = pq_fwd.pop()[1]
        intersection = [node for node in visited if node in visited_back]
        if len(intersection) > 0:
            break

        #backwards
        if back not in visited_back:
            visited_back.append(back)
            intersection = [node for node in visited if node in visited_back]
            if len(intersection) > 0:
                break
            #pq_back = PriorityQueue()
            for neighbor in graph[back]:
                if neighbor not in visited_back:
                    weight = node_data_back[back]['weight'] + graph.get_edge_weight(back,neighbor)
                    if weight < node_data_back[neighbor]['weight']:
                        node_data_back[neighbor]['weight'] = weight
                        node_data_back[neighbor]['pred'] = [back] + node_data_back[back]['pred']
                    pq_back.append((node_data_back[neighbor]['weight'] + heuristic(graph,neighbor,goal), neighbor))
        #print("BACKWARDS")
        #print(pq_back)
        if len(pq_back.queue) > 0:
            back = pq_back.pop()[1]

        intersection = [node for node in visited if node in visited_back]
    
    result = []

    if len(node_data[intersection[0]]['pred']) > 0:
        result = node_data[intersection[0]]['pred']

    result.append(intersection[0])

    if len(node_data_back[intersection[0]]['pred']) > 0:
        result += node_data_back[intersection[0]]['pred']

    return result


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    raise NotImplementedError


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula


#################### TODO:BORRAR

import pickle
from explorable_graph import ExplorableGraph

class grafica:
    def __init__(self):
        self.nodes = ['A', 'B', 'C', 'D', 'E', 'F']
        self.edges = [
            ['B', 'C'],
            ['A', 'C', 'D'],
            ['A', 'B', 'E', 'D'],
            ['B', 'C', 'E', 'F'],
            ['C', 'D', 'F'],
            ['D', 'E']
        ]
    def __getitem__(self, item):
        return self.edges[self.nodes.index(item)]


def main():
    p = PriorityQueue()
    
    with open('romania_graph.pickle', 'rb') as rom:
        romania = pickle.load(rom)
    x = romania
    romania = ExplorableGraph(romania)
    romania.reset_search()


    with open('atlanta_osm.pickle', 'rb') as atl:
            atlanta = pickle.load(atl)
    atlanta = ExplorableGraph(atlanta)
    atlanta.reset_search()
         

    start = 'a'
    goal = 'u'
    #start = '69581003'
    #goal = '69581000'
    pq = PriorityQueue()

    pq.append((1,"b1"))

    graph = romania
    #graph = atlanta
    #print(breadth_first_search(graph, start, goal))
    #print(uniform_cost_search(graph, start, goal))
    #print(a_star(graph, start, goal))
    #print(a_star(graph, start, goal, null_heuristic))

    #print(bidirectional_ucs(graph, start, goal))
    #print(bidirectional_a_star(graph, start, goal))
    #print(romania.explored_nodes())
    #print(sum(list(romania.explored_nodes().values())))
    #print(breadth_first_search(grafica(), 'A', 'F'))
    #print(breadth_first_search(romania, 'c', 'u'))

main()