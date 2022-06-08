import pickle
import random
from explorable_graph import ExplorableGraph

from submission import PriorityQueue




queue = PriorityQueue()
temp_list = []

for _ in range(10):
    a = random.randint(0, 10000)
    queue.append((a, 'a'))
    temp_list.append(a)

print(temp_list)
print(queue)

temp_list = sorted(temp_list)
popped_2=[]



for item in temp_list:
    popped = queue.pop()
    print(popped[0], item)

"""
with open('atlanta_osm.pickle', 'rb') as atl:
    atlanta = pickle.load(atl)
y = atlanta
atlanta = ExplorableGraph(atlanta)
atlanta.reset_search()

with open('romania_graph.pickle', 'rb') as rom:
    romania = pickle.load(rom)
x = romania
romania = ExplorableGraph(romania)
romania.reset_search()

print(y.edges)



    def setUp(self):
        with open('atlanta_osm.pickle', 'rb') as atl:
            atlanta = pickle.load(atl)
        self.atlanta = ExplorableGraph(atlanta)
        self.atlanta.reset_search()

        with open('romania_graph.pickle', 'rb') as rom:
            romania = pickle.load(rom)
        self.romania = ExplorableGraph(romania)
        self.romania.reset_search()

    def test_bidirectional_ucs(self):
        path = bidirectional_ucs(self.atlanta, '69581003', '69581000')
        all_explored = self.atlanta.explored_nodes()
        plot_search(self.atlanta, 'atlanta_search_bidir_ucs.json', path,
                    all_explored)

"""