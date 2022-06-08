import heapq

queue_1=[]
queue_1=heapq.heapify(queue_1)
temp_list = []

for _ in range(10):
    a = random.randint(0, 10000)
    heapq.heappush(queue_1, a)
    temp_list.append(a)


temp_list = sorted(temp_list)