import numpy as np

from ant_colony import AntColony

cities = {0: "Киров", 1: "Советск", 2: "Яранск", 3: "Орлов", 4: "Котельнич"}

distances = np.array([[np.inf, 10, 2, 7, 9],
                      [10, np.inf, 4, 8, 2],
                      [2, 4, np.inf, 1, 3],
                      [7, 8, 1, np.inf, 2],
                      [9, 2, 3, 2, np.inf]])

ant_colony = AntColony(distances, 10, 3, 20, 0.7, alpha=1, beta=1)
shortest_path = ant_colony.run()


shortest_list = list(map(list, shortest_path[0]))
for item in shortest_list:
    item[0] = cities[item[0]]
    item[1] = cities[item[1]]

print("Кратчайший маршрут: {}".format(shortest_list), shortest_path[1])
