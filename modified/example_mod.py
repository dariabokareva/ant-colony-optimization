import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
import os

num_points = 40

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
# print(points_coordinate)
relevance = np.random.randint(1, 6, size=num_points)
print('Релевантности:', relevance)
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

s_max = 7

def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %% Do ACA
# from ant_colony import Ant_Colony

aca = Ant_Colony(func=cal_total_distance, n_dim=num_points,
                 size_pop=40, max_iter=20,
                 distance_matrix=distance_matrix,
                 s_max=s_max,
                 relevance=relevance)

best_x, best_y = aca.run()

if best_x != [] and best_y != 0: 
    print('Лучший маршрут:', best_x, 'Длительность:', best_y)

    # %% Plot
    fig, ax = plt.subplots(1, 2)
    best_points_ = best_x # np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    plt.show()
else: print("Локации с наименьшей релевантностью были исключены, с целью нахождения маршрута, соответствующего ограничениям.")
