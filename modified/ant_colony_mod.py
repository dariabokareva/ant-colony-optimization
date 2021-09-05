import numpy as np
 
 
class Ant_Colony:
    def __init__(self, func, n_dim, s_max,
                 size_pop=10, max_iter=20,
                 distance_matrix=None, relevance=None,
                 alpha=1, beta=2, rho=0.9,
                 ):
        self.func = func # Функция минимизации
        self.n_dim = n_dim # Количество городов
        self.size_pop = size_pop # Популяция муравьёв
        self.max_iter = max_iter # Максимальное число итераций
        self.alpha = alpha # Коэффициент влияния феромонного усиления, при alpha = 0 ACO сводится к простому жадному алгоритму
        self.beta = beta # Коэффициент влияния видимости, при beta = 0 маршруты вырождаются к одному субоптимальному решению
        self.rho = rho # коэффициент испарения феромона
        self.relevance = relevance # массив релевантностей локация
        self.s_max = s_max # Максимальная длительность маршрута
        self.s_max_start = s_max
        self.distance_matrix = distance_matrix # Матрица попарных расстояний между локациями
 
        self.Tau = np.ones((n_dim, n_dim)) # матрица начальных значений феромона
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int)
        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))
 
        self.y = None
        self.x_best_history, self.y_best_history = [], []
        self.best_x, self.best_y = [], 0. #None, None
 
    def run(self, max_iter=None):
        
        self.max_iter = max_iter or self.max_iter
        
        # Нормирование релевантности
        relevance = np.copy(self.relevance)
        relevance_sum = relevance.sum() #  Sr
        norm_relevance = []
            
        for item in range(len(self.relevance)):
            norm_relevance.append(relevance[item] / relevance_sum)
        norm_relevance = np.copy(norm_relevance)
 
        for i in range(self.max_iter):
            # print('---------------------->Итерация номер ', i+1, '\n')
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta
 
            for j in range(self.size_pop):
                self.Table[j, 0] = 0
                self.Table[j, self.n_dim - 1] = self.n_dim - 1
                for k in range(self.n_dim - 1):
                    taboo_set = set(self.Table[j, :k + 1]) 
                    allow_list = list(set(range(self.n_dim)) - taboo_set) 
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k+1] = next_point # k+1
            y = np.array([self.func(i) for i in self.Table])
            index_best = y.argmin()
            index_last = self.n_dim - 1
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.x_best_history.append(x_best)
            self.y_best_history.append(y_best)
 
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):
                funcm = 0
                # print('Муравей под номером: ', j+1, '\n')
                for k in range(self.n_dim  - 1):
                    funcm += norm_relevance[k]
                    n1 = self.Table[j, k]
                    n2 = self.Table[j, k + 1]
                    delta_tau[n1, n2] += (self.s_max * pow(funcm, 1./self.rho)) / (y[j] * max(norm_relevance[0:k+1])) # Old: delta_tau[n1, n2] += 1 / y[j]
                    
                # n1, n2 = self.Table[j, self.n_dim - 2], self.Table[j, self.n_dim - 1]  
                # delta_tau[n1, n2] += (self.s_max * pow(funcm, 1./self.rho)) / (y[j] * max(norm_relevance[0:k+1]))
                    # print('Длина пройденного:', y[j])
                    # print('Нормированная релевантность ', k+1, 'города: ', norm_relevance[k])
                    # print('Суммируем релевантности пройденных городов, шаг ', k+1, ', сумма:', funcm)
                    # print('Макс F на шаге:', max(norm_relevance[0:k+1]))
                    # print('dTau на шаге = ', delta_tau[n1, n2], '\n')
 
            self.Tau = (1 - self.rho) * self.Tau + delta_tau
            best_generation = np.array(self.y_best_history).argmin()
            self.best_x = self.x_best_history[best_generation]
            self.best_y = self.y_best_history[best_generation]
 
            if self.best_x is None or self.best_y is None:
                self.best_x, self.best_y = 0, 0

            # Если прошли все итерации, а найденный маршрут длинее s_max
            # ДОБАВИТЬ УСЛОВИЕ: или если все муравьи в рамках одной итерации прошли по одному маршруту
            if (i == self.max_iter - 1) and (y[j] > self.s_max): 
                print('Не найден оптимальный маршрут, повторяю поиск...')
                relevance = list(relevance[:])
                new_relevance = relevance[1:-1] # берём релевантности, без начальной и конечной точки
                if not new_relevance or self.n_dim == 2:
                    best_generation = np.array(self.y_best_history).argmin()
                    
                    self.best_x = self.x_best_history[best_generation]
                    self.best_y = self.y_best_history[best_generation]
                    
                    if self.best_x is None or self.best_y is None:
                        self.best_x, self.best_y = [], 0
 
                    return self.best_x, self.best_y
                else:
                    min_relevance = min(new_relevance) # ищем минимальную релевантность
                    min_index = new_relevance.index(min_relevance)
                    del relevance[min_index+1]
                    print('Оставшиеся значения релевантностей:', relevance)
                    '''берём индекс, чтобы по нему найти эту точку в distance_matrix
                    и удаляем город оттуда, num points сокращаем на единицу и нужно заново запустить run''' 
                    distance_matrix = list(self.distance_matrix[:])
                    del distance_matrix[min_index+1]
                
                    new_distance_matrix = []
                
                    for el in distance_matrix:
                        el = list(el)
                        del el[min_index + 1]
                        new_distance_matrix.append(list(el))
 
                    new_n_dim = self.n_dim - 1
                    aca = Ant_Colony(func=self.func, n_dim=new_n_dim,
                                     size_pop=self.size_pop, max_iter=self.max_iter,
                                     distance_matrix=new_distance_matrix,
                                     s_max=self.s_max_start,
                                     relevance=relevance)
                    # best_x, best_y = [], 0.
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
                    return [], 0
 
                    #print('Лучший маршрут:', result[0], 'Длительность:', result[1])
            #elif i != self.max_iter - 1: continue
            elif y[j] <= self.s_max:
            #else:
                return self.best_x, self.best_y
    fit = run
