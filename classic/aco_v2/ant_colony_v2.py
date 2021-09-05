import numpy as np


class Ant_Colony:
    def __init__(self, func, n_dim,
                 size_pop=10, max_iter=20,
                 distance_matrix=None, relevance=None,
                 alpha=1, beta=2, rho=0.1,
                 ):
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.relevance = relevance

        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))

        self.Tau = np.ones((n_dim, n_dim))
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int)
        self.y = None
        self.x_best_history, self.y_best_history = [], []
        self.best_x, self.best_y = None, None

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta
            for j in range(self.size_pop):
                self.Table[j, 0] = 0
                for k in range(self.n_dim - 1):
                    taboo_set = set(self.Table[j, :k + 1])
                    allow_list = list(set(range(self.n_dim)) - taboo_set)
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            y = np.array([self.func(i) for i in self.Table])

            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.x_best_history.append(x_best)
            self.y_best_history.append(y_best)

            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):
                for k in range(self.n_dim - 1):
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]
                    delta_tau[n1, n2] += 1 / y[j]
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]
                delta_tau[n1, n2] += 1 / y[j]

            self.Tau = (1 - self.rho) * self.Tau + delta_tau

        best_generation = np.array(self.y_best_history).argmin()
        self.best_x = self.x_best_history[best_generation]
        self.best_y = self.y_best_history[best_generation]
        return self.best_x, self.best_y

    fit = run
