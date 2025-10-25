import numpy as np


seeds = [ 500, 501, 502, 503, 504 ]


# Бокс-Мюллер для нормального
def generate_normal(mu, theta, n=1000):
    U1 = np.random.uniform(0, 1, n)
    U2 = np.random.uniform(0, 1, n)
    Z = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    X = mu + theta * Z
    return X


# Обратное преобразование геометрическое
def generate_geometric(theta, n=1000):
    U = np.random.uniform(0, 1, n)
    X = np.ceil(np.log(1 - U) / np.log(1 - theta))
    return X


# Параметры
mu = 22.5
sigma = 4.0
theta_geom = 0.4
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]

normal_data = []
geometric_data = []
for seed in seeds:
    np.random.seed(seed)
    normal_data += [generate_normal(mu, sigma, n=50000)]
    np.random.seed(seed)
    geometric_data += [generate_geometric(theta_geom, n=50000)]
