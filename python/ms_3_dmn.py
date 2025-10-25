from itertools import combinations
import pandas as pd
from ms_1_generator import *


def ecdf_value(sample, x):
    return np.sum(sample <= x) / len(sample)


def compute_dmn(sample_n, sample_m):
    n = len(sample_n)
    m = len(sample_m)

    # Объединяем все уникальные точки из обеих выборок
    all_points = np.unique(np.concatenate([sample_n, sample_m]))

    # Вычисляем супремум разности
    max_diff = 0
    for x in all_points:
        fn = ecdf_value(sample_n, x)
        fm = ecdf_value(sample_m, x)
        diff = abs(fn - fm)
        if diff > max_diff:
            max_diff = diff

    # Вычисляем D_{m,n}
    dmn = np.sqrt((n * m) / (n + m)) * max_diff
    return dmn


def compute_all_dmn_averaged(data_list, sample_sizes):
    # Словарь для накопления значений D_{m,n}
    dmn_accumulator = {}

    # Для каждой пары размеров выборок
    for n, m in combinations(sample_sizes, 2):
        dmn_values = []

        # Вычисляем D_{m,n} для каждого seed
        for data in data_list:
            sample_n = data[:n]
            sample_m = data[:m]
            dmn = compute_dmn(sample_n, sample_m)
            dmn_values.append(dmn)

        # Усредняем по 5 реализациям
        dmn_avg = np.mean(dmn_values)
        dmn_accumulator[(n, m)] = dmn_avg

    return dmn_accumulator


# Размеры выборок
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000, 1001]

# Настройка pandas для отображения всех строк и столбцов
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.3f}'.format)  # 3 знака после запятой

# Вычисление усредненных D_{m,n} для нормального распределения
print("\n" + "=" * 60)
print("Усредненные D_{m,n} для нормального распределения (по 5 выборкам):")
print("=" * 60)
normal_dmn_dict = compute_all_dmn_averaged(normal_data, sample_sizes)

# Создание матрицы для нормального распределения
normal_matrix = pd.DataFrame(index=sample_sizes, columns=sample_sizes, dtype=float)
for (n, m), dmn_avg in normal_dmn_dict.items():
    normal_matrix.loc[n, m] = dmn_avg
    normal_matrix.loc[m, n] = dmn_avg  # Симметрично

# Диагональ = 0 (сравнение с самим собой)
for size in sample_sizes:
    normal_matrix.loc[size, size] = 0.0

print(normal_matrix)
print("\n")

# Вычисление усредненных D_{m,n} для геометрического распределения
print("=" * 60)
print("Усредненные D_{m,n} для геометрического распределения (по 5 выборкам):")
print("=" * 60)
geometric_dmn_dict = compute_all_dmn_averaged(geometric_data, sample_sizes)

# Создание матрицы для геометрического распределения
geometric_matrix = pd.DataFrame(index=sample_sizes, columns=sample_sizes, dtype=float)
for (n, m), dmn_avg in geometric_dmn_dict.items():
    geometric_matrix.loc[n, m] = dmn_avg
    geometric_matrix.loc[m, n] = dmn_avg  # Симметрично

# Диагональ = 0 (сравнение с самим собой)
for size in sample_sizes:
    geometric_matrix.loc[size, size] = 0.0

print(geometric_matrix)
print("\n")

# Дополнительно: сохранение в LaTeX формат для отчета
print("=" * 60)
print("LaTeX код для таблиц:")
print("=" * 60)
print("\nНормальное распределение:")
print(normal_matrix.to_latex(float_format="%.3f"))
print("\nГеометрическое распределение:")
print(geometric_matrix.to_latex(float_format="%.3f"))
