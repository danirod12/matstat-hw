from itertools import combinations

from scipy.stats import ks_2samp
from ms_1_generator import *


# Функция для подсчёта статистики Смирнова (двухвыборочный KS-тест)
def smirnov_statistic(sample1, sample2):
    result = ks_2samp(sample1, sample2)
    D = result.statistic
    p_value = result.pvalue
    decision = "однородны" if p_value > 0.05 else "не однородны"
    return D, decision

def make_smirnov_tables(discrete_samples, continuous_samples, pairs):
    """
    :param discrete_samples: список numpy массивов для геом. распределения (5 сгенерированных выборок)
    :param continuous_samples: список numpy массивов для нормального распределения
    :param pairs: список пар (n1, n2) для сравнения
    :return: latex коды двух таблиц
    """
    lines_geometric = []
    lines_normal = []
    for n1, n2 in pairs:
        # Объединяем все 5 сэмплов каждого объема n1 и n2 для дискретных
        sample1_geom = np.concatenate([s[:n1] for s in discrete_samples])
        sample2_geom = np.concatenate([s[:n2] for s in discrete_samples])
        D_geom, dec_geom = smirnov_statistic(sample1_geom, sample2_geom)
        lines_geometric.append(rf"{n1} & {n2} & {D_geom:.4f} & {dec_geom} \\")

        # Объединяем для нормальных
        sample1_norm = np.concatenate([s[:n1] for s in continuous_samples])
        sample2_norm = np.concatenate([s[:n2] for s in continuous_samples])
        D_norm, dec_norm = smirnov_statistic(sample1_norm, sample2_norm)
        lines_normal.append(rf"{n1} & {n2} & {D_norm:.4f} & {dec_norm} \\")

    table_geom = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Статистика Смирнова для геометрического распределения}",
        r"\begin{tabular}{|c|c|c|c|}",
        r"\hline",
        r"\textbf{$n_1$} & \textbf{$n_2$} & \textbf{$D_{n_1,n_2}$} & \textbf{Вывод однородности} \\",
        r"\hline",
        *lines_geometric,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}"
    ]

    table_norm = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Статистика Смирнова для нормального распределения II}",
        r"\begin{tabular}{|c|c|c|c|}",
        r"\hline",
        r"\textbf{$n_1$} & \textbf{$n_2$} & \textbf{$D_{n_1,n_2}$} & \textbf{Вывод однородности} \\",
        r"\hline",
        *lines_normal,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}"
    ]

    return "\n".join(table_geom), "\n".join(table_norm)

sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]

# Генерация всех пар (ni, nj), где i < j
pairs_to_check = list(combinations(sample_sizes, 2))

geom_table_latex, norm_table_latex = make_smirnov_tables(geometric_data, normal_data, pairs_to_check)

print("%%% Таблица статистики Смирнова геометрическое %%%")
print(geom_table_latex)
print()
print("%%% Таблица статистики Смирнова нормальное %%%")
print(norm_table_latex)
