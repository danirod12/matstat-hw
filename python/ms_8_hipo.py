from ms_1_generator import *
import numpy as np
from scipy.stats import kstest, chisquare, chi2

# Функции для вычисления статистик и заполнения таблиц

def kolmogorov_smirnov_statistic(sample, cdf):
    # Вычисляет Dn (KS-статистику) и поправку Большева S
    sample_sorted = np.sort(sample)
    n = len(sample)
    cdf_vals = cdf(sample_sorted)
    ecdf = np.arange(1, n+1) / n
    D_plus = np.max(ecdf - cdf_vals)
    D_minus = np.max(cdf_vals - (np.arange(n) / n))
    Dn = max(D_plus, D_minus)
    S = (6*n*Dn + 1) / (6 * np.sqrt(n))
    return Dn, S

def chi_square_statistic(sample, probs, bins):
    # Подсчёт статистики хи-квадрат по группам bins и вероятностям probs для sample
    counts, _ = np.histogram(sample, bins=bins)
    expected = np.array(len(sample) * np.array(probs))
    # Сводим последние категории если ожидаемых < 5
    while np.any(expected < 5) and len(expected) > 1:
        expected[-2] += expected[-1]
        counts[-2] += counts[-1]
        expected = expected[:-1]
        counts = counts[:-1]
    chi2_stat, p_value = chisquare(counts, f_exp=expected)
    return chi2_stat, expected.size - 1  # возвращаем статистику и число степеней свободы

def geometric_cdf_factory(theta):
    def cdf(x):
        return 1 - (1-theta) ** np.floor(x)
    return cdf

def chi_square_threshold(df, alpha=0.05):
    return chi2.ppf(1-alpha, df)

def create_geometric_bins(sample, k):
    # Для дискретной геом. распределения интервалы это отдельные значения 1..k-1 и ">=k"
    max_val = int(np.max(sample))
    bins = list(range(1, k)) + [max_val+1]
    bins_edges = [0.5] + [x+0.5 for x in bins]  # для гистограммы, -0.5 + values +0.5
    probs = [(0.4 * (0.6)**(x-1)) for x in range(1, k)]
    probs.append((0.6)**(k-1))  # хвост
    bins_edges[-1] = np.inf
    return bins_edges, probs

# Основные данные
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]

def make_geometric_hypothesis_tables(samples, true_theta=0.4, alpha=0.05):
    lines_simple = []
    lines_complex = []
    cdf_true = geometric_cdf_factory(true_theta)

    for n in sample_sizes:
        # Объединяем 5 сгенерированных выборок по n элементов
        combined_sample = np.concatenate([x[:n] for x in samples])

        # Прямая гипотеза: theta известен
        Dn, S = kolmogorov_smirnov_statistic(combined_sample, cdf_true)

        k = int(np.ceil(1 + np.log2(n)))
        bins, probs = create_geometric_bins(combined_sample, k)
        chi2_stat, df = chi_square_statistic(combined_sample, probs, bins)
        chi2_thresh = chi_square_threshold(df, alpha)

        # Вывод по критериям
        decision_ks = "принимается" if Dn <= 1.36 / np.sqrt(n) else "отвергается"
        decision_chi2 = "принимается" if chi2_stat <= chi2_thresh else "отвергается"
        lines_simple.append(
            rf"{n} & {Dn:.4f} & {S:.4f} & {decision_ks} & {chi2_stat:.4f} & {chi2_thresh:.4f} & {decision_chi2} \\"
        )

        # Сложная гипотеза: theta неизвестен, оценён по выборке
        theta_hat = 1 / np.mean(combined_sample)
        cdf_hat = geometric_cdf_factory(theta_hat)
        Dn_hat, _ = kolmogorov_smirnov_statistic(combined_sample, cdf_hat)
        # Число степеней свободы для сложной гипотезы уменьшается на 1
        chi2_stat_hat, df_hat = chi_square_statistic(combined_sample, probs, bins)
        chi2_thresh_hat = chi_square_threshold(df_hat - 1, alpha)
        decision_ks_hat = "принимается" if Dn_hat <= 1.36 / np.sqrt(n) else "отвергается"
        decision_chi2_hat = "принимается" if chi2_stat_hat <= chi2_thresh_hat else "отвергается"
        lines_complex.append(
            rf"{n} & {theta_hat:.4f} & {Dn_hat:.4f} & {decision_ks_hat} & {chi2_stat_hat:.4f} & {chi2_thresh_hat:.4f} & {decision_chi2_hat} \\"
        )

    table_simple = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Проверка гипотез для геометрического распределения ($\theta = 0.4$)}",
        r"\begin{tabular}{|c|c|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{n} & \textbf{$D_n$} & \textbf{$S$} & \textbf{Выв. К} & \textbf{$\chi^2$} & \textbf{$\chi^2_{0.95}$} & \textbf{Выв. $\chi^2$} \\",
        r"\hline",
        *lines_simple,
        r"\hline",
        r"\multicolumn{7}{|c|}{\textit{Выв.: вывод (H0 отвергается/принимается)}} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    table_complex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Проверка сложной гипотезы для геометрического распределения}",
        r"\begin{tabular}{|c|c|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{n} & \textbf{$\hat{\theta}$} & \textbf{$D_n$} & \textbf{Выв. К} & \textbf{$\chi^2$} & \textbf{$\chi^2_{0.95}(k-2)$} & \textbf{Выв. $\chi^2$} \\",
        r"\hline",
        *lines_complex,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(table_simple), "\n".join(table_complex)


simple_table_latex, complex_table_latex = make_geometric_hypothesis_tables(geometric_data)

print("%%%%%%%% Проверка гипотез - простая гипотеза %%%%%%%%")
print(simple_table_latex)
print()
print("%%%%%%%% Проверка гипотез - сложная гипотеза %%%%%%%%")
print(complex_table_latex)
