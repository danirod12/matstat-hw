from scipy.stats import kstest, chisquare, chi2, norm
from ms_1_generator import *


def kolmogorov_smirnov_statistic(sample, cdf):
    sample_sorted = np.sort(sample)
    n = len(sample)
    cdf_vals = cdf(sample_sorted)
    ecdf = np.arange(1, n+1) / n
    D_plus = np.max(ecdf - cdf_vals)
    D_minus = np.max(cdf_vals - (np.arange(n) / n))
    Dn = max(D_plus, D_minus)
    S = (6*n*Dn + 1) / (6 * np.sqrt(n))
    return Dn, S

def chi_square_statistic(sample, bins, probs):
    counts, _ = np.histogram(sample, bins=bins)
    expected = np.array(len(sample) * np.array(probs))
    while np.any(expected < 5) and len(expected) > 1:
        expected[-2] += expected[-1]
        counts[-2] += counts[-1]
        expected = expected[:-1]
        counts = counts[:-1]
    chi2_stat, p_value = chisquare(counts, f_exp=expected)
    return chi2_stat, len(expected) - 1

def get_equal_prob_bins(sample, k):
    # Разбиваем данные на k равновероятных интервалов (квантили)
    quantiles = np.quantile(sample, np.linspace(0, 1, k + 1))
    # Вероятности равны 1/k
    probs = np.full(k, 1.0 / k)
    return quantiles, probs

def chi_square_threshold(df, alpha=0.05):
    return chi2.ppf(1-alpha, df)

def make_normal_hypothesis_tables(norm_samples, mu=22.5, theta=4.0, alpha=0.05):
    lines_simple = []
    lines_complex = []
    cdf_true = lambda x: norm.cdf(x, loc=mu, scale=theta)

    for n in sample_sizes:
        combined_sample = np.concatenate([x[:n] for x in norm_samples])

        # Простая гипотеза: известны параметры
        Dn, S = kolmogorov_smirnov_statistic(combined_sample, cdf_true)

        k = int(np.ceil(1 + np.log2(n)))
        bins, probs = get_equal_prob_bins(combined_sample, k)
        chi2_stat, df = chi_square_statistic(combined_sample, bins, probs)
        chi2_thresh = chi_square_threshold(df, alpha)

        decision_ks = "принимается" if Dn <= 1.36 / np.sqrt(n) else "отвергается"
        decision_chi2 = "принимается" if chi2_stat <= chi2_thresh else "отвергается"
        lines_simple.append(
            rf"{n} & {Dn:.4f} & {S:.4f} & {decision_ks} & {chi2_stat:.4f} & {chi2_thresh:.4f} & {decision_chi2} \\"
        )

        # Сложная гипотеза: параметры неизвестны, оцениваем по выборке
        mu_hat = np.mean(combined_sample)
        theta_hat = np.sqrt(np.mean((combined_sample - mu_hat) ** 2))
        cdf_hat = lambda x: norm.cdf(x, loc=mu_hat, scale=theta_hat)
        Dn_hat, _ = kolmogorov_smirnov_statistic(combined_sample, cdf_hat)

        # Для сложной гипотезы df уменьшается на 2 (оценка mu и theta)
        chi2_stat_hat, df_hat = chi_square_statistic(combined_sample, bins, probs)
        chi2_thresh_hat = chi_square_threshold(df_hat - 2, alpha)
        decision_ks_hat = "принимается" if Dn_hat <= 1.36 / np.sqrt(n) else "отвергается"
        decision_chi2_hat = "принимается" if chi2_stat_hat <= chi2_thresh_hat else "отвергается"
        lines_complex.append(
            rf"{n} & {mu_hat:.4f} & {theta_hat:.4f} & {Dn_hat:.4f} & {decision_ks_hat} & {chi2_stat_hat:.4f} & {decision_chi2_hat} \\"
        )

    table_simple = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Проверка гипотез для нормального распределения II ($\mu = 22.5$, $\theta = 4.0$)}",
        r"\begin{tabular}{|c|c|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{n} & \textbf{$D_n$} & \textbf{$S$} & \textbf{Выв. К} & \textbf{$\chi^2$} & \textbf{$\chi^2_{0.95}$} & \textbf{Выв. $\chi^2$} \\",
        r"\hline",
        *lines_simple,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    table_complex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Проверка сложной гипотезы для нормального распределения II}",
        r"\begin{tabular}{|c|c|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{n} & \textbf{$\hat{\mu}$} & \textbf{$\hat{\theta}$} & \textbf{$D_n$} & \textbf{Выв. К} & \textbf{$\chi^2$} & \textbf{Выв. $\chi^2$} \\",
        r"\hline",
        *lines_complex,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(table_simple), "\n".join(table_complex)

simple_table_norm, complex_table_norm = make_normal_hypothesis_tables(normal_data)

print("%%%%%%%% Проверка гипотез - простая гипотеза (нормальное) %%%%%%%%")
print(simple_table_norm)
print()
print("%%%%%%%% Проверка гипотез - сложная гипотеза (нормальное) %%%%%%%%")
print(complex_table_norm)
