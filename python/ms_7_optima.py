from ms_1_generator import *


def make_geom_opt_table(geom_samples, sample_sizes, true_theta=theta_geom):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Оптимальные оценки для геометрического распределения}")
    lines.append(r"\begin{tabular}{|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{n} & \textbf{$\hat{\tau}_{opt} = \bar{X}$} & \textbf{$\hat{\theta} = 1/\bar{X}$} \\")
    lines.append(r"\hline")
    for n in sample_sizes:
        subs = [x[:n] for x in geom_samples]
        all_data = np.concatenate(subs)
        x_bar = np.mean(all_data)
        tau_hat = x_bar                 # оптимальная оценка для 1/theta
        theta_hat = 1.0 / x_bar         # оценка для theta
        lines.append(
            rf"{n} & {tau_hat:.4f} & {theta_hat:.4f} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\multicolumn{3}{|c|}{\textbf{Истинные значения:} $\tau(\theta) = 2.5$, $\theta = 0.4$} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_norm_opt_table(norm_samples, sample_sizes, mu=mu, true_theta=sigma):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Оптимальные оценки для нормального распределения II}")
    lines.append(r"\begin{tabular}{|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{n} & \textbf{$\hat{\tau}_{opt} = \frac{1}{n}\sum(X_i-\mu)^2$} & \textbf{$\hat{\theta} = \sqrt{\hat{\tau}_{opt}}$} \\")
    lines.append(r"\hline")
    for n in sample_sizes:
        subs = [x[:n] for x in norm_samples]
        all_data = np.concatenate(subs)
        tau_hat = np.mean((all_data - mu) ** 2)   # оптимальная оценка для theta^2
        theta_hat = np.sqrt(tau_hat)              # естественная оценка для theta
        lines.append(
            rf"{n} & {tau_hat:.4f} & {theta_hat:.4f} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\multicolumn{3}{|c|}{\textbf{Истинные значения:} $\tau(\theta) = 16.0$, $\theta = 4.0$} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# генерация LaTeX для вставки
geom_opt_table_latex = make_geom_opt_table(geometric_data, sample_sizes)
norm_opt_table_latex = make_norm_opt_table(normal_data, sample_sizes)

print("%%%%%%%% Оптимальные оценки: геометрическое %%%%%%%%")
print(geom_opt_table_latex)
print()
print("%%%%%%%% Оптимальные оценки: нормальное II %%%%%%%%")
print(norm_opt_table_latex)
