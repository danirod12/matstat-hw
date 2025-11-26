from ms_1_generator import *


def make_geom_table(geom_samples, sample_sizes, true_theta=theta_geom):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Оценки параметра $\theta$ для геометрического распределения}")
    lines.append(r"\begin{tabular}{|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{n} & \textbf{$\bar{X}$} & \textbf{$\hat{\theta}$} & \textbf{Ошибка} \\")
    lines.append(r"\hline")
    for n in sample_sizes:
        subs = [x[:n] for x in geom_samples]
        all_data = np.concatenate(subs)
        x_bar = np.mean(all_data)
        theta_hat = 1.0 / x_bar
        error = abs(theta_hat - true_theta)
        lines.append(rf"{n} & {x_bar:.4f} & {theta_hat:.4f} & {error:.4f} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_norm_table(norm_samples, sample_sizes, mu=mu, true_theta=sigma):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Оценки параметра $\theta$ для нормального распределения II}")
    lines.append(r"\begin{tabular}{|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{n} & \textbf{$\hat{\theta}_{мм}$} & \textbf{$\hat{\theta}_{ммп}$} & \textbf{Ошибка} \\")
    lines.append(r"\hline")
    for n in sample_sizes:
        subs = [x[:n] for x in norm_samples]
        all_data = np.concatenate(subs)
        # ММ-оценка: sqrt(1/N * sum (Xi - mu)^2)
        theta_hat_mm = np.sqrt(np.mean((all_data - mu) ** 2))
        # ММП-оценка совпадает с ММ при известном mu
        theta_hat_ml = theta_hat_mm
        error = abs(theta_hat_ml - true_theta)
        lines.append(
            rf"{n} & {theta_hat_mm:.4f} & {theta_hat_ml:.4f} & {error:.4f} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


geom_table_latex = make_geom_table(geometric_data, sample_sizes)
norm_table_latex = make_norm_table(normal_data, sample_sizes)

print("%%%%%%%% Геометрическое распределение %%%%%%%%")
print(geom_table_latex)
print()
print("%%%%%%%% Нормальное распределение II %%%%%%%%")
print(norm_table_latex)
