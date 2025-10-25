import pandas as pd
from matplotlib import pyplot as plt
from ms_1_generator import *


def compute_sample_moments_averaged(data_list, sample_sizes):
    results = []
    for size in sample_sizes:
        means = []
        variances = []

        # Вычисляем моменты для каждой из 5 выборок
        for data in data_list:
            sample = data[:size]
            means.append(np.mean(sample))
            variances.append(np.var(sample, ddof=0))

        # Усредняем по 5 реализациям
        avg_mean = np.mean(means)
        avg_var = np.mean(variances)

        # Опционально: вычисляем стандартные отклонения для отображения разброса
        std_mean = np.std(means, ddof=1)
        std_var = np.std(variances, ddof=1)

        results.append((size, avg_mean, avg_var, std_mean, std_var))

    return results


sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]

# Истинные параметры
# Нормальное распределение N(μ, θ²)
true_mean_normal = mu
true_var_normal = sigma ** 2

# Геометрическое распределение Geom(θ)
true_mean_geometric = 1 / theta_geom
true_var_geometric = (1 - theta_geom) / (theta_geom ** 2)

# Настройка pandas
pd.set_option('display.float_format', '{:.3f}'.format)

# Вычисление усредненных моментов для нормального распределения
print("=" * 80)
print("Усредненные выборочные моменты для нормального распределения N(22.5, 16)")
print("(усреднено по 5 выборкам)")
print(f"Истинные значения: E[X] = {true_mean_normal}, Var[X] = {true_var_normal}")
print("=" * 80)
normal_moments = compute_sample_moments_averaged(normal_data, sample_sizes)
normal_moments_df = pd.DataFrame(normal_moments,
                                 columns=['n', 'X̄', 'S²', 'SD(X̄)', 'SD(S²)'])
print(normal_moments_df.to_string(index=False))

# Вычисление усредненных моментов для геометрического распределения
print("\n" + "=" * 80)
print("Усредненные выборочные моменты для геометрического распределения Geom(0.4)")
print("(усреднено по 5 выборкам)")
print(f"Истинные значения: E[X] = {true_mean_geometric}, Var[X] = {true_var_geometric}")
print("=" * 80)
geometric_moments = compute_sample_moments_averaged(geometric_data, sample_sizes)
geometric_moments_df = pd.DataFrame(geometric_moments,
                                    columns=['n', 'X̄', 'S²', 'SD(X̄)', 'SD(S²)'])
print(geometric_moments_df.to_string(index=False))

# Визуализация сходимости выборочного среднего
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График для нормального распределения
axes[0].plot(normal_moments_df['n'], normal_moments_df['X̄'], 'bo-',
             label='Выборочное среднее X̄', linewidth=2, markersize=6)
axes[0].axhline(y=true_mean_normal, color='r', linestyle='--',
                linewidth=2, label=f'E[X] = {true_mean_normal}')
axes[0].set_xlabel('Размер выборки n', fontsize=11)
axes[0].set_ylabel('Значение', fontsize=11)
axes[0].set_title('Сходимость выборочного среднего\n(Нормальное распределение)',
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# График для геометрического распределения
axes[1].plot(geometric_moments_df['n'], geometric_moments_df['X̄'], 'go-',
             label='Выборочное среднее X̄', linewidth=2, markersize=6)
axes[1].axhline(y=true_mean_geometric, color='r', linestyle='--',
                linewidth=2, label=f'E[X] = {true_mean_geometric}')
axes[1].set_xlabel('Размер выборки n', fontsize=11)
axes[1].set_ylabel('Значение', fontsize=11)
axes[1].set_title('Сходимость выборочного среднего\n(Геометрическое распределение)',
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Усредненные по 5 выборкам результаты', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# Визуализация сходимости выборочной дисперсии
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(normal_moments_df['n'], normal_moments_df['S²'], 'bo-',
             label='Выборочная дисперсия S²', linewidth=2, markersize=6)
axes[0].axhline(y=true_var_normal, color='r', linestyle='--',
                linewidth=2, label=f'D[X] = {true_var_normal}')
axes[0].set_xlabel('Размер выборки n', fontsize=11)
axes[0].set_ylabel('Значение', fontsize=11)
axes[0].set_title('Сходимость выборочной дисперсии\n(Нормальное распределение)',
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(geometric_moments_df['n'], geometric_moments_df['S²'], 'go-',
             label='Выборочная дисперсия S²', linewidth=2, markersize=6)
axes[1].axhline(y=true_var_geometric, color='r', linestyle='--',
                linewidth=2, label=f'D[X] = {true_var_geometric}')
axes[1].set_xlabel('Размер выборки n', fontsize=11)
axes[1].set_ylabel('Значение', fontsize=11)
axes[1].set_title('Сходимость выборочной дисперсии\n(Геометрическое распределение)',
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Усредненные по 5 выборкам результаты', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# Дополнительно: LaTeX таблицы для отчета
print("\n" + "=" * 80)
print("LaTeX код для таблиц:")
print("=" * 80)
print("\nНормальное распределение:")
print(normal_moments_df[['n', 'X̄', 'S²']].to_latex(index=False, float_format="%.3f"))
print("\nГеометрическое распределение:")
print(geometric_moments_df[['n', 'X̄', 'S²']].to_latex(index=False, float_format="%.3f"))
