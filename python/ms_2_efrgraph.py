from matplotlib import pyplot as plt

from ms_1_generator import *

# def plot_ecdf(data, sample_sizes, mu, sigma, dist_type='normal'):
#     plt.figure(figsize=(12, 6))
#
#     # Определяем диапазон для графика
#     t_values = np.linspace(data.min() - 1, data.max() + 1, 2500)
#
#     # Строим эмпирические функции для каждого размера выборки
#     for size in sample_sizes:
#         sample = data[:size]
#         sample_sorted = np.sort(sample)
#         ecdf = np.arange(1, len(sample_sorted) + 1) / len(sample_sorted)
#
#         plt.step(sample_sorted, ecdf, label=f'n={size}', where='post', alpha=0.7)
#
#     # Строим теоретическую функцию распределения
#     if dist_type == 'normal':
#         # Функция распределения нормального: Φ((x - μ) / σ)
#         from scipy.special import erf
#         z = (t_values - mu) / sigma
#         theoretical_cdf = 0.5 * (1 + erf(z / np.sqrt(2)))
#         plt.plot(t_values, theoretical_cdf, 'k-', linewidth=2, label='Теоретическая F(t)')
#         plt.title(f'Эмпирические функции распределения\nНормальное распределение N({mu}, {sigma}²)')
#     else:  # geometric
#         # Функция распределения геометрического: F(k) = 1 - (1-θ)^k
#         t_int = np.arange(1, int(data.max()) + 2)
#         theoretical_cdf = 1 - (1 - sigma)**t_int
#         plt.step(t_int, theoretical_cdf, 'k-', linewidth=2, label='Теоретическая F(t)', where='post')
#         plt.title(f'Эмпирические функции распределения\nГеометрическое распределение Geom({sigma})')
#
#     plt.xlabel('t')
#     plt.ylabel('F(t)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

def plot_ecdf(data_list, sample_sizes, mu, sigma, dist_type='normal'):
    # Определяем диапазон для графика на основе всех данных
    all_data = np.concatenate(data_list)

    # Цвета для разных сидов
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Создаем сетку графиков 3x3
    n_plots = len(sample_sizes)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols  # округление вверх

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()  # Преобразуем в одномерный массив

    for idx, size in enumerate(sample_sizes):
        ax = axes[idx]

        # Строим ЭФР для каждого seed
        for i, data in enumerate(data_list):
            sample = data[:size]
            sample_sorted = np.sort(sample)
            ecdf = np.arange(1, len(sample_sorted) + 1) / len(sample_sorted)

            ax.step(sample_sorted, ecdf,
                    label=f'seed={seeds[i]}',
                    where='post',
                    alpha=0.6,
                    color=colors[i],
                    linewidth=1.2)

        # Строим теоретическую функцию распределения
        if dist_type == 'normal':
            from scipy.special import erf
            t_values = np.linspace(all_data.min() - 1, all_data.max() + 1, 1000)
            z = (t_values - mu) / sigma
            theoretical_cdf = 0.5 * (1 + erf(z / np.sqrt(2)))
            ax.plot(t_values, theoretical_cdf, 'k-', linewidth=2, label='Теоретическая F(t)')
            ax.set_title(f'n={size}', fontsize=12, fontweight='bold')
        else:  # geometric
            t_int = np.arange(1, int(all_data.max()) + 2)
            theoretical_cdf = 1 - (1 - sigma) ** t_int
            ax.step(t_int, theoretical_cdf, 'k-', linewidth=2, label='Теоретическая F(t)', where='post')
            ax.set_title(f'n={size}', fontsize=12, fontweight='bold')

        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('F(t)', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Скрываем лишние оси, если количество графиков не кратно количеству столбцов
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    # Общий заголовок
    if dist_type == 'normal':
        fig.suptitle(f'Эмпирические функции распределения\nНормальное распределение N({mu}, {sigma}²)',
                     fontsize=16, fontweight='bold', y=0.995)
    else:
        fig.suptitle(f'Эмпирические функции распределения\nГеометрическое распределение Geom({sigma})',
                     fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


sample_sizes = sample_sizes + [50000]

# Построение графиков
print("График для нормального распределения:")
plot_ecdf(normal_data, sample_sizes, mu, sigma, dist_type='normal')

print("\nГрафик для геометрического распределения:")
plot_ecdf(geometric_data, sample_sizes, mu, theta_geom, dist_type='geometric')
