from matplotlib import pyplot as plt
from ms_1_generator import *


def plot_frequency_polygon_normal(data_list, sample_sizes, mu, sigma):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # ФИКСИРОВАННЫЙ диапазон для всех графиков
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 2500)
    theoretical_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    for idx, size in enumerate(sample_sizes):
        # Определяем единую сетку bins на основе всех 5 выборок
        all_data_for_size = np.concatenate([data[:size] for data in data_list])
        bins = np.linspace(all_data_for_size.min(), all_data_for_size.max(), 21)  # 20 интервалов

        # Собираем плотности для каждой выборки
        all_densities = []
        for data in data_list:
            sample = data[:size]
            counts, _ = np.histogram(sample, bins=bins)
            # Вычисляем высоты: h_j = count_j / (n * delta_j)
            bin_widths = np.diff(bins)
            densities = counts / (size * bin_widths)
            all_densities.append(densities)

        # Усредняем плотности
        avg_densities = np.mean(all_densities, axis=0)

        # Строим усредненную гистограмму
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = np.diff(bins)
        axes[idx].bar(bin_centers, avg_densities, width=bin_widths,
                      alpha=0.6, color='blue', edgecolor='black',
                      label='Эмпирическое')

        # Теоретическая кривая
        axes[idx].plot(x, theoretical_pdf, 'r-', linewidth=2,
                       label='Теоретическое f(x)')

        axes[idx].set_title(f'n = {size}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('Плотность')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(f'Полигон частот для нормального распределения N({mu}, {sigma}²)\n(усреднено по 5 выборкам)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_frequency_polygon_geometric(data_list, sample_sizes, theta):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, size in enumerate(sample_sizes):
        # Находим все уникальные значения из всех 5 выборок
        all_values_set = set()
        for data in data_list:
            sample = data[:size]
            all_values_set.update(sample)
        all_values = sorted(all_values_set)

        # Для каждого значения собираем частоты из всех 5 выборок
        freq_dict = {val: [] for val in all_values}

        for data in data_list:
            sample = data[:size]
            values, counts = np.unique(sample, return_counts=True)
            frequencies = dict(zip(values, counts / size))

            # Для каждого возможного значения записываем частоту (или 0)
            for val in all_values:
                freq_dict[val].append(frequencies.get(val, 0))

        # Усредняем частоты
        avg_frequencies = {val: np.mean(freq_dict[val]) for val in all_values}

        # Строим усредненный полигон частот
        values_sorted = sorted(avg_frequencies.keys())
        freqs_sorted = [avg_frequencies[v] for v in values_sorted]

        axes[idx].bar(values_sorted, freqs_sorted, alpha=0.6, color='blue',
                      edgecolor='black', label='Эмпирическое', width=0.8)

        # Теоретическая функция вероятности
        k_max = int(max(all_values)) + 5
        k_values = np.arange(1, k_max)
        theoretical_pmf = theta * (1 - theta) ** (k_values - 1)
        axes[idx].plot(k_values, theoretical_pmf, 'ro-', linewidth=2,
                       markersize=5, label='Теоретическое P(X=k)')

        axes[idx].set_title(f'n = {size}')
        axes[idx].set_xlabel('k')
        axes[idx].set_ylabel('Вероятность')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(f'Полигон частот для геометрического распределения Geom({theta})\n(усреднено по 5 выборкам)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Построение графиков
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]

print("Полигон частот для нормального распределения (усредненный):")
plot_frequency_polygon_normal(normal_data, sample_sizes, mu, sigma)

print("\nПолигон частот для геометрического распределения (усредненный):")
plot_frequency_polygon_geometric(geometric_data, sample_sizes, theta_geom)
