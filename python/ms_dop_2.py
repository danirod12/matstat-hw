# =============================================================================
# Раздел 4.3: Проверка статистических гипотез для реальных данных
# ИСПРАВЛЕННАЯ ВЕРСИЯ
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import kstest, chi2, norm

# ============================================
# Часть 1: Геометрическое распределение (конверсия)
# ============================================
print("="*70)
print("ПРОВЕРКА ГИПОТЕЗ: ГЕОМЕТРИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ")
print("="*70)

from ucimlrepo import fetch_ucirepo
online_retail = fetch_ucirepo(id=352)
df_retail = online_retail.data.features.copy()
df_retail = df_retail.dropna(subset=['CustomerID'])
df_retail['TotalPrice'] = df_retail['Quantity'] * df_retail['UnitPrice']
df_retail = df_retail[df_retail['TotalPrice'] > 0]

THRESHOLD = 8
geometric_data = []
for customer_id, group in df_retail.groupby('CustomerID'):
    purchases = group['TotalPrice'].values
    count = 0
    for price in purchases:
        count += 1
        if price > THRESHOLD:
            geometric_data.append(count)
            break

geometric_data = np.array(geometric_data)
np.random.seed(42)
np.random.shuffle(geometric_data)
geometric_data = geometric_data[:500]
n = len(geometric_data)
print(f"\nДанные: n = {n}, min = {np.min(geometric_data)}, max = {np.max(geometric_data)}")

# --- Простая гипотеза: θ = 0.4 ---
print("-"*70)
print("1. ПРОСТАЯ ГИПОТЕЗА: H0: θ = 0.4 (известен)")
print("-"*70)
theta_0 = 0.4

def geom_cdf(x, theta):
    return 1 - (1 - theta) ** np.floor(x)

data_sorted = np.sort(geometric_data)
ecdf = np.arange(1, n+1) / n
theoretical_cdf = geom_cdf(data_sorted, theta_0)

D_plus = np.max(ecdf - theoretical_cdf)
D_minus = np.max(theoretical_cdf - (np.arange(n) / n))
D_n = max(D_plus, D_minus)
critical_value_ks = 1.36 / np.sqrt(n)

print(f"\nКритерий Колмогорова-Смирнова:")
print(f"  D_n = {D_n:.4f}")
print(f"  Критическое значение: {critical_value_ks:.4f}")
print(f"  Вывод: H0 {'отвергается' if D_n > critical_value_ks else 'принимается'}")

# χ² для геометрического — ИСПРАВЛЕНО
k_chi = min(10, int(np.max(geometric_data)))
bins_edges = [0.5] + [i + 0.5 for i in range(1, k_chi)] + [np.max(geometric_data) + 0.5]
observed, _ = np.histogram(geometric_data, bins=bins_edges)

expected_probs = []
for i in range(1, k_chi):
    prob = theta_0 * (1 - theta_0)**(i - 1)
    expected_probs.append(prob)
prob_last = (1 - theta_0)**(k_chi - 1)
expected_probs.append(prob_last)

expected = n * np.array(expected_probs)

observed_list = list(observed)
expected_list = list(expected)
while len(expected_list) > 1 and expected_list[-1] < 5:
    expected_list[-2] += expected_list[-1]
    observed_list[-2] += observed_list[-1]
    expected_list.pop()
    observed_list.pop()

observed_final = np.array(observed_list)
expected_final = np.array(expected_list)

chi2_stat = np.sum((observed_final - expected_final)**2 / expected_final)
df_chi = len(expected_final) - 1
chi2_critical = chi2.ppf(0.95, df_chi)

print(f"\nКритерий хи-квадрат:")
print(f"  χ² = {chi2_stat:.4f}")
print(f"  Степеней свободы: {df_chi}")
print(f"  Критическое значение: {chi2_critical:.4f}")
print(f"  Вывод: H0 {'отвергается' if chi2_stat > chi2_critical else 'принимается'}")

# --- Сложная гипотеза ---
print("-"*70)
print("2. СЛОЖНАЯ ГИПОТЕЗА: H0: θ неизвестен")
print("-"*70)
theta_hat = 1 / np.mean(geometric_data)
print(f"\nОценка θ̂ = {theta_hat:.4f}")

theoretical_cdf_hat = geom_cdf(data_sorted, theta_hat)
D_plus_hat = np.max(ecdf - theoretical_cdf_hat)
D_minus_hat = np.max(theoretical_cdf_hat - (np.arange(n) / n))
D_n_hat = max(D_plus_hat, D_minus_hat)
print(f"\nКолмогоров-Смирнов:")
print(f"  D_n = {D_n_hat:.4f}")
print(f"  Критическое значение: {critical_value_ks:.4f}")
print(f"  Вывод: H0 {'отвергается' if D_n_hat > critical_value_ks else 'принимается'}")

expected_probs_hat = []
for i in range(1, k_chi):
    prob = theta_hat * (1 - theta_hat)**(i - 1)
    expected_probs_hat.append(prob)
prob_last_hat = (1 - theta_hat)**(k_chi - 1)
expected_probs_hat.append(prob_last_hat)
expected_hat = n * np.array(expected_probs_hat)

observed_list_hat = list(observed)
expected_list_hat = list(expected_hat)
while len(expected_list_hat) > 1 and expected_list_hat[-1] < 5:
    expected_list_hat[-2] += expected_list_hat[-1]
    observed_list_hat[-2] += observed_list_hat[-1]
    expected_list_hat.pop()
    observed_list_hat.pop()

observed_final_hat = np.array(observed_list_hat)
expected_final_hat = np.array(expected_list_hat)

chi2_stat_hat = np.sum((observed_final_hat - expected_final_hat)**2 / expected_final_hat)
df_chi_hat = len(expected_final_hat) - 1 - 1
chi2_critical_hat = chi2.ppf(0.95, df_chi_hat)

print(f"\nХи-квадрат:")
print(f"  χ² = {chi2_stat_hat:.4f}")
print(f"  Степеней свободы: {df_chi_hat}")
print(f"  Критическое значение: {chi2_critical_hat:.4f}")
print(f"  Вывод: H0 {'отвергается' if chi2_stat_hat > chi2_critical_hat else 'принимается'}")


# ============================================
# Часть 2: Нормальное распределение II (температура, Houston)
# ============================================

print("\n"+"="*70)
print("ПРОВЕРКА ГИПОТЕЗ: НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ II")
print("="*70)

url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/us-weather-history/KHOU.csv"
df_temp = pd.read_csv(url)
df_temp['date'] = pd.to_datetime(df_temp['date'])
df_temp['month'] = df_temp['date'].dt.month
df_temp['temp_c'] = (df_temp['actual_mean_temp'] - 32) * 5/9
spring_data = df_temp[df_temp['month'].isin([3, 4, 5])]['temp_c'].values
n_norm = len(spring_data)
print(f"\nТемпературные данные: n = {n_norm}, mean = {np.mean(spring_data):.2f}°C")

# --- Простая гипотеза ---
print("-"*70)
print("1. ПРОСТАЯ ГИПОТЕЗА: H0: μ=22.5, θ=4.0")
print("-"*70)
mu_0 = 22.5
theta_0_norm = 4.0

ks_stat, ks_pvalue = kstest(spring_data, lambda x: norm.cdf(x, loc=mu_0, scale=theta_0_norm))
critical_value_ks_norm = 1.36 / np.sqrt(n_norm)
print(f"\nКолмогоров-Смирнов:")
print(f"  D_n = {ks_stat:.4f}")
print(f"  p-value = {ks_pvalue:.4f}")
print(f"  Критическое значение: {critical_value_ks_norm:.4f}")
print(f"  Вывод: H0 {'отвергается' if ks_pvalue < 0.05 else 'принимается'}")

k_norm = int(np.ceil(1 + np.log2(n_norm)))
observed_norm, bin_edges = np.histogram(spring_data, bins=k_norm)
expected_probs_norm = [norm.cdf(bin_edges[i+1], mu_0, theta_0_norm) - norm.cdf(bin_edges[i], mu_0, theta_0_norm) for i in range(len(bin_edges)-1)]
expected_norm = n_norm * np.array(expected_probs_norm)

observed_list_norm = list(observed_norm)
expected_list_norm = list(expected_norm)
while len(expected_list_norm) > 1 and expected_list_norm[-1] < 5:
    expected_list_norm[-2] += expected_list_norm[-1]
    observed_list_norm[-2] += observed_list_norm[-1]
    expected_list_norm.pop()
    observed_list_norm.pop()
while len(expected_list_norm) > 1 and expected_list_norm[0] < 5:
    expected_list_norm[1] += expected_list_norm[0]
    observed_list_norm[1] += observed_list_norm[0]
    expected_list_norm.pop(0)
    observed_list_norm.pop(0)

observed_final_norm = np.array(observed_list_norm)
expected_final_norm = np.array(expected_list_norm)

chi2_stat_norm = np.sum((observed_final_norm - expected_final_norm)**2 / expected_final_norm)
df_chi_norm = len(expected_final_norm) - 1
chi2_critical_norm = chi2.ppf(0.95, df_chi_norm)
print(f"\nХи-квадрат:")
print(f"  χ² = {chi2_stat_norm:.4f}")
print(f"  Степеней свободы: {df_chi_norm}")
print(f"  Критическое значение: {chi2_critical_norm:.4f}")
print(f"  Вывод: H0 {'отвергается' if chi2_stat_norm > chi2_critical_norm else 'принимается'}")

# --- Сложная гипотеза ---
print("-"*70)
print("2. СЛОЖНАЯ ГИПОТЕЗА: H0: μ, θ неизвестны")
print("-"*70)
mu_hat = np.mean(spring_data)
theta_hat_norm = np.std(spring_data, ddof=0)
print(f"\nОценки параметров: μ̂ = {mu_hat:.4f}, θ̂ = {theta_hat_norm:.4f}")

ks_stat_hat, ks_pvalue_hat = kstest(spring_data, lambda x: norm.cdf(x, loc=mu_hat, scale=theta_hat_norm))
print(f"\nКолмогоров-Смирнов:")
print(f"  D_n = {ks_stat_hat:.4f}")
print(f"  p-value = {ks_pvalue_hat:.4f}")
print(f"  Критическое значение: {critical_value_ks_norm:.4f}")
print(f"  Вывод: H0 {'отвергается' if ks_pvalue_hat < 0.05 else 'принимается'}")

expected_probs_norm_hat = [norm.cdf(bin_edges[i+1], mu_hat, theta_hat_norm) - norm.cdf(bin_edges[i], mu_hat, theta_hat_norm) for i in range(len(bin_edges)-1)]
expected_norm_hat = n_norm * np.array(expected_probs_norm_hat)

observed_list_norm_hat = list(observed_norm)
expected_list_norm_hat = list(expected_norm_hat)
while len(expected_list_norm_hat) > 1 and expected_list_norm_hat[-1] < 5:
    expected_list_norm_hat[-2] += expected_list_norm_hat[-1]
    observed_list_norm_hat[-2] += observed_list_norm_hat[-1]
    expected_list_norm_hat.pop()
    observed_list_norm_hat.pop()
while len(expected_list_norm_hat) > 1 and expected_list_norm_hat[0] < 5:
    expected_list_norm_hat[1] += expected_list_norm_hat[0]
    observed_list_norm_hat[1] += observed_list_norm_hat[0]
    expected_list_norm_hat.pop(0)
    observed_list_norm_hat.pop(0)

observed_final_norm_hat = np.array(observed_list_norm_hat)
expected_final_norm_hat = np.array(expected_list_norm_hat)

chi2_stat_norm_hat = np.sum((observed_final_norm_hat - expected_final_norm_hat)**2 / expected_final_norm_hat)
df_chi_norm_hat = len(expected_final_norm_hat) - 1 - 2
if df_chi_norm_hat < 1:
    df_chi_norm_hat = 1
chi2_critical_norm_hat = chi2.ppf(0.95, df_chi_norm_hat)
print(f"\nХи-квадрат:")
print(f"  χ² = {chi2_stat_norm_hat:.4f}")
print(f"  Степеней свободы: {df_chi_norm_hat}")
print(f"  Критическое значение: {chi2_critical_norm_hat:.4f}")
print(f"  Вывод: H0 {'отвергается' if chi2_stat_norm_hat > chi2_critical_norm_hat else 'принимается'}")

print("\n"+"="*70)
print("ПРОВЕРКА ГИПОТЕЗ ЗАВЕРШЕНА")
print("="*70)
