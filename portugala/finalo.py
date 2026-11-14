import numpy as np
import pandas as pd

# 1. Загружаем датасет с экспериментальными участками виноградников
df = pd.read_csv('douro_vineyard_data.csv')

# 2. Выводим общий результат по всей таблице (здесь новый полив treatment впереди)
print('=== Общий процент успеха (до разделения) ===')
print(df.groupby('treatment')['harvest_success'].agg(['count', 'mean']))
print()

# 3. Список числовых столбцов, которые мы проверяем на наличие скрытого порога
num_cols = [
    'soil_depth_cm',
    'sun_hours_july',
    'vine_density_per_ha',
    'annual_rainfall_mm',
    'distance_from_porto_km',
    'elevation_m',
    'soil_ph',
]

print('=== Поиск порога разделения на 2 группы (<= T и > T) ===')

# Перебираем каждый числовой столбец по очереди
for col in num_cols:
  # Генерируем 100 возможных границ разреза от 5% до 95% значений столбца
  thresholds = np.quantile(df[col], np.linspace(0.05, 0.95, 100))
  best_t = None

  # Проверяем каждую уникальную границу t
  for t in np.unique(thresholds):
    # Делим таблицу на 2 группы по порогу t
    g1 = df[df[col] <= t]  # Группа 1: значения меньше или равны t
    g2 = df[df[col] > t]  # Группа 2: значения больше t

    # Пропускаем слишком мелкие группы (меньше 30 участков), чтобы избежать случайностей
    if len(g1) < 30 or len(g2) < 30:
      continue

    # Считаем долю успешных участков (mean от 0 и 1) в первой группе
    r1_c = g1[g1['treatment'] == 'control']['harvest_success'].mean()
    r1_t = g1[g1['treatment'] == 'treatment']['harvest_success'].mean()

    # Считаем долю успешных участков во второй группе
    r2_c = g2[g2['treatment'] == 'control']['harvest_success'].mean()
    r2_t = g2[g2['treatment'] == 'treatment']['harvest_success'].mean()

    # Проверяем парадокс Симпсона: новый полив (treatment) должен быть хуже
    # старого (control) одновременно в ОБЕИХ группах
    if (r1_t < r1_c) and (r2_t < r2_c):
      best_t = (t, r1_c, r1_t, len(g1), r2_c, r2_t, len(g2))
      break

  # Если такой порог найден, выводим результат и готовый флаг
  if best_t:
    t, r1_c, r1_t, n1, r2_c, r2_t, n2 = best_t
    print(f'>>> НАЙДЕНО! Столбец: {col} | Порог: {t:.2f} <<<')
    print(
        f'  Группа 1 (<= {t:.2f}, участков={n1}): control={r1_c:.3f},'
        f' treatment={r1_t:.3f}'
    )
    print(
        f'  Группа 2 (>  {t:.2f}, участков={n2}): control={r2_c:.3f},'
        f' treatment={r2_t:.3f}'
    )
    print(f'  Флаг: SIGMOID_{col.upper()}\n')
