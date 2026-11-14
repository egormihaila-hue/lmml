import numpy as np
import pandas as pd

# Загружаем датасет
df = pd.read_csv('douro_vineyard_data.csv')

num_cols = [
    'soil_depth_cm',
    'sun_hours_july',
    'vine_density_per_ha',
    'annual_rainfall_mm',
    'distance_from_porto_km',
    'elevation_m',
    'soil_ph',
]

print('=== 1. ПОИСК ТОЧНОГО ПОРОГА РАЗРЕЗА (2 ГРУППЫ: <= T и > T) ===')
for col in num_cols:
  thresholds = np.quantile(df[col], np.linspace(0.05, 0.95, 100))
  best_t = None
  for t in np.unique(thresholds):
    g1 = df[df[col] <= t]
    g2 = df[df[col] > t]
    if len(g1) < 30 or len(g2) < 30:
      continue
    r1_c = g1[g1['treatment'] == 'control']['harvest_success'].mean()
    r1_t = g1[g1['treatment'] == 'treatment']['harvest_success'].mean()
    r2_c = g2[g2['treatment'] == 'control']['harvest_success'].mean()
    r2_t = g2[g2['treatment'] == 'treatment']['harvest_success'].mean()

    if (r1_t < r1_c) and (r2_t < r2_c):
      best_t = (t, r1_c, r1_t, len(g1), r2_c, r2_t, len(g2))
      break

  if best_t:
    t, r1_c, r1_t, n1, r2_c, r2_t, n2 = best_t
    print(f'>>> НАЙДЕНО! Столбец: {col} | Порог: {t:.2f} <<<')
    print(
        f'  Группа 1 (<= {t:.2f}, n={n1}): control={r1_c:.3f},'
        f' treatment={r1_t:.3f}'
    )
    print(
        f'  Группа 2 (>  {t:.2f}, n={n2}): control={r2_c:.3f},'
        f' treatment={r2_t:.3f}'
    )
    print(f'  ФЛАГ: SIGMOID_{col.upper()}\n')

print('=== 2. ПРОВЕРКА МЕЛКИХ ИНТЕРВАЛОВ (bins от 6 до 20) ===')
for col in num_cols:
  for bins in range(6, 21):
    for method_name, binner in [
        ('qcut', lambda s: pd.qcut(s, q=bins, duplicates='drop')),
        ('cut', lambda s: pd.cut(s, bins=bins)),
    ]:
      temp = df.copy()
      temp['bin'] = binner(temp[col])
      stats = (
          temp.groupby(['bin', 'treatment'], observed=False)['harvest_success']
          .mean()
          .unstack()
      )
      if (
          not stats.isna().any().any()
          and (stats['treatment'] < stats['control']).all()
      ):
        print(
            f'>>> НАЙДЕНО! Столбец: {col} ({method_name}, bins={bins}) -> ФЛАГ:'
            f' SIGMOID_{col.upper()}'
        )
        print(stats)
        break

print('\n=== 3. ПРОВЕРКА УНИКАЛЬНЫХ ЗНАЧЕНИЙ И СИЛЬНОЙ СВЯЗИ С TREATMENT ===')
for col in num_cols:
  c_mean = df[df['treatment'] == 'control'][col].mean()
  t_mean = df[df['treatment'] == 'treatment'][col].mean()
  print(
      f'{col:23s} | unique={df[col].nunique():4d} | control_mean={c_mean:7.2f}'
      f' | treatment_mean={t_mean:7.2f}'
  )
