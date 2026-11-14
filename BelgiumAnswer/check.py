import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Загружаем датасет с замерами полей
df = pd.read_csv('potato_fields.csv', sep=None, engine='python')

# 2. Восстанавливаем базовую геометрию поля (из площади и длинной стороны)
df['short_side_m'] = (df['area_ha'] * 10000) / df['long_side_m']
df['rect_perim'] = 2 * (df['long_side_m'] + df['short_side_m'])

# 3. Используем ТОЛЬКО признаки из декларации фермера (без замера Мариетты hedge_m)
farmer_features = [
    'area_ha',
    'long_side_m',
    'short_side_m',
    'rect_perim',
    'ditch_m',
    'lat',
    'lon',
    'crop_code',
    'farm_type',
]
X = pd.get_dummies(df[farmer_features], drop_first=True).fillna(0)
y = df['edge_m']

# 4. Обучаем линейную регрессию и вычисляем остатки (необъяснённую вариацию)
model = LinearRegression()
model.fit(X, y)
df['residual'] = y - model.predict(X)

# 5. Строим и сохраняем итоговый график со словом SPUDNIK
plt.figure(figsize=(14, 5))
plt.scatter(df['ditch_m'], df['residual'], s=18, alpha=0.8, color='#1f77b4')
plt.title(
    'Остаток модели по декларации фермера (edge_m - predicted) против ditch_m'
)
plt.xlabel('Длина канавы вдоль поля, ditch_m (м)')
plt.ylabel('Остаток модели, residual (м)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Сохраняем график в файл для прикрепления на GitHub
plt.savefig('spudnik_flag.png', dpi=300)
plt.show()
