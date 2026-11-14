import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Загружаем датасет с замерами полей
df = pd.read_csv('potato_fields.csv', sep=None, engine='python')

# 2. Вычисляем геометрические размеры поля (считаем, что поле прямоугольное)
# 1 гектар = 10 000 м², поэтому короткая сторона равна (area_ha * 10000) / long_side_m
df['short_side_m'] = (df['area_ha'] * 10000) / df['long_side_m']
df['rect_perim'] = 2 * (df['long_side_m'] + df['short_side_m'])

# 3. Формируем матрицу признаков и обучаем модель линейной регрессии для предсказания edge_m
features = [
    'area_ha',
    'long_side_m',
    'short_side_m',
    'rect_perim',
    'ditch_m',
    'hedge_m',
    'crop_code',
    'farm_type',
]
X = pd.get_dummies(df[features], drop_first=True).fillna(0)
y = df['edge_m']

model = LinearRegression()
model.fit(X, y)

# 4. Находим остаток (ошибку) модели: разница между реальным замером и предсказанием
df['residual'] = y - model.predict(X)

# 5. Строим четвёртый график: остаток модели против длины канавы (ditch_m)
plt.figure(figsize=(14, 5))
plt.scatter(df['ditch_m'], df['residual'], s=15, alpha=0.8, color='#1f77b4')
plt.title(
    'Остаток модели предсказания периметра (residual) против длины канавы'
    ' (ditch_m)'
)
plt.xlabel('Длина канавы вдоль поля, ditch_m (м)')
plt.ylabel('Остаток модели: edge_m - predicted_edge_m (м)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()