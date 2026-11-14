import matplotlib.pyplot as plt
import pandas as pd

# Загружаем датасет
df = pd.read_csv('potato_fields.csv', sep=None, engine='python')

# Считаем теоретический периметр прямоугольника
short_side_m = (df['area_ha'] * 10000) / df['long_side_m']
rect_perim = 2 * (df['long_side_m'] + short_side_m)
diff_geom = df['edge_m'] - rect_perim
#diff geom - это разница между (данные по периметру измеренные геодезистом) - (высчитанные данные через указанные фермерами)
# Строим только График Г (крупно и с сеткой)
plt.figure(figsize=(14, 6))
plt.scatter(df['ditch_m'], diff_geom, s=15, alpha=0.8, color='black')
plt.title('(edge_m - rect_perim) против ditch_m')
plt.xlabel('ditch_m')
plt.ylabel('edge_m - rect_perim')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()