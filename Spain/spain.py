import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#считываю таблицы
guide = pd.read_csv("office_price_guide.csv")
purch = pd.read_csv("purchases.csv")

#очищение строк от лишних пробелов, если они есть
for d in (guide, purch):
    for c in d.select_dtypes("object"):
        d[c] = d[c].str.strip()

# 2. Объединение таблиц если код в столбце Product совпадает
df = purch.merge(guide, on="Product", how="left")
df["unit_price"] = df["Total_paid_eur"] / df["Quantity"]  #читаю длину за единицу товара: общая сумма/кол-во товара

#должны соблюдаться два условия
# minPrice <= price <= maxPrice    && city == HomeCity
valid = df[
    (df["unit_price"] >= df["Fair_min_eur"]) &
    (df["unit_price"] <= df["Fair_max_eur"]) &
    (df["City"] == df["Home_city"])
]

# создаю график x-пилоты     y-номер продукта   там где условия выполняються 1, где нет - 0
board = np.zeros((6, 21), dtype=int)
for _, r in valid.iterrows():
    board[int(r["List_no"]) - 1, int(r["Pilot_no"]) - 1] = 1

#принчу эту таблицу и получаеться красота
fig, ax = plt.subplots(figsize=(12, 3))
ax.imshow(board, cmap="gray_r", aspect="equal")
ax.set_xticks(range(21)); ax.set_xticklabels(range(1, 22))
ax.set_yticks(range(6));  ax.set_yticklabels(range(1, 7))
ax.set_xlabel("Pilot_no"); ax.set_ylabel("List_no")
plt.savefig("board.png", dpi=150, bbox_inches="tight")
plt.show()