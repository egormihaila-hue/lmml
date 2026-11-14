from difflib import get_close_matches
import re
import unicodedata
import pandas as pd

df = pd.read_csv('routes.csv')


def normalize_town(s):
  s = str(s).strip()
  s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')   #переводит строку в чистый английский алфавит (ASCII), а всё, что в него не входит (те самые точки и галочки над буквами), просто выбрасывает ('ignore').
  s = re.sub(r'^[\W_]+|[\W_]+$', '', s) #удаляет любые небуквенные знаки (точки, дефисы, кавычки, знаки подчёркивания) в самом начале (^) и в самом конце ($) строки.
  s = re.sub(r'\s+', ' ', s) #заменяет любые двойные или тройные пробелы внутри названия на один обычный пробел.
  return s.title() #делает первую букву каждого слова заглавной, а остальные строчными


df['from_town'] = df['from_town'].apply(normalize_town)
df['to_town'] = df['to_town'].apply(normalize_town)

towns = sorted(set(df['from_town']).union(set(df['to_town'])))
#собирает уникальные города из столбца from_town (set(...)), объединяет их (.union(...)) с уникальными городами из to_town и сортирует общий список по алфавиту (sorted(...)).

print(f'Unique towns: {len(towns)} (target: 116)\n')


print('\n=== ALL 177 TOWNS ===')
for i, t in enumerate(towns, 1):
  cnt = ((df['from_town'] == t) | (df['to_town'] == t)).sum()
  print(f'{i:3d}. {t} ({cnt})')
