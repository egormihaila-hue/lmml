from difflib import get_close_matches
import re
import unicodedata
import pandas as pd

df = pd.read_csv('routes.csv')


def normalize_town(s):
  s = str(s).strip()
  s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
  s = re.sub(r'^[\W_]+|[\W_]+$', '', s)
  s = re.sub(r'\s+', ' ', s)
  return s.title()


df['from_town'] = df['from_town'].apply(normalize_town)
df['to_town'] = df['to_town'].apply(normalize_town)

# 1. Склеиваем опечатки через difflib
towns = sorted(set(df['from_town']).union(set(df['to_town'])))
counts = pd.concat([df['from_town'], df['to_town']]).value_counts().to_dict()
towns_by_freq = sorted(towns, key=lambda x: (-counts[x], x))

groups = []
visited = set()

for t in towns_by_freq:
  if t in visited:
    continue
  similar = get_close_matches(t, towns_by_freq, n=10, cutoff=0.78)
  group = sorted(
      [s for s in similar if s not in visited], key=lambda x: (-counts[x], x)
  )
  visited.update(group)
  groups.append(group)

typo_map = {}
for g in groups:
  for variant in g[1:]:
    typo_map[variant] = g[0]  # Важно: берём именно первый (главный) элемент g[0]

df['from_town'] = df['from_town'].replace(typo_map)
df['to_town'] = df['to_town'].replace(typo_map)

# 2. Словарь шведских -> финских названий
swedish_to_finnish = {
    'Abo': 'Turku',
    'Alavo': 'Alavus',
    'Bjorneborg': 'Pori',
    'Borga': 'Porvoo',
    'Brahestad': 'Raahe',
    'Esbo': 'Espoo',
    'Fredrikshamn': 'Hamina',
    'Grankulla': 'Kauniainen',
    'Helsingfors': 'Helsinki',
    'Hogfors': 'Karkkila',
    'Hyvinge': 'Hyvinkaa',
    'Idensalmi': 'Iisalmi',
    'Ikalis': 'Ikaalinen',
    'Ilmola': 'Ilmajoki',
    'Inga': 'Inkoo',
    'Jakobstad': 'Pietarsaari',
    'Kajana': 'Kajaani',
    'Karleby': 'Kokkola',
    'Kasko': 'Kaskinen',
    'Kervo': 'Kerava',
    'Keuru': 'Keuruu',
    'Kimitoon': 'Kemionsaari',
    'Korsholm': 'Mustasaari',
    'Kristinestad': 'Kristiinankaupunki',
    'Kronoby': 'Kruunupyy',
    'Kyrkslatt': 'Kirkkonummi',
    'Lahtis': 'Lahti',
    'Lappo': 'Lapua',
    'Larsmo': 'Luoto',
    'Lojo': 'Lohja',
    'Lundo': 'Lieto',
    'Malax': 'Maalahti',
    'Nadendal': 'Naantali',
    'Narpes': 'Narpio',
    'Nykarleby': 'Uusikaarlepyy',
    'Nystad': 'Uusikaupunki',
    'Pargas': 'Parainen',
    'Raseborg': 'Raasepori',
    'Reso': 'Raisio',
    'S:T Karins': 'Kaarina',
    'S:T Michel': 'Mikkeli',
    'Sibbo': 'Sipoo',
    'Sjundea': 'Siuntio',
    'Tammerfors': 'Tampere',
    'Tavastehus': 'Hameenlinna',
    'Tornea': 'Tornio',
    'Traskanda': 'Jarvenpaa',
    'Tusby': 'Tuusula',
    'Uleaborg': 'Oulu',
    'Ulvsby': 'Ulvila',
    'Vanda': 'Vantaa',
    'Villmanstrand': 'Lappeenranta',
    'Virmo': 'Mynamaki',
    'Vora': 'Voyri',
}

df['from_town'] = df['from_town'].replace(swedish_to_finnish)
df['to_town'] = df['to_town'].replace(swedish_to_finnish)

# 3. Конечный список уникальных городов
final_towns = sorted(set(df['from_town']).union(set(df['to_town'])))
print(f'Final towns count: {len(final_towns)} (target: 116)\n')
for i, town in enumerate(final_towns, 1):
  print(f'{i:3d}. {town}')

# 4. Строим граф уникальных связей (без петель и без учёта направления A-B / B-A)
valid = df[df['from_town'] != df['to_town']]
edges = set(
    tuple(sorted([a, b])) for a, b in zip(valid['from_town'], valid['to_town'])
)
print(f'\nUnique connections count: {len(edges)} (target: 448)')

# 5. Считаем количество уникальных связей для каждого города и формируем флаг
degree = {t: 0 for t in final_towns}
for a, b in edges:
  degree[a] += 1
  degree[b] += 1

top_8 = sorted(degree.items(), key=lambda x: (-x[1], x[0]))[:8]
print('\n=== TOP 8 BUSIEST HUBS ===')
for rank, (town, deg) in enumerate(top_8, 1):
  print(f'{rank}. {town}: {deg} connections')

flag_word = ''.join(town[0].upper() for town, _ in top_8[:7])
print(f'\nFLAG: SIGMOID_{flag_word}')
