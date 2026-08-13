import pandas as pd
import json
from tqdm import tqdm

print("Načítavam CSV...")
df = pd.read_csv('../data/book-recommendation-dataset/ISBN_Recommendations.csv', dtype=str).fillna('')

print("Generujem súbor s Redis príkazmi (seed.txt)...")
with open('seed.txt', 'w', encoding='utf-8') as f:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        isbn = row['ISBN']
        
        # Vytvoríme JSON (napr. ["OL123W", "OL456W"])
        odporucania = [row[f'Rec_{j}'] for j in range(1, 6) if row[f'Rec_{j}'] != '']
        json_str = json.dumps(odporucania)
        
        # Aby to redis-cli pochopil ako jeden text, musíme úvodzovky " escapovať na \"
        json_escaped = json_str.replace('"', '\\"')
        
        # Zapíšeme čistý Redis príkaz do súboru
        f.write(f'SET "{isbn}" "{json_escaped}"\n')

print("✅ Súbor seed.txt bol úspešne vygenerovaný!")