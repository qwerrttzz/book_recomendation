import pandas as pd
import json
from tqdm import tqdm

# 1. NAČÍTANIE A ZGRUPOVANIE METADÁT
print("1. Načítavam katalóg kníh pre meta-dáta...")
knihy_df = pd.read_csv('../data/book-recommendation-dataset/Books_Cleaned.csv', dtype=str).fillna('')

# ZGRUPOVANIE: Zabezpečíme, že pre jedno Work-ID zostane len JEDEN záznam (prvý, na ktorý narazí)
# Týmto splníme tvoju podmienku: 1 Work-ID = 1 Názov a 1 Autor
meta_df = knihy_df.drop_duplicates(subset=['Work-ID'])

# Vytvoríme si slovník pre bleskové vyhľadávanie v pamäti RAM
meta_dict = meta_df.set_index('Work-ID')[['Book-Title', 'Book-Author']].to_dict('index')


# 2. NAČÍTANIE ODPORÚČANÍ (podľa tvojej ukážky)
print("2. Načítavam tabuľku odporúčaní...")
# Keďže tvoja ukážka nemá hlavičku, povieme to Pandas-u a pomenujeme si stĺpce sami
col_names = ['ISBN', 'Rec_1', 'Rec_2', 'Rec_3', 'Rec_4', 'Rec_5']
rec_df = pd.read_csv('../data/book-recommendation-dataset/ISBN_Recommendations.csv', names=col_names, dtype=str)

# Nahradíme NaN (napr. prázdne čiarky na konci ukážky) za prázdne stringy
rec_df = rec_df.fillna('')

# 3. GENEROVANIE SEED.TXT PRE REDIS
print("3. Generujem obohatený súbor seed.txt v Redis Mass Insertion formáte...")
with open('seed.txt', 'w', encoding='utf-8') as f:
    
    for _, row in tqdm(rec_df.iterrows(), total=len(rec_df)):
        isbn = row['ISBN']
        odporucania_bohate = []
        
        # Prejdeme stĺpce Rec_1 až Rec_5
        for i in range(1, 6):
            work_id = row[f'Rec_{i}']
            
            # Ak tam Work-ID je (nie je to prázdna čiarka)
            if work_id != '':
                # Vytiahneme názov a autora z nášho slovníka. 
                # Ak z nejakého dôvodu Work-ID v metadátach chýba, dáme "Neznáme"
                detaily = meta_dict.get(work_id, {})
                
                odporucania_bohate.append({
                    "work_id": work_id,
                    "title": detaily.get('Book-Title', 'Neznámy názov'),
                    "author": detaily.get('Book-Author', 'Neznámy autor')
                })
        
        # Preklopíme to do JSON stringu. Zabezpečíme, že sa zachová správne kódovanie znakov (ensure_ascii=False)
        json_str = json.dumps(odporucania_bohate, ensure_ascii=False)
        
        # --- REDIS MASS INSERTION FORMAT ---
        # Týmto povieme Redisu: "Nasledujúci príkaz má 3 argumenty (SET, kľúč, hodnota)"
        f.write("*3\r\n")
        
        # Prvý argument je slovo SET (dĺžka 3)
        f.write("$3\r\nSET\r\n")
        
        # Druhý argument je ISBN a jeho dĺžka v bytoch
        isbn_bytes = isbn.encode('utf-8')
        f.write(f"${len(isbn_bytes)}\r\n{isbn}\r\n")
        
        # Tretí argument je ten dlhý JSON a jeho dĺžka v bytoch
        json_bytes = json_str.encode('utf-8')
        f.write(f"${len(json_bytes)}\r\n{json_str}\r\n")

print("✅ Hotovo! Vygenerovaný nový seed.txt pripravený pre redis-cli --pipe.")