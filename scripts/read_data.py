import pandas as pd
import requests
import time
import os
from tqdm import tqdm

# Cesta k tvojmu záložnému súboru
VYSTUPNY_SUBOR = 'spracovane_work_ids.csv'

# 1. Načítanie pôvodného datasetu
books_df = pd.read_csv(
    '../data/book-recommendation-dataset/Books.csv', 
    sep=',', 
    encoding='latin-1', 
    on_bad_lines='skip', 
    low_memory=False
)

# Na testovanie zoberieme napr. 10 000 kníh. Keď to zbehne, môžeš dať preč .head()
vsetky_isbn = books_df['ISBN'].tolist()

# 2. CHECKPOINTING: Zistíme, čo už máme stiahnuté, aby sme nezačínali odznova
hotove_isbn = set()
if os.path.exists(VYSTUPNY_SUBOR):
    # Načítame už stiahnuté dáta
    hotove_df = pd.read_csv(VYSTUPNY_SUBOR, dtype=str)
    hotove_isbn = set(hotove_df['ISBN'].tolist())
    print(f"✅ Našiel som zálohu! Už máme spracovaných {len(hotove_isbn)} kníh.")
else:
    # Ak súbor neexistuje, vytvoríme ho s hlavičkou
    pd.DataFrame(columns=['ISBN', 'Work-ID']).to_csv(VYSTUPNY_SUBOR, index=False)
    print("🚀 Vytváram nový záložný súbor pre sťahovanie.")

# Vyfiltrujeme len tie ISBN, ktoré ešte nemáme
isbn_na_spracovanie = [isbn for isbn in vsetky_isbn if isbn not in hotove_isbn]

print(f"Zostáva stiahnuť: {len(isbn_na_spracovanie)} kníh.")


# 3. DÁVKOVÉ SŤAHOVANIE S PRIEBEŽNÝM UKLADANÍM
batch_size = 50  # VIAC NEDÁVAJ, lebo dostaneš chybu 414 (URI Too long)

for i in tqdm(range(0, len(isbn_na_spracovanie), batch_size), desc="Sťahovanie z API"):
    batch = isbn_na_spracovanie[i : i + batch_size]
    
    # Sformátujeme URL a odstránime neviditeľné medzery!
    bibkeys = ",".join([f"ISBN:{str(isbn).strip()}" for isbn in batch])
    url = f"https://openlibrary.org/api/books?bibkeys={bibkeys}&format=json&jscmd=details"
    
    vysledky_davky = []
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            for isbn in batch:
                ciste_isbn = str(isbn).strip()
                key = f"ISBN:{ciste_isbn}"
                work_id = None
                
                if key in data:
                    works_list = data[key].get('details', {}).get('works', [])
                    if works_list:
                        work_id = works_list[0].get('key', '').split('/')[-1]
                
                vysledky_davky.append({'ISBN': isbn, 'Work-ID': work_id})
                
        else:
            # Tu kód vypíše, kde presne zlyhal server (napr. error 500, 502)
            print(f"\n❌ [SERVER ERROR] Status {response.status_code} pri riadkoch {i} až {i + len(batch)}")
            for isbn in batch:
                vysledky_davky.append({'ISBN': isbn, 'Work-ID': f"ERROR_{response.status_code}"})
                
    except Exception as e:
        # Tu to spadne, ak ti vypadne internet alebo vyprší timeout
        print(f"\n🚨 [CRITICAL ERROR] Zlyhalo spojenie pri riadkoch {i} až {i + len(batch)}. Chyba: {e}")
        for isbn in batch:
            vysledky_davky.append({'ISBN': isbn, 'Work-ID': "ERROR_CONNECTION"})

    # 4. OKAMŽITÉ ULOŽENIE DO CSV (Append mode)
    if vysledky_davky:
        temp_df = pd.DataFrame(vysledky_davky)
        # mode='a' znamená append (prilepí na koniec súboru), header=False znamená, že nepridá znova hlavičku
        temp_df.to_csv(VYSTUPNY_SUBOR, mode='a', header=False, index=False)
        
    time.sleep(1)


# 5. ZÁVEREČNÉ SPOJENIE DÁT (Keď sa sťahovanie úspešne dokončí)
print("\nSťahovanie je hotové! Spájam stiahnuté Work IDs s pôvodnou tabuľkou...")

final_work_ids_df = pd.read_csv(VYSTUPNY_SUBOR, dtype=str)

# Využijeme Pandas merge (ako SQL JOIN). Zachová VŠETKY pôvodné stĺpce a pridá Work-ID
konecny_df = pd.merge(books_df, final_work_ids_df, on='ISBN', how='left')

# TOTO JE KĽÚČOVÉ: Uložíme to do nového kompletného CSV súboru!
finálny_subor = '../data/book-recommendation-dataset/Books_Enriched.csv'
konecny_df.to_csv(finálny_subor, index=False, encoding='latin-1')

print(f"✅ Kompletná tabuľka bola úspešne uložená do: {finálny_subor}")

# Vypíšeme prvých 5 riadkov, ale tentokrát úplne VŠETKY stĺpce
print("\nUkážka prvých riadkov tvojej novej, obohatenej tabuľky:")
print(konecny_df.head(5))