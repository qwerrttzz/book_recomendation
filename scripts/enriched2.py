import pandas as pd
import requests
import time
import os
from tqdm import tqdm

# Cesta k tvojmu novému záložnému súboru pre texty
VYSTUPNY_SUBOR = 'spracovane_texty.csv'

# 1. Načítanie pôvodného datasetu (použijeme ten obohatený o Work-ID)
books_df = pd.read_csv(
    '../data/book-recommendation-dataset/Books_Enriched.csv', 
    sep=',', 
    encoding='latin-1', 
    on_bad_lines='skip', 
    low_memory=False
)

vsetky_isbn = books_df['ISBN'].tolist()

# 2. CHECKPOINTING: Zistíme, čo už máme stiahnuté, aby sme nezačínali odznova
hotove_isbn = set()
if os.path.exists(VYSTUPNY_SUBOR):
    # Načítame už stiahnuté dáta
    hotove_df = pd.read_csv(VYSTUPNY_SUBOR, dtype=str)
    hotove_isbn = set(hotove_df['ISBN'].tolist())
    print(f"✅ Našiel som zálohu! Už máme spracovaných {len(hotove_isbn)} kníh.")
else:
    # Ak súbor neexistuje, vytvoríme ho s novou hlavičkou
    pd.DataFrame(columns=['ISBN', 'Subjects', 'Description']).to_csv(VYSTUPNY_SUBOR, index=False)
    print("🚀 Vytváram nový záložný súbor pre sťahovanie textov.")

# Vyfiltrujeme len tie ISBN, ktoré ešte nemáme
isbn_na_spracovanie = [isbn for isbn in vsetky_isbn if isbn not in hotove_isbn]

print(f"Zostáva stiahnuť texty pre: {len(isbn_na_spracovanie)} kníh.")


# 3. DÁVKOVÉ SŤAHOVANIE S PRIEBEŽNÝM UKLADANÍM (S FALLBACKOM)
batch_size = 50

for i in tqdm(range(0, len(isbn_na_spracovanie), batch_size), desc="Sťahovanie z API"):
    batch = isbn_na_spracovanie[i : i + batch_size]
    
    # 3.1 Prvý dopyt: jscmd=details
    bibkeys_details = ",".join([f"ISBN:{str(isbn).strip()}" for isbn in batch])
    url_details = f"https://openlibrary.org/api/books?bibkeys={bibkeys_details}&format=json&jscmd=details"
    
    vysledky_davky = {} # Použijeme dictionary pre ľahšie updatovanie
    isbn_s_chybajucimi_datami = []
    
    try:
        response_details = requests.get(url_details)
        
        if response_details.status_code == 200:
            data_details = response_details.json()
            
            for isbn in batch:
                ciste_isbn = str(isbn).strip()
                key = f"ISBN:{ciste_isbn}"
                
                subjects_str = ""
                description_str = ""
                
                if key in data_details:
                    details = data_details[key].get('details', {})
                    
                    # --- 1. TÉMY z details ---
                    subjects_list = details.get('subjects', [])
                    if subjects_list:
                        subjects_str = " | ".join([str(s) for s in subjects_list])
                        
                    # --- 2. POPIS z details ---
                    desc = details.get('description')
                    if isinstance(desc, dict):
                        description_str = desc.get('value', '')
                    elif desc:
                        description_str = str(desc)
                        
                    description_str = description_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
                
                # Uložíme dočasný výsledok
                vysledky_davky[ciste_isbn] = {
                    'ISBN': isbn, 
                    'Subjects': subjects_str, 
                    'Description': description_str
                }
                
                # Ak chýba popis ALEBO témy, zaradíme ISBN na zoznam pre druhý dopyt
                if not subjects_str or not description_str:
                    isbn_s_chybajucimi_datami.append(ciste_isbn)
                    
        else:
            print(f"\n❌ [SERVER ERROR DETAILS] Status {response_details.status_code}")
            for isbn in batch:
                vysledky_davky[str(isbn).strip()] = {'ISBN': isbn, 'Subjects': f"ERROR_{response_details.status_code}", 'Description': ''}

        # ---------------------------------------------------------------------
        # 3.2 Druhý dopyt (FALLBACK): jscmd=data (Len pre tie, kde niečo chýba)
        # ---------------------------------------------------------------------
        if isbn_s_chybajucimi_datami and response_details.status_code == 200:
            bibkeys_data = ",".join([f"ISBN:{isbn}" for isbn in isbn_s_chybajucimi_datami])
            url_data = f"https://openlibrary.org/api/books?bibkeys={bibkeys_data}&format=json&jscmd=data"
            
            response_data = requests.get(url_data)
            
            if response_data.status_code == 200:
                data_data = response_data.json()
                
                for isbn in isbn_s_chybajucimi_datami:
                    key = f"ISBN:{isbn}"
                    if key in data_data:
                        book_data = data_data[key]
                        
                        # Dopĺňame TÉMY (ak chýbali v details)
                        if not vysledky_davky[isbn]['Subjects']:
                            subj_data = book_data.get('subjects', [])
                            if subj_data:
                                # V jscmd=data sú subjects často list of dicts: [{'name': 'Fiction'}]
                                if isinstance(subj_data[0], dict) and 'name' in subj_data[0]:
                                    vysledky_davky[isbn]['Subjects'] = " | ".join([s.get('name', '') for s in subj_data])
                                else:
                                    vysledky_davky[isbn]['Subjects'] = " | ".join([str(s) for s in subj_data])
                        
                        # Dopĺňame POPIS (ak chýbal v details)
                        if not vysledky_davky[isbn]['Description']:
                            desc_data = book_data.get('description')
                            nova_desc = ""
                            if isinstance(desc_data, str):
                                nova_desc = desc_data
                            elif isinstance(desc_data, dict):
                                nova_desc = desc_data.get('value', '')
                            
                            # Ak stále nemáme popis, skúsime "notes" (často tam je skrátený obsah)
                            if not nova_desc and book_data.get('notes'):
                                nova_desc = str(book_data.get('notes'))
                                
                            nova_desc = nova_desc.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
                            vysledky_davky[isbn]['Description'] = nova_desc
            else:
                # Ak zlyhá fallback, nevadí, použijeme to, čo sme získali z details
                pass
                
    except Exception as e:
        print(f"\n🚨 [CRITICAL ERROR] Zlyhalo spojenie pri riadkoch {i} až {i + len(batch)}. Chyba: {e}")
        # V prípade spadnutia networku zabezpečíme, že sa aspoň zapíše chyba a batch sa nestratí
        if not vysledky_davky: 
            for isbn in batch:
                vysledky_davky[str(isbn).strip()] = {'ISBN': isbn, 'Subjects': "ERROR_CONNECTION", 'Description': ''}

    # 4. OKAMŽITÉ ULOŽENIE DO CSV (Append mode)
    if vysledky_davky:
        # Prekonvertujeme dictionary hodnôt späť na list pre Pandas
        temp_df = pd.DataFrame(list(vysledky_davky.values()))
        temp_df.to_csv(VYSTUPNY_SUBOR, mode='a', header=False, index=False)
        
    time.sleep(1) # Zlušnosť voči Open Library API