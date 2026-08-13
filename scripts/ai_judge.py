from google import genai
import requests
import pandas as pd
import random
import time
import re
import os
from dotenv import load_dotenv


# 1. NASTAVENIE GEMINI API
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) # <--- Toto je správny spôsob inicializácie
#model = client.models.generate_content(model='gemini-2.5-flash')

# 2. NAČÍTANIE DÁT PRE TESTOVANIE
print("Načítavam metadáta o knihách...")
knihy_df = pd.read_csv('../data/book-recommendation-dataset/Books_Cleaned.csv', dtype=str).fillna('')
meta_dict = knihy_df.drop_duplicates('ISBN').set_index('ISBN')[['Book-Title', 'Book-Author']].to_dict('index')

print("Načítavam hodnotenia, aby sme zistili, ktorý model bol použitý...")
ratings_df = pd.read_csv('../data/book-recommendation-dataset/Ratings.csv', dtype=str)
rating_counts = ratings_df['ISBN'].value_counts().to_dict()

# 3. ROZDELENIE KNÍH DO 2 KATEGÓRIÍ (CF vs TF-IDF)
isbn_cf = []
isbn_tfidf = []
vsetky_isbn = knihy_df['ISBN'].unique().tolist()

for isbn in vsetky_isbn:
    pocet_hodnoteni = rating_counts.get(isbn, 0)
    if pocet_hodnoteni > 10:
        isbn_cf.append(isbn)
    else:
        isbn_tfidf.append(isbn)

VZORKA_CF = random.sample(isbn_cf, min(10, len(isbn_cf)))
VZORKA_TFIDF = random.sample(isbn_tfidf, min(10, len(isbn_tfidf)))

testovacie_knihy = [(isbn, "CF (>10 hodnotení)") for isbn in VZORKA_CF] + \
                   [(isbn, "TF-IDF (<=10 hodnotení)") for isbn in VZORKA_TFIDF]
random.shuffle(testovacie_knihy)

# 4. PRÍPRAVA NA ŠTATISTIKY (Teraz zaznamenávame aj jednotlivé pozície 1-5)
skore_statistika = {
    "CF (>10 hodnotení)": {
        "celkove_sucet": 0, "celkove_pocet": 0, 
        "item_sucet": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "item_pocet": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    },
    "TF-IDF (<=10 hodnotení)": {
        "celkove_sucet": 0, "celkove_pocet": 0, 
        "item_sucet": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "item_pocet": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    }
}

print(f"Začínam hodnotiť {len(testovacie_knihy)} kníh pomocou Gemini...\n")
print("-" * 50)


for isbn, typ_modelu in testovacie_knihy:
    zdrojova_kniha = meta_dict.get(isbn)
    if not zdrojova_kniha:
        continue
    
    nazov_zdroj = zdrojova_kniha['Book-Title']
    autor_zdroj = zdrojova_kniha['Book-Author']
    
    
    try:
        odpoved = requests.get(f"http://localhost:8000/recommendations/{isbn}")
        print(odpoved)
        data = odpoved.json()
    except Exception as e:
        print(f"Chyba pripojenia na lokálne API: {e}")
        break

    odporucania = data.get("recommendations", [])
    if not odporucania:
        continue

    zoznam_knih_text = ""
    pocet_odporucani = len(odporucania)
    for i, rec in enumerate(odporucania, 1):
        zoznam_knih_text += f"{i}. {rec['title']} od {rec['author']}\n"

    # C. NOVÝ PROMPT PRE DETAILNÉ HODNOTENIE
    prompt = f"""
    Si expert na literatúru a knižný vkus.
    Čitateľ práve dočítal túto knihu: "{nazov_zdroj}" od autora "{autor_zdroj}".
    
    Náš systém mu na základe toho navrhol týchto {pocet_odporucani} podobných kníh:
    {zoznam_knih_text}
    
    Ohodnoť kvalitu na škále od 1 do 5 (1 = nezmysel, 5 = perfektné).
    Potrebujem celkové skóre za zoznam ako celok a následne skóre pre každú jednu knihu v zozname (ako veľmi sa hodí k tej pôvodnej).
    
    Odpovedz PRESNE v tomto formáte (dodrž názvy):
    CELKOVE_SKORE: [1-5]
    SKORE_1: [1-5]
    SKORE_2: [1-5]
    SKORE_3: [1-5]
    SKORE_4: [1-5]
    SKORE_5: [1-5]
    DOVOD: [Stručné zhodnotenie prečo si dal takéto známky a ak nejaká kniha úplne uletela, spomeň prečo]
    """

    try:
        response = client.models.generate_content(model = 'gemini-3.6-flash', contents={'text': prompt},
    config={
        'temperature': 0,
        'top_p': 0.95,
        'top_k': 20,
    },)
        odpoved_text = response.text
        
        # D. ROZPARSOVANIE VÝSLEDKOV (Regex)
        celkove_match = re.search(r'CELKOVE_SKORE:\s*(\d+)', odpoved_text, re.IGNORECASE)
        celkove_skore = int(celkove_match.group(1)) if celkove_match else 0
        
        item_scores = {}
        for idx in range(1, pocet_odporucani + 1):
            match = re.search(fr'SKORE_{idx}:\s*(\d+)', odpoved_text, re.IGNORECASE)
            if match:
                item_scores[idx] = int(match.group(1))
        
        print(f"📖 ZDROJ: {nazov_zdroj} ({autor_zdroj}) | 🤖 MODEL: {typ_modelu}")
        print(odpoved_text.strip())
        print("-" * 50)
        
        # E. ULOŽENIE DO ŠTATISTÍK
        if celkove_skore > 0:
            skore_statistika[typ_modelu]["celkove_sucet"] += celkove_skore
            skore_statistika[typ_modelu]["celkove_pocet"] += 1
            
            for idx, sc in item_scores.items():
                if sc > 0:
                    skore_statistika[typ_modelu]["item_sucet"][idx] += sc
                    skore_statistika[typ_modelu]["item_pocet"][idx] += 1
            
    except Exception as e:
        print(f"Chyba pri volaní Gemini API: {e}")
    
    time.sleep(4)

# F. ZÁVEREČNÝ REPORT S VÝPISOM PODĽA POZÍCIE
print(f"\n✅ HODNOTENIE DOKONČENÉ! POROVNANIE MODELOV A POZÍCIÍ:")
print("=" * 60)
for model_name, stats in skore_statistika.items():
    if stats["celkove_pocet"] > 0:
        celkovy_priemer = stats["celkove_sucet"] / stats["celkove_pocet"]
        print(f"🏆 {model_name}")
        print(f"   ▶ CELKOVÉ SKÓRE ZBIEKRY: {celkovy_priemer:.2f} / 5.00 (z {stats['celkove_pocet']} dopytov)")
        
        print("   ▶ SKÓRE PODĽA POZÍCIE V API (Rank):")
        for idx in range(1, 6):
            if stats["item_pocet"][idx] > 0:
                p = stats["item_sucet"][idx] / stats["item_pocet"][idx]
                print(f"       Kniha #{idx}: {p:.2f} / 5.00")
    else:
        print(f"⚠️ {model_name}: Nemáme žiadne úspešné hodnotenia.")
    print("-" * 60)