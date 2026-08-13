import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. NAČÍTANIE DÁT
# ==========================================
print("1. Načítavam vyčistené knihy a ratingy...")
books_df = pd.read_csv('../data/book-recommendation-dataset/Books_Cleaned.csv', dtype=str).fillna('')
ratings_df = pd.read_csv('../data/book-recommendation-dataset/Ratings.csv', dtype={'ISBN': str})

# Postaráme sa o to, aby Work-ID_Group skutočne existovalo (krok z minula)
books_df['Work-ID_Group'] = books_df['Work-ID']
books_df.loc[books_df['Work-ID_Group'] == '', 'Work-ID_Group'] = books_df['ISBN']

# ==========================================
# 2. ZLÚČENIE RATINGOV PODĽA WORK-ID (Kľúčový produkčný krok!)
# ==========================================
print("2. Prepisujem ratingy z ISBN na Work-ID...")
# Vytvoríme rýchly slovník na preklad ISBN -> Work-ID
isbn_to_work = dict(zip(books_df['ISBN'], books_df['Work-ID_Group']))

# Namapujeme Work-ID do ratingov (tie, ktoré nemáme v knihách, vyhodíme)
ratings_df['Work-ID_Group'] = ratings_df['ISBN'].map(isbn_to_work)
ratings_df = ratings_df.dropna(subset=['Work-ID_Group'])

# AGREGÁCIA: Ak jeden user hodnotil viacero vydaní tej istej knihy, vezmeme jeho MAX hodnotenie
ratings_work = ratings_df.groupby(['User-ID', 'Work-ID_Group'])['Book-Rating'].max().reset_index()

# Spočítame, koľko unikátnych ratingov (od rôznych ľudí) má každé Work-ID
work_rating_counts = ratings_work.groupby('Work-ID_Group').size().reset_index(name='Pocet_Ratingov')

# ==========================================
# 3. ZJEDNOTENIE KNÍH (Pre Redis a FastAPI potrebujeme 1 Work-ID = 1 Záznam)
# ==========================================
print("3. Vytváram unikátny zoznam kníh pre Redis...")
# Necháme si len unikátne Work-ID. Ako reprezentatívne ISBN si necháme to prvé.
unique_books = books_df.drop_duplicates(subset=['Work-ID_Group'], keep='first').copy()

# Pripojíme k nim počet ratingov
unique_books = pd.merge(unique_books, work_rating_counts, on='Work-ID_Group', how='left')
unique_books['Pocet_Ratingov'] = unique_books['Pocet_Ratingov'].fillna(0).astype(int)

# ==========================================
# 4. ANALÝZA "PREPOJITEĽNOSTI" TÉM
# ==========================================
print("4. Analyzujem témy (hľadám tie, ktoré sú spoločné pre viac ako 5 kníh)...")
# min_df=6 znamená: Slovo/téma musí byť aspoň v 6 knihách (teda 1 kniha + >5 ďalších)
vectorizer = CountVectorizer(min_df=6)

# Skúsime vektorizovať. Výsledkom je matica. 
# Ak má kniha v tejto matici samé nuly, znamená to, že nemá ŽIADNU prepojiteľnú tému.
X = vectorizer.fit_transform(unique_books['Content_Clean'])

# Sčítame hodnoty v každom riadku. Ak je súčet > 0, kniha má aspoň jednu použiteľnú tému.
unique_books['Ma_Prepojitelne_Temy'] = (X.sum(axis=1) > 0).A1  # .A1 to prevedie z matice späť na 1D pole (True/False)

# ==========================================
# 5. ROZDELENIE DO 3 DATASETOV
# ==========================================
print("5. Rozdeľujem dataset na 3 časti podľa tvojej logiky...")

# DATASET 1: Rating Model (>10 ratingov)
dataset_1 = unique_books[unique_books['Pocet_Ratingov'] > 10]

# DATASET 2: Content Model (<=10 ratingov, ALE má aspoň 1 prepojiteľnú tému)
dataset_2 = unique_books[(unique_books['Pocet_Ratingov'] <= 10) & (unique_books['Ma_Prepojitelne_Temy'] == True)]

# DATASET 3: ODPAD (<=10 ratingov a ŽIADNA prepojiteľná téma)
dataset_3 = unique_books[(unique_books['Pocet_Ratingov'] <= 10) & (unique_books['Ma_Prepojitelne_Temy'] == False)]

# ==========================================
# 6. MODEL 1: RATING-BASED (Collaborative Filtering)
# ==========================================
print("\n" + "="*50)
print("⚙️ TRÉNUJEM RATING MODEL (Pre Dataset 1)")
print("="*50)

# Vyfiltrujeme z našich agregovaných ratingov (ratings_work) len tie, 
# ktoré patria do "Zlatého fondu" (dataset_1)
validne_work_ids = set(dataset_1['Work-ID_Group'])
ratings_model_1 = ratings_work[ratings_work['Work-ID_Group'].isin(validne_work_ids)].copy()

print("Vytváram pamäťovo úspornú Pivot Maticu...")
# Prevedieme IDčka na vnútorné číselné kategórie (0, 1, 2...)
user_kategorie = ratings_model_1['User-ID'].astype('category')
book_kategorie = ratings_model_1['Work-ID_Group'].astype('category')

# Kľúč na preklad "z vnútorných čísiel späť na reálne Work-ID"
index_to_work_id = dict(enumerate(book_kategorie.cat.categories))

# Sparse matica (Riadky = Knihy, Stĺpce = Používatelia)
pivot_sparse = csr_matrix((
    ratings_model_1['Book-Rating'], 
    (book_kategorie.cat.codes, user_kategorie.cat.codes)
))

print("Počítam Kosínovú vzdialenosť (Item-Item CF)...")
sim_matrix = cosine_similarity(pivot_sparse)

print("Generujem precomputed tabuľku s TOP 5 odporúčaniami...")
odporucania_rating = []
pocet_knih = sim_matrix.shape[0]

for i in tqdm(range(pocet_knih), desc="Spracovávam Zlatý fond"):
    aktualne_work_id = index_to_work_id[i]
    skore = sim_matrix[i]
    
    # Hľadáme indexy 6 najpodobnejších kníh (lebo 1. miesto je kniha sama so sebou)
    top_6_indexy = np.argsort(skore)[-6:][::-1]
    
    # Získame ich skutočné Work-ID, ignorujeme aktuálnu knihu
    top_5_work_ids = [index_to_work_id[idx] for idx in top_6_indexy if idx != i][:5]
    
    # Ak má menej ako 5 prepojení, doplníme prázdne znaky
    while len(top_5_work_ids) < 5:
        top_5_work_ids.append('')
        
    odporucania_rating.append({
        'Work-ID_Group': aktualne_work_id,
        'Rec_1': top_5_work_ids[0],
        'Rec_2': top_5_work_ids[1],
        'Rec_3': top_5_work_ids[2],
        'Rec_4': top_5_work_ids[3],
        'Rec_5': top_5_work_ids[4]
    })

# Uložíme finálnu tabuľku do CSV
odporucania_rating_df = pd.DataFrame(odporucania_rating)
# vystup_rating_redis = '../data/book-recommendation-dataset/Redis_Precomputed_Ratings.csv'
# odporucania_rating_df.to_csv(vystup_rating_redis, index=False)

print(odporucania_rating_df.head())

# ==========================================
# 8. MODEL 2: CONTENT-BASED (TF-IDF na témach)
# ==========================================
print("\n" + "="*50)
print("🧠 TRÉNUJEM CONTENT MODEL (Pre Dataset 2)")
print("="*50)

# Uistíme sa, že indexy sú pekne zarovnané od 0, aby sa nám to ľahko mapovalo
dataset_2 = dataset_2.reset_index(drop=True)

print("1. Vytváram TF-IDF maticu...")
# max_features zabezpečí, že ak by tam bolo extrémne veľa slov, zoberie len top 10 000
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(dataset_2['Content_Clean'])

print(f"Veľkosť TF-IDF matice: {tfidf_matrix.shape} (Knihy x Unikátne slová)")

print("2. Hľadám TOP 5 podobných kníh (Počítam Kosínus za behu, kvôli ochrane RAM)...")
odporucania_content = []
pocet_knih_content = tfidf_matrix.shape[0]

# Slovník na rýchly preklad z indexu matice na skutočné Work-ID
index_to_work_id_content = dataset_2['Work-ID_Group'].to_dict()

# Ideme cez každú knihu
for i in tqdm(range(pocet_knih_content), desc="Spracovávam Content model"):
    aktualne_work_id = index_to_work_id_content[i]
    
    # KÚZLO PRE RAM: Vypočítame podobnosť len JEDNEJ knihy voči VŠETKÝM
    # Výsledkom nie je obrovská matica, ale len jedno 1D pole skóre pre danú knihu
    skore = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
    
    # Nájdeme indexy 6 najlepších kníh (1. miesto je kniha sama so sebou)
    top_6_indexy = np.argsort(skore)[-6:][::-1]
    
    # Preložíme ich na reálne Work-ID (a vynecháme knihu samu seba)
    top_5_work_ids = [index_to_work_id_content[idx] for idx in top_6_indexy if idx != i][:5]
    
    # Poistka, ak by kniha nemala dosť prienikov
    while len(top_5_work_ids) < 5:
        top_5_work_ids.append('')
        
    odporucania_content.append({
        'Work-ID_Group': aktualne_work_id,
        'Rec_1': top_5_work_ids[0],
        'Rec_2': top_5_work_ids[1],
        'Rec_3': top_5_work_ids[2],
        'Rec_4': top_5_work_ids[3],
        'Rec_5': top_5_work_ids[4]
    })

odporucania_content_df = pd.DataFrame(odporucania_content)

# ==========================================
# 10. FINÁLNA PRÍPRAVA PRE REDIS (Mapovanie na všetky ISBN)
# ==========================================
print("\n" + "="*50)
print("🚀 GENERUJEM FINÁLNY SÚBOR PRE REDIS (Všetky ISBN)")
print("="*50)

# 1. Spojíme odporúčania z oboch modelov do jednej tabuľky
vsetky_odporucania_workid = pd.concat([odporucania_rating_df, odporucania_content_df], ignore_index=True)

print(f"Spolu máme vypočítané odporúčania pre {len(vsetky_odporucania_workid):,} unikátnych diel (Work-IDs).")

# 2. Zoberieme pôvodný dataset VŠETKÝCH kníh (kde má každé vydanie svoje ISBN)
# books_df sme si načítali úplne hore v kroku 1.
isbn_mapping = books_df[['ISBN', 'Work-ID_Group']].copy()

# 3. Kúzlo LEFT JOINU: Prepojíme odporúčania ku každému ISBN
# Knihy z Datasetu 3 (Trash) tu prirodzene dostanú prázdne hodnoty (NaN), 
# pretože sa ich Work-ID nenájde v tabuľke 'vsetky_odporucania_workid'.
final_df = pd.merge(isbn_mapping, vsetky_odporucania_workid, on='Work-ID_Group', how='left')

# 4. Upratanie dát pre Redis
# Všetky NaN (či už Dataset 3, alebo ak mala kniha len 2 odporúčania) nahradíme prázdnym znakom ""
stlpce_na_vyplnenie = ['Rec_1', 'Rec_2', 'Rec_3', 'Rec_4', 'Rec_5']
final_df[stlpce_na_vyplnenie] = final_df[stlpce_na_vyplnenie].fillna('')

# Work-ID_Group už vo finálnom Redis slovníku nepotrebujeme (Kľúčom bude ISBN)
final_df = final_df.drop(columns=['Work-ID_Group'])

# ==========================================
# 11. ULOŽENIE FINÁLNEHO SÚBORU
# ==========================================
vystupny_redis_subor = '../data/book-recommendation-dataset/ISBN_Recommendations.csv'
final_df.to_csv(vystupny_redis_subor, index=False)

print(f"✅ Hotovo! Finálny súbor vytvorený. Obsahuje presne {len(final_df):,} záznamov (ISBN).")
print(f"Uložené v: {vystupny_redis_subor}")

# Výpis pre kontrolu
print("\n--- UKÁŽKA: ISBN s plnými odporúčaniami (Zlatý fond / Content) ---")
print(final_df[final_df['Rec_1'] != ''].head(3))

print("\n--- UKÁŽKA: ISBN bez odporúčaní (API vie, že existujú, vráti prázdny list) ---")
print(final_df[final_df['Rec_1'] == ''].head(3))