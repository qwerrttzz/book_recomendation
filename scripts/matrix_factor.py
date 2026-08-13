import pandas as pd
import numpy as np


print("1. Načítavam dáta...")
# Načítame hodnotenia a našu obohatenú tabuľku kníh
ratings_df = pd.read_csv('../data/book-recommendation-dataset/Ratings.csv')
books_df = pd.read_csv('../data/book-recommendation-dataset/Books_Enriched.csv', low_memory=False)

print("2. Mapujem Work-ID do tabuľky hodnotení...")
# Z tabuľky kníh si zoberieme LEN to, čo potrebujeme (aby sme nepreťažili pamäť)
mapovacia_tabulka = books_df[['ISBN', 'Work-ID']]

# MERGE: Toto je ten Pandas zázrak. Kód vezme každé ISBN v ratingoch a prilepí k nemu správne Work-ID
ratings_merged = pd.merge(ratings_df, mapovacia_tabulka, on='ISBN', how='left')

# FALLBACK PRE "SIROTY" (Veľmi dôležité!)
# Knihy, ktoré Open Library nepoznalo, budú mať Work-ID = NaN. 
# Aby sme o ne neprišli, skopírujeme im namiesto Work-ID ich pôvodné ISBN.
ratings_merged['Work-ID'] = ratings_merged['Work-ID'].fillna(ratings_merged['ISBN'])

print("3. Riešim duplicitné hodnotenia...")
# Ak niekto ohodnotil 2 rôzne vydania tej istej knihy, vezmeme z nich priemer (mean)
final_ratings = ratings_merged.groupby(['User-ID', 'Work-ID'])['Book-Rating'].mean().reset_index()

print("\n🔥 Hotovo! Takto vyzerá tvoja nová tabuľka pripravená na Matrix Faktorizáciu:")
print(final_ratings.head())


# Predpokladáme, že máme tvoj 'final_ratings' z predchádzajúceho kroku

print("1. Filtrujem dáta pre lepšiu kvalitu a výkon...")
# Zoberieme len používateľov, ktorí ohodnotili aspoň 20 kníh
user_counts = final_ratings['User-ID'].value_counts()
active_users = user_counts[user_counts >= 20].index
filtered_df = final_ratings[final_ratings['User-ID'].isin(active_users)]

# Zoberieme len knihy, ktoré majú aspoň 20 hodnotení
book_counts = filtered_df['Work-ID'].value_counts()
popular_books = book_counts[book_counts >= 20].index
filtered_df = filtered_df[filtered_df['Work-ID'].isin(popular_books)]

print(f"Počet záznamov po filtrovaní: {len(filtered_df)}")

print("2. Vytváram Pivot maticu (Knihy vs Používatelia)...")
# index = Knihy, columns = Používatelia, values = Hodnotenie
book_user_matrix = filtered_df.pivot(index='Work-ID', columns='User-ID', values='Book-Rating')

# Miesta, kde používateľ knihu nehodnotil, vyplníme nulou
book_user_matrix.fillna(0, inplace=True)

from sklearn.metrics.pairwise import cosine_similarity

print("3. Počítam Kosínovú podobnosť medzi knihami...")
# Vypočíta maticu podobnosti (Kniha vs Kniha)
item_similarity = cosine_similarity(book_user_matrix)

# Pre lepšiu prácu z toho urobíme Pandas DataFrame
item_similarity_df = pd.DataFrame(item_similarity, index=book_user_matrix.index, columns=book_user_matrix.index)

def odporuc_knihy(id_knihy, matica_podobnosti, top_n=5):
    # Vyhľadáme riadok pre danú knihu
    podobnosti_knihy = matica_podobnosti[id_knihy]
    
    # Zoradíme od najväčšej podobnosti (1.0) po najmenšiu
    zoradene = podobnosti_knihy.sort_values(ascending=False)
    
    # Preskočíme prvú (lebo kniha je 100% podobná sama sebe) a vezmeme top N
    odporucania = zoradene.iloc[1:top_n+1]
    
    return odporucania

# TEST: Dosad sem reálne Work-ID Pána Prsteňov (alebo inej knihy, ktorá zostala po filtrovaní)
ukazkove_id = item_similarity_df.index[0] 
print(f"\nOdporúčania pre knihu: {ukazkove_id}")
print(odporuc_knihy(ukazkove_id, item_similarity_df))