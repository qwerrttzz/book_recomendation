import pandas as pd

# Ak máš súbory v iných priečinkoch, uprav si cesty (napr. '../data/...')
CESTA_BOOKS = '../data/book-recommendation-dataset/Books_Enriched.csv'
CESTA_TEXTY = '../data/book-recommendation-dataset/spracovane_texty.csv'
VYSTUP = '../data/book-recommendation-dataset/Books_Enriched_2.csv'

print("1. Načítavam pôvodné obohatené knihy (Books_Enriched)...")
# dtype=str zaistí, že nám Pandas nezmaže nuly na začiatku ISBN kódov
books_df = pd.read_csv(CESTA_BOOKS, dtype=str, low_memory=False)

print("2. Načítavam stiahnuté texty a témy (spracovane_texty)...")
texts_df = pd.read_csv(CESTA_TEXTY, dtype=str, low_memory=False)

print("3. Prebieha spájanie (MERGE) podľa ISBN...")
# Odstránime duplicity v textoch pre istotu (ak sa náhodou niečo stiahlo dvakrát)
texts_df = texts_df.drop_duplicates(subset=['ISBN'])

# LEFT JOIN: Zachováme VŠETKY knihy z books_df a kde nájdeme zhodu v texts_df, prilepíme texty
konecny_df = pd.merge(books_df, texts_df, on='ISBN', how='left')

print("4. Vytváram spojený text pre tvoj model (Content_Text)...")
# Hneď si môžeme pripraviť aj ten ultimátny stĺpec pre budúci hybridný model
konecny_df['Content_Text'] = (
    konecny_df['Book-Title'].fillna('') + " " + 
    konecny_df['Book-Author'].fillna('') + " " + 
    konecny_df['Subjects'].fillna('') + " " + 
    konecny_df['Description'].fillna('')
)
# Zbavíme sa prebytočných medzier, ak nejaké polia boli prázdne
konecny_df['Content_Text'] = konecny_df['Content_Text'].str.replace('  ', ' ').str.strip()

print(f"5. Ukladám kompletný dataset do: {VYSTUP}")
konecny_df.to_csv(VYSTUP, index=False)

print("\n✅ Všetko hotovo! Tu je ukážka tvojej finálnej tabuľky:")
# Vypíšeme si len tie najzaujímavejšie stĺpce
print(konecny_df[['ISBN', 'Book-Title', 'Work-ID', 'Subjects']].head(5))