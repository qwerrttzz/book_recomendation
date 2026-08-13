import pandas as pd

# 1. Načítanie obohateného datasetu
print("Načítavam obohatený dataset...")
df = pd.read_csv('../data/book-recommendation-dataset/Books_Enriched.csv', encoding='latin-1', low_memory=False)

# 2. Očistenie dát (Veľmi dôležitý krok)
# Musíme vyhodiť knihy, ktoré Work-ID nemajú, alebo kde server vrátil ERROR
df_platne = df[df['Work-ID'].notna()]
df_platne = df_platne[~df_platne['Work-ID'].str.startswith('ERROR', na=False)]

print(f"Počet kníh s platným Work-ID na zoskupenie: {len(df_platne)}")

# 3. Samotné zoskupenie (GROUP BY)
print("\nZoskupujem vydania (ISBN) pod spoločné Work-ID...")
works_df = df_platne.groupby('Work-ID').agg({
    'ISBN': list,                  # Všetky ISBN dáme do jedného poľa (zoznamu)
    'Book-Title': 'first',         # Názov knihy si zoberieme z prvého vydania
    'Book-Author': 'first',        # Autora si tiež zoberieme z prvého vydania
    'Year-Of-Publication': list,   # Zoznam rokov, kedy kniha vyšla
    'Publisher': list              # Zoznam vydavateľstiev
}).reset_index()

# Pridáme si stĺpec, ktorý nám povie, koľko edícií (ISBN) daná kniha má
works_df['Pocet_Edicii'] = works_df['ISBN'].apply(len)

# 4. Zoradenie od diel s najviac edíciami po najmenej
works_df = works_df.sort_values(by='Pocet_Edicii', ascending=False)

# 5. Výsledky!
print("\n🔥 TOP 5 Diel s najväčším počtom rôznych vydaní v tvojom datasete:")
print(works_df[['Book-Title', 'Book-Author', 'Pocet_Edicii']].head(5))

print("\nUkážka, ako vyzerá zoskupený riadok pre víťaznú knihu:")
# Vyberieme prvý riadok (knihu s najviac vydaniami) a vypíšeme jej detaily
najviac_vydani = works_df.iloc[0]
print(f"Dielo: {najviac_vydani['Book-Title']} (Work ID: {najviac_vydani['Work-ID']})")
print(f"Počet verzií: {najviac_vydani['Pocet_Edicii']}")
print(f"Zoznam všetkých ISBN: {najviac_vydani['ISBN'][:5]} ... (a ďalšie)")

# 6. (Voliteľné) Uloženie zoskupeného datasetu na ďalšiu prácu
#works_df.to_csv('../data/book-recommendation-dataset/Unique_Works.csv', index=False)

# Zistenie počtu riadkov
povodny_pocet = len(df_platne)
novy_pocet = len(works_df)
rozdiel = povodny_pocet - novy_pocet

print("\n📊 ŠTATISTIKA ZOSKUPOVANIA:")
print(f"Počet riadkov PRED zoskupením (počet ISBN s Work-ID): {povodny_pocet}")
print(f"Nový počet riadkov PO zoskupení (počet unikátnych diel): {novy_pocet}")
print(f"Podarilo sa ti odstrániť {rozdiel} duplicitných vydaní!")

# Zistenie počtu kníh, pre ktoré Open Library nenašlo Work-ID
pocet_bez_work_id = df['Work-ID'].isna().sum()

print(f"Počet kníh bez Work-ID (hodnota None/NaN): {pocet_bez_work_id}")
print(f"To je {pocet_bez_work_id / len(df) * 100:.2f} % z celkového počtu {len(df)} kníh.")