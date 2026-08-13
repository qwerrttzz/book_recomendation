import pandas as pd
import numpy as np

# 1. NAČÍTANIE DÁT
print("Načítavam datasety...")
books_df = pd.read_csv('../data/book-recommendation-dataset/Books_Enriched_2.csv', dtype=str, low_memory=False)
ratings_df = pd.read_csv('../data/book-recommendation-dataset/Ratings.csv', dtype={'ISBN': str})

# 2. VÝPOČET POČTU HODNOTENÍ PRE KAŽDÚ KNIHU
print("Počítam ratingy...")
ratings_count = ratings_df.groupby('ISBN').size().reset_index(name='Pocet_Hodnoteni')
df = pd.merge(books_df, ratings_count, on='ISBN', how='left')
df['Pocet_Hodnoteni'] = df['Pocet_Hodnoteni'].fillna(0).astype(int)

# 3. VYTVORENIE PODMIENOK
pod_malo_ratingov = df['Pocet_Hodnoteni'] < 10
pod_bez_temy = df['Subjects'].isna() | (df['Subjects'].str.strip() == '') | (df['Subjects'] == 'NaN')
pod_bez_popisu = df['Description'].isna() | (df['Description'].str.strip() == '') | (df['Description'] == 'NaN')

# 4. VÝPOČTY ZÁKLADNÝCH KATEGÓRIÍ
celkovo_knih = len(df)
malo_ratingov = pod_malo_ratingov.sum()
bez_temy = pod_bez_temy.sum()
bez_popisu = pod_bez_popisu.sum()

# 5. VÝPOČTY KOMBINÁCIÍ
komb_malo_ratingov_bez_temy = (pod_malo_ratingov & pod_bez_temy).sum()
komb_malo_ratingov_bez_popisu = (pod_malo_ratingov & pod_bez_popisu).sum()
komb_bez_temy_bez_popisu = (pod_bez_temy & pod_bez_popisu).sum()
komb_vsetko_zle = (pod_malo_ratingov & pod_bez_temy & pod_bez_popisu).sum()
komb_zlate_knihy = (~pod_malo_ratingov & ~pod_bez_temy & ~pod_bez_popisu).sum()
komb_bez_temy_s_popisom = (pod_bez_temy & ~pod_bez_popisu).sum()

# 6. VÝPIS VÝSLEDKOV (Základ)
print("\n" + "="*50)
print("📊 ANALÝZA KVALITY DATASETU")
print("="*50)
print(f"Celkový počet kníh (ISBN): {celkovo_knih:,}")

print("\n--- ZÁKLADNÉ PROBLÉMY ---")
print(f"Knihy s < 10 hodnoteniami:         {malo_ratingov:,} ({(malo_ratingov/celkovo_knih)*100:.1f} %)")
print(f"Knihy BEZ tém (Subjects):          {bez_temy:,} ({(bez_temy/celkovo_knih)*100:.1f} %)")
print(f"Knihy BEZ popisu (Description):    {bez_popisu:,} ({(bez_popisu/celkovo_knih)*100:.1f} %)")
print(f"BEZ tém, ale MÁ popis:                    {komb_bez_temy_s_popisom:,}")

print("\n--- PRIENIKY PROBLÉMOV (Kombinácie) ---")
print(f"👻 ÚPLNÍ DUCHOVIA (<10 hod + bez všetkého): {komb_vsetko_zle:,}")
print(f"🏆 ZLATÝ FOND (≥10 hod + popis + témy):     {komb_zlate_knihy:,}")


# ==========================================
# 7. HĹBKOVÁ ANALÝZA TÉM (SUBJECTS)
# ==========================================
print("\n" + "="*50)
print("🏷️ HĹBKOVÁ ANALÝZA TÉM (SUBJECTS)")
print("="*50)

# Vyfiltrujeme len knihy, ktoré MAJÚ nejaké témy
df_s_temami = df[~pod_bez_temy].copy()

if not df_s_temami.empty:
    # Rozdelíme reťazce podľa oddeľovača "|" (vyčistíme od medzier)
    df_s_temami['List_Subjects'] = df_s_temami['Subjects'].apply(
        lambda x: [s.strip() for s in str(x).split('|') if s.strip()]
    )
    
    # 1. Štatistiky na úroveň knihy
    pocet_tem_na_knihu = df_s_temami['List_Subjects'].apply(len)
    priemer_tem = pocet_tem_na_knihu.mean()
    max_tem = pocet_tem_na_knihu.max()
    
    # Rozbalíme všetky zoznamy tém do jedného obrovského Pandas stĺpca (Series)
    vsetky_temy_exploded = df_s_temami['List_Subjects'].explode()
    
    # Získame frekvenciu (početnosť) každej unikátnej témy
    temy_counts = vsetky_temy_exploded.value_counts()
    
    pocet_unikatnych_tem = len(temy_counts)
    temy_len_raz = (temy_counts == 1).sum()
    temy_viac_ako_10x = (temy_counts >= 10).sum()
    
    print(f"Počet kníh, ktoré majú aspoň 1 tému: {len(df_s_temami):,}")
    print(f"Priemerný počet tém na jednu knihu:  {priemer_tem:.1f}")
    print(f"Rekordman: najviac tém na 1 knihe:   {max_tem}")
    print(f"\nCelkový počet UNIKÁTNYCH tém:        {pocet_unikatnych_tem:,}")
    
    print("\n--- DISTRIBÚCIA A ZNOVUPOUŽITEĽNOSŤ TÉM ---")
    print(f"Témy použité LEN RAZ (šum):          {temy_len_raz:,} ({(temy_len_raz/pocet_unikatnych_tem)*100:.1f} % zo všetkých tém)")
    print(f"Silné témy (použité ≥ 10x):          {temy_viac_ako_10x:,} ({(temy_viac_ako_10x/pocet_unikatnych_tem)*100:.1f} % zo všetkých tém)")
    
    print("\n--- 🏆 TOP 15 NAJČASTEJŠÍCH TÉM ---")
    for tema, pocet in temy_counts.head(15).items():
        print(f"  - {tema}: {pocet:,} kníh")
else:
    print("Nenašli sa žiadne knihy s témami.")
    
import random  # Uisti sa, že máš toto na začiatku súboru, alebo to daj sem

# ... (tvoj doterajší kód končí výpisom TOP 15 tém) ...

print("\n--- 🕵️ UKÁŽKA 100 TÉM Z KATEGÓRIE 'ŠUM' (Použité len raz) ---")
# Vyfiltrujeme len tie témy, ktoré majú count == 1
temy_sum_zoznam = temy_counts[temy_counts == 1].index.tolist()

if temy_sum_zoznam:
    # Vyberieme 100 náhodných (aby sme nevideli len tie zoradené podľa abecedy)
    vzorka_sumu = random.sample(temy_sum_zoznam, min(100, len(temy_sum_zoznam)))
    
    for i, tema in enumerate(vzorka_sumu, 1):
        print(f"{i:3d}. {tema}")