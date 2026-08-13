import pandas as pd

# Ak si súbor uložil inak, uprav si cestu:
subor = '../data/book-recommendation-dataset/ISBN_Recommendations.csv'

print("Načítavam dáta...")
# dtype=str zabezpečí, že ISBN s nulami na začiatku (napr. '01234') sa nepokazia
df = pd.read_csv(subor, dtype=str)

# Pre istotu nahradíme všetky prázdne (NaN) hodnoty prázdnym reťazcom
df = df.fillna('')

# Zoznam stĺpcov, kde sa nachádzajú odporúčania
rec_stlpce = ['Rec_1', 'Rec_2', 'Rec_3', 'Rec_4', 'Rec_5']

# Kúzlo s Pandas:
# 1. df[rec_stlpce] != '' zistí pre každú bunku, či v nej niečo je (True) alebo nie je (False).
# 2. .sum(axis=1) spočíta hodnoty True pre každý riadok. Výsledkom bude číslo od 0 do 5.
df['Pocet_Odporucani'] = (df[rec_stlpce] != '').sum(axis=1)

# Spočíta výskyty pre 0, 1, 2, 3, 4, 5
statistika = df['Pocet_Odporucani'].value_counts()
celkovy_pocet = len(df)

# Výpis výsledkov
print("\n" + "="*45)
print("📊 ŠTATISTIKA KVALITY ODPORÚČANÍ (Podľa ISBN)")
print("="*45)

# Vypíšeme to pekne zarovnané od 5 do 0
for i in range(5, -1, -1):
    # Ak sa náhodou nejaké číslo nevyskytlo vôbec, priradíme mu 0
    pocet_knih = statistika.get(i, 0)
    percento = (pocet_knih / celkovy_pocet) * 100
    
    # Špeciálne vizuálne označenie pre knihy bez odporúčaní
    ikona = "🗑️ " if i == 0 else "✅ " if i == 5 else "⚠️ "
    
    print(f"{ikona} Počet kníh s {i} odporúčaniami: {pocet_knih:>9,} ({percento:>5.1f} %)")

print("-" * 45)
print(f"Spolu kníh (ISBN) v datasete:      {celkovy_pocet:>9,}")
print("="*45)