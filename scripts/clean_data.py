import pandas as pd

# Načítanie dát z Books.csv
# Používame špecifické parametre pre tento Kaggle dataset
books_df = pd.read_csv(
    '../data/book-recommendation-dataset/Books.csv', 
    sep=';',                # Oddeľovačom je bodkočiarka
    encoding='latin-1',     # Rieši problémy so špeciálnymi znakmi v názvoch
    on_bad_lines='skip',    # Ak je v datasete nejaký rozbitý riadok, jednoducho ho preskočí
    low_memory=False        # Zabráni varovaniam pri zmiešaných dátových typoch
)

# Zobrazenie prvých 5 riadkov, aby si videl, čo v tom datasete vlastne máš
print("Ukážka dát:")
print(books_df.head())

# Vypísanie informácií o datasete (počet riadkov, názvy stĺpcov)
print("\nZákladné info o tabuľke:")
books_df.info()