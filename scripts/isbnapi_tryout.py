import requests
import time

def ziskaj_data_openlibrary(isbn):
    url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=details"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            key = f"ISBN:{isbn}"
            
            if key in data:
                book_info = data[key]
                
                # VŠETKO dôležité sa teraz nachádza v 'details'
                details = book_info.get('details', {})
                
                # --- Získanie Work ID ---
                works_list = details.get('works', [])
                work_id = None
                if works_list:
                    work_id = works_list[0].get('key', '').split('/')[-1]
                            
                # --- Získanie tém (z details) ---
                temy_data = details.get('subjects', [])
                temy = [tema['name'] for tema in temy_data] if temy_data else []
                
                # --- Získanie popisu (z details) ---
                popis = details.get('description', 'Popis nie je k dispozícii')
                if isinstance(popis, dict) and 'value' in popis:
                    popis = popis['value']
                    
                return popis, temy, work_id
                
    except Exception as e:
        print(f"Chyba pri sťahovaní ISBN {isbn}: {e}")
        
    return None, None, None

# --- Ukážka použitia ---
zoznam_isbn = ["9780141036137", "0439785960", "9780553103540"]

for isbn in zoznam_isbn:
    print(f"Sťahujem dáta pre ISBN: {isbn}...")
    popis, temy, work_id = ziskaj_data_openlibrary(isbn)
    
    print(f"Work ID: {work_id}")
    if temy:
        print(f"Nájdené témy: {temy[:5]}...") 
    else:
        print("Témy nenájdené.")
        
    print("-" * 30)
    time.sleep(1)