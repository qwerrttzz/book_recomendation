from fastapi import FastAPI, HTTPException
import redis
import json

app = FastAPI()
db = redis.Redis(host='redis-db', port=6379, db=0, decode_responses=True)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/recommendations/{isbn}")
def get_recommendations(isbn: str):
    # 1. Hľadáme knihu v Redise
    vysledok = db.get(isbn)
    
    # 2. Ak ISBN v databáze vôbec nemáme (Napríklad úplne neznáma kniha)
    if not vysledok:
        raise HTTPException(status_code=404, detail=f"Kniha s ISBN {isbn} nebola nájdená v našej databáze.")
    
    # 3. Redis nám vrátil String, preklopíme ho späť na JSON (Zoznam objektov)
    odporucania = json.loads(vysledok)
    
    # 4. Ak je kniha v DB, ale nenašli sa pre ňu žiadne prepojenia (Tvoj "Trash" dataset)
    if not odporucania:
        return {
            "isbn": isbn,
            "status": "No recommendations",
            "message": "Knihu poznáme, ale zatiaľ pre ňu nemáme dostatok dát na odporúčania.",
            "recommendations": []
        }
        
    # 5. Úspech: Vraciame krásny zoznam kníh
    return {
        "isbn": isbn,
        "status": "OK",
        "recommendations": odporucania
    }