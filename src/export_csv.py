import os

import psycopg2
import pandas as pd

def export_to_csv():
    conn = psycopg2.connect(
        host="db",
        database="bluesky_db",
        user="user_thumalien",
        password=os.getenv("POSTGRES_PASSWORD")
    )
    
    # récupèration les données
    query = "SELECT content, lang FROM posts ORDER BY created_at DESC"
    df = pd.read_sql(query, conn)
    
    # exportation dans le dossier data 
    df.to_csv('/app/data/export_posts.csv', index=False)
    print("Export réussi dans data/export_posts.csv !")
    conn.close()

if __name__ == "__main__":
    export_to_csv()