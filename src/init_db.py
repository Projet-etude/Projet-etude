import psycopg2
import os

def init_database():
    try:
        conn = psycopg2.connect(
            host="db",
            database="bluesky_db",
            user="user_thumalien",
            password=os.getenv("POSTGRES_PASSWORD")
        )
        cur = conn.cursor()

        cur.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id SERIAL PRIMARY KEY,
                bluesky_id TEXT UNIQUE,
                content TEXT,
                lang VARCHAR(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')

        conn.commit()
        cur.close()
        conn.close()
        print("Table 'posts' creee avec succes.")
    except Exception as e:
        print(f"Erreur d'initialisation : {e}")

if __name__ == "__main__":
    init_database()