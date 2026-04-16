import requests
import os
from codecarbon import EmissionsTracker
import psycopg2
import time
import traceback
import logging

logging.getLogger("codecarbon").setLevel(logging.WARNING)

HANDLE = os.getenv("BLUESKY_HANDLE")
PASSWORD = os.getenv("BLUESKY_PASSWORD")

KEYWORDS = [
    # Fake news / désinformation
    "complot", "desinformation", "fake news", "intox", "rumeur",
    "fact check", "vrai ou faux", "hoax",
    # Sujets sensibles FR
    "vaccin", "immigration", "islam", "woke", "censure",
    "medias", "gouvernement", "macron","politique", "france", "actu", "europe", "climat", "elections",
    "breaking news", "tech", "AI", "gaming", "music", "world news",
    "politics", "climate", "science", "health", "space", "trending",
    # EN
    "misinformation", "conspiracy", "propaganda", "donald trump", "climate change", "election fraud", "vaccine safety", "5g health", "deepfake"
]

def get_db_connection():
    return psycopg2.connect(
        host="db",
        database="bluesky_db",
        user="user_thumalien",
        password=os.getenv("POSTGRES_PASSWORD")
    )

def get_session():
    url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    resp = requests.post(url, json={"identifier": HANDLE, "password": PASSWORD})
    j = resp.json()
    if 'accessJwt' not in j:
        raise RuntimeError(f"Auth failed : {j}")
    print(f"Session ouverte pour : {j.get('handle')}")
    return j

def search_posts(session, keyword):
    headers = {"Authorization": f"Bearer {session['accessJwt']}"}
    posts_all = []
    cursor = None

    
    for _ in range(10):
        params = {"q": keyword, "limit": 25}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(
            "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
            headers=headers,
            params=params
        )

        if not response.ok:
            print(f"    Erreur ({response.status_code}) pour '{keyword}'")
            break

        body = response.json()
        posts = body.get('posts', [])
        posts_all.extend(posts)
        cursor = body.get('cursor')
        if not cursor:
            break
        time.sleep(0.3)

    return posts_all

def collect_all_keywords(session):
    all_posts = []
    seen_ids = set()

    for keyword in KEYWORDS:
        print(f"  Recherche : '{keyword}'...", end=" ")
        posts = search_posts(session, keyword)

        new = 0
        for post in posts:
            cid = post.get('cid')
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_posts.append(post)
                new += 1

        print(f"{new} nouveaux")
        time.sleep(0.5)

    return all_posts

def save_batch_to_db(posts):
    if not posts:
        print("   -> Aucun post a inserer.")
        return

    inserted = 0
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        for post in posts:
            record = post.get('record', {})
            lang_val = record.get('langs', ['unknown'])[0] if record.get('langs') else 'unknown'
            text = record.get('text', '')
            if not text:
                continue

            cur.execute(
                "INSERT INTO posts (bluesky_id, content, lang) VALUES (%s, %s, %s) ON CONFLICT (bluesky_id) DO NOTHING",
                (post['cid'], text, lang_val)
            )
            if cur.rowcount > 0:
                inserted += 1

        conn.commit()
        cur.close()
        conn.close()
        print(f"   -> {inserted} nouveaux posts inseres en base.")
    except Exception as e:
        print(f"Erreur SQL : {e}")
        traceback.print_exc()

def get_total_count():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM posts")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except:
        return "?"

if __name__ == "__main__":
    tracker = EmissionsTracker(log_level="warning")
    tracker.start()

    print("Pipeline de collecte en cours d'execution...")

    try:
        session = get_session()

        cycle = 0
        while True:
            cycle += 1
            total = get_total_count()
            print(f"\n=== Cycle #{cycle} | Total en base : {total} posts ===")

            all_posts = collect_all_keywords(session)
            print(f"  Collecte : {len(all_posts)} posts uniques ce cycle")
            save_batch_to_db(all_posts)

            total_after = get_total_count()
            print(f"  Total apres insertion : {total_after} posts")

            if int(total_after) >= 10000:
                print("\nObjectif 10000 posts atteint ! Arret de la collecte.")
                break

            print(f"  Pause de 1 minute...")
            time.sleep(60)

    except KeyboardInterrupt:
        print("\nArret manuel.")
    except Exception as e:
        print(f"Erreur : {e}")
        traceback.print_exc()
    finally:
        emissions = tracker.stop()
        if emissions:
            print(f"Total CO2 estime : {emissions:.6f} kg")