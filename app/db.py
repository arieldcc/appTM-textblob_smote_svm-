import sqlite3
import os

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'history.db')

def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # This allows for column names to be accessed like dicts
    return conn

# Function to insert data into the SQLite database
def insert_into_history(title, sentiment, score):
    conn = get_db()
    cursor = conn.cursor()

    # Insert the sentiment analysis result into the history_pred table
    cursor.execute('''
        INSERT INTO history_pred (judul_berita, hasil, score, create_at)
        VALUES (?, ?, ?, datetime('now'))
    ''', (title, sentiment, score))

    conn.commit()
    conn.close()