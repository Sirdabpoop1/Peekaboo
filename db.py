import pickle
import sqlite3

conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL,
               encoding BLOB NOT NULL
    )
    
''')

conn.commit()

def get_faces():
    cursor.execute("SELECT name, encoding FROM faces")
    rows = cursor.fetchall()
    names = []
    encodings = []
    for name, blob in rows:
        encodings.append(pickle.loads(blob))
        names.append(name)

    return names, encodings

def new_face(name, encoding):
    encoding_blob = pickle.dumps(encoding)
    cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_blob))
    conn.commit()

def close_db():
    conn.close