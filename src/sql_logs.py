import sqlite3
from sentence_transformers import SentenceTransformer
from .config import MODEL_NAME
import sqlite3
import pandas as pd

# Path to your SQLite database

def view_all_entries(exclude_query=True,db_path=r"D:\Desktop\Github for resume\LatticeBuild_Project\src\logs\rag_logs.db"):
    conn = sqlite3.connect(db_path)

    # Specify columns to exclude 'query_token'
    if exclude_query:
        query = """
            SELECT 
                id, timestamp, query_exec_time, query_token_length,
                model_name, vector_db, local_use 
            FROM rag_logs;
        """
    else:
        query = "SELECT * FROM queries;"

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def log_rag_event(
    query_exec_time,
    query,
    model_name,
    vector_db,
    local_use=True,
    db_path=r"D:\Desktop\Github for resume\LatticeBuild_Project\src\logs\rag_logs.db"
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    model = SentenceTransformer(MODEL_NAME)
    query_token = model.encode(query)
    cursor.execute("""
        INSERT INTO rag_logs (
            query_exec_time, query_token, query_token_length,
            model_name, vector_db, local_use
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        query_exec_time,
        query_token,
        len(query_token),
        model_name,
        vector_db,
        local_use
    ))

    conn.commit()
    conn.close()
# Connect to the database (or create it if it doesn't exist)

from datetime import datetime



def create_log_table():
    conn = sqlite3.connect(r"src/logs/rag_logs.db")
    cursor = conn.cursor()

    # SQL to create the table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS rag_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        query_exec_time INTEGER,
        query_token TEXT,
        query_token_length INTEGER,
        model_name TEXT,
        vector_db TEXT,
        local_use BOOLEAN
    );
    """


    # Execute the SQL and commit changes
    cursor.execute(create_table_sql)
    conn.commit()

    # Close the connection
    conn.close()

    print("Table 'rag_logs' created successfully.")




if __name__ == "__main__":
    df = view_all_entries()
    print(df.to_markdown(index=False))  # Clean table-like output




