import sqlite3
import json

class SentimentDatabase:
    def __init__(self, db_name='stock_sentiment.db'):
        # Create an SQLite database with the specified name
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.c = self.conn.cursor()

    def setup_db(self) -> None:
        """
        Set up the SQLite database for storing sentiment data.

        This function creates the 'sentiment_data' table if it doesn't exist.
        """
        # Create the table with the appropriate schema
        self.c.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                title_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                sentiment TEXT,
                prosusai_finbert TEXT,
                finbert_tone TEXT,
                distilled_roberta TEXT,
                sigma TEXT,
                farshid_allagree2 TEXT,
                twitter_roberta TEXT,
                deberta_v3 TEXT,
                stockname TEXT
            )
        ''')
        self.conn.commit()

    def insert_stock_title(self, stockname, title):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        # Insert stock name and title into the database with None values for the additional fields
        c.execute('''
            INSERT INTO sentiment_data (
                stockname,
                title,
                prosusai_finbert,
                finbert_tone,
                distilled_roberta,
                sigma,
                farshid_allagree2,
                twitter_roberta,
                deberta_v3
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (stockname, title, None, None, None, None, None, None, None))
        
        # Commit the changes and close the connection within the current thread
        conn.commit()
        conn.close()

    def update_sentiment(self, title_id, title, sentiments):
        # Update sentiment data in the database
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''
            UPDATE sentiment_data
            SET
                sentiment = ?,
                prosusai_finbert = ?,
                finbert_tone = ?,
                distilled_roberta = ?,
                sigma = ?,
                farshid_allagree2 = ?,
                twitter_roberta = ?,
                deberta_v3 = ?
            WHERE title_id = ? AND title = ?
        ''', (
            sentiments["sentiment"],
            sentiments["prosusai_finbert"],
            sentiments["finbert_tone"],
            sentiments["distilled_roberta"],
            sentiments["sigma"],
            sentiments["farshid_allagree2"],
            sentiments["twitter_roberta"],
            sentiments["deberta_v3"],
            title_id, title
        ))
        conn.commit()
        conn.close()

    def dump_to_jsonl(self, file_path):
        # Dump data from the database to a JSONL file
        with open(file_path, "a") as jsonl_file:
            for row in self.c.execute('''
                SELECT title,
                       sentiment,
                       prosusai_finbert,
                       finbert_tone,
                       distilled_roberta,
                       sigma,
                       farshid_allagree2,
                       twitter_roberta,
                       deberta_v3
                FROM sentiment_data
            '''):
                data = {
                    "Title": row[0],
                    "Analyze Title ProsusAI Finbert": {
                        "Sentiment": row[2]
                    },
                    "Analyze Title Finbert Tone": {
                        "Sentiment": row[3]
                    },
                    "Analyze Title Distilled Roberta": {
                        "Sentiment": row[4]
                    },
                    "Analyze Title Sigma": {
                        "Sentiment": row[5]
                    },
                    "Analyze Title Farshid Allagree2": {
                        "Sentiment": row[6]
                    },
                    "Analyze Title Twitter Roberta": {
                        "Sentiment": row[7]
                    },
                    "Analyze Title Deberta v3": {
                        "Sentiment": row[8]
                    },
                    "ChatGPT4": {
                        "Sentiment": None  # Initialize to None
                    },
                    "Total result": row[1]
                }
                jsonl_file.write(json.dumps(data) + "\n")

    def drop_table(self):
        try:
            # Drop the 'sentiment_data' table
            self.c.execute('DROP TABLE IF EXISTS sentiment_data')
            self.conn.commit()
            print("Table dropped successfully.")
        except sqlite3.Error as e:
            print(f"Error dropping table: {e}")

    def close_connection(self):
        # Close the database connection
        self.conn.close()
