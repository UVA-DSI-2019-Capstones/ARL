import sqlite3
from sqlite3 import Error

import sqlite3
from sqlite3 import Error
 
 
def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        conn.close()
 
if __name__ == '__main__':
    create_connection("database.db")




# CREATE TABLE IF NOT EXISTS avatar_prompts (
#  id integer PRIMARY KEY,
#  avatar_name text NOT NULL,
#  experiment text,
#  end_date text
# );