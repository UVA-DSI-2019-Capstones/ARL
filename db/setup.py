import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
        return(conn)
    except Error as e:
        print(e)

db = create_connection("database.db")

create_avatar_prompt = """
CREATE TABLE IF NOT EXISTS avatar_prompt (
 id integer PRIMARY KEY,
 avatar_name text NOT NULL,
 experiment text NOT NULL,
 sub_section integer NOT NULL,
 identifier text NOT NULL,
 avatar_prompt_text text NOT NULL,
 avatar_culture text,
 comment text
);
"""

create_trainee_response = """
CREATE TABLE IF NOT EXISTS trainee_response (
 id integer PRIMARY KEY,
 avatar_prompt_id int NOT NULL,
 identifier text NOT NULL,
 response_text text NOT NULL,
 response_score real NOT NULL,
 response_feedback text NOT NULL,
 comment text,
 FOREIGN KEY(avatar_prompt_id) REFERENCES avatar_prompt(id)
);
"""

create_turk_response = """
CREATE TABLE IF NOT EXISTS turk_response (
 id integer PRIMARY KEY,
 turk_response_text text NOT NULL,
 prompt_type text NOT NULL,
 assignment text NOT NULL,
 identifier text NOT NULL,
 comment text,
 FOREIGN KEY(identifier) REFERENCES trainee_response(identifier)
);
"""

db.execute(create_avatar_prompt)
db.execute(create_trainee_response)
db.execute(create_turk_response)
db.commit()
db.close()