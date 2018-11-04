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
 id integer AUTO_INCREMENT,
 avatar_name text NOT NULL,
 experiment text NOT NULL,
 sub_section integer NOT NULL,
 identifier text NOT NULL,
 avatar_prompt_text text NOT NULL,
 avatar_culture text,
 comment text,
 primary key (id)
);
"""

create_trainee_response = """
CREATE TABLE IF NOT EXISTS trainee_response (
 id integer AUTO_INCREMENT,
 avatar_prompt_id int NOT NULL,
 identifier text NOT NULL,
 response_text text NOT NULL,
 response_score real NOT NULL,
 response_feedback text NOT NULL,
 comment text,
 FOREIGN KEY(avatar_prompt_id) REFERENCES avatar_prompt(id),
 primary key (id)
);
"""

create_turk_response = """
CREATE TABLE IF NOT EXISTS turk_response (
 id integer AUTO_INCREMENT,
 trainee_response_id int NOT NULL,
 turk_id text NOT NULL,
 turk_response_text text NOT NULL,
 prompt_type text NOT NULL,
 comment text,
 FOREIGN KEY(trainee_response_id) REFERENCES trainee_response(id),
 primary key (id)
);
"""

db.execute(create_avatar_prompt)
db.execute(create_trainee_response)
db.execute(create_turk_response)
db.commit()
db.close()