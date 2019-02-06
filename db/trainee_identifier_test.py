import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error

path = '../mechanical_turk/Dialogue.xlsx'
dme = pd.ExcelFile(path)
dme_scores = dme.parse('DME Score')
dme_sheet = dme.parse('DME')

identifiers = []
trainee_response_data = []
feedbacks = []
n = 1

for index, row in dme_scores.iterrows():
    avatar_prompt_id = n
    n = n + 1
    identifier = row['Identifier']
    response_text = row['Dialogue Text']
    response_score = row['Average']
    sheet_row = dme_sheet[dme_sheet['Identifier'] == identifier]

    if sheet_row is None or sheet_row['Feedback']is None or sheet_row['Feedback'].empty:
        feedback = ''
    else:
        feedback = sheet_row['Feedback'].values[0]
    comment = ''
    trainee_response_data.append((avatar_prompt_id, identifier, response_text, round(float(response_score), 2), feedback, comment))

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return(conn)
    except Error as e:
        print(e)

db = create_connection("database.db")
c = db.cursor()
c.executemany('insert into trainee_response(avatar_prompt_id, identifier, response_text, response_score, response_feedback, comment) values (?,?,?,?,?,?)', trainee_response_data)

db.commit()
db.close()