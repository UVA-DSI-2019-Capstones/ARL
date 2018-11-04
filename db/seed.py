import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
        return(conn)
    except Error as e:
        print(e)

path = '../mechanical_turk/Dialogue.xlsx'
dme = pd.ExcelFile(path)
sheet = dme.parse('DME')
avatar_text = sheet[sheet['Speaker'] != 'Player']

avatar_prompt_data = []

for index, row in avatar_text.iterrows(): 
    avatar_name = row['Speaker']
    if avatar_name == 'CPT Wang':
        culture = 'chinese'
    else:
        culture = 'american'

    experiment = 'DME'
    subsection = row['Sub-Section']
    identifier = row['Identifier']
    text = row['Dialogue Text']
    comment = ''

    avatar_prompt_data.append((avatar_name,experiment,subsection,identifier,text,culture,comment))


trainee_response_data = []





db = create_connection("database.db")
c = db.cursor()
c.executemany('insert into avatar_prompt(avatar_name, experiment, sub_section, identifier, avatar_prompt_text, avatar_culture, comment) values ( ?,?,?,?,?,?,?)', avatar_prompt_data)
db.commit()
db.close()







# for row in sheet:
#     print(row)
