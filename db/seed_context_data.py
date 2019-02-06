import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
from os import listdir, path
import os

#context_file_path = '../mechanical_turk/Dialogue.xlsx'

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return(conn)
    except Error as e:
        print(e)

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def insert_csv_data_into_db(file_name):
    turk_data = pd.read_csv(file_name)
    data_to_insert = []
    for index, row in turk_data.iterrows():
        # whatever the ID columns is called
        prompt_type = 'context'
        assignmentID = row['ID']
        text = row['Unlabeled Text']
        section = str(row['Section'])
        # identifier = section += letter that maps to corresponding turk response
        data_to_insert.append((response, prompt_type, assignmentID, identifier, ''))
    assignments = c.execute("SELECT assignment FROM turk_response WHERE prompt_type = 'context'").fetchall()

    assignments = [thing[0] for thing in assignments]
    stuff = [(row[2]) for row in data_to_insert]
    not_in_db = diff(stuff, assignments)

    final_data_tuples = []
    
    for data_tuple in data_to_insert:
        if data_tuple[2] in not_in_db:
            final_data_tuples.append(data_tuple)

    c.executemany('INSERT INTO turk_response(turk_response_text, prompt_type, assignment, identifier, comment) VALUES (?,?,?,?,?) ', final_data_tuples)
    db.commit()
    db.close()

insert_csv_data_into_db(context_file_path)

# if NA don't include that as a datapoint

# data to include: 
# text K
# id K
# section K
# prompt_type K
# ranking/grouping
# comment
