import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
from os import listdir, path
import os

#turk_file = "./turk_data/Batch_3413977_batch_results.csv"

def create_connection(db_file):
        try:
            conn = sqlite3.connect(db_file)
            return(conn)
        except Error as e:
            print(e)



def diff(first, second):
            second = set(second)
            return [item for item in first if item not in second]
    

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


  #print(pd.read_csv(turk_csv).head())

db = create_connection("database.db")
c = db.cursor()

def insert_csv_data_into_db(file_name):
    turk_data = pd.read_csv(file_name)
    data_to_insert = []
    
    for index, row in turk_data.iterrows():
        keywords = row['Keywords']
        prompt_number = [int(s) for s in keywords.split() if s.isdigit()][0]
        assignment = row['AssignmentId']
        if 'rewrite' in row['Title'].lower():
            prompt_type = 'rewrite'
        else:
            prompt_type = 'feedback'
        
        responses = row['Answer.Q5FreeTextInput'].split('|')
        first_character = 97
        for response in responses:
            new_assignment = assignment
            new_assignment += chr(first_character)
            data_to_insert.append((response, prompt_type, new_assignment, str(prompt_number) + chr(first_character),''))
            first_character += 1
    
    
    
    assignments = c.execute("SELECT assignment FROM turk_response").fetchall()
    
    assignments = [thing[0] for thing in assignments]
    stuff = [(row[2]) for row in data_to_insert]
    not_in_db = diff(stuff, assignments)
    
    final_data_tuples = []
    
    for data_tuple in data_to_insert:
        if data_tuple[2] in not_in_db:
            final_data_tuples.append(data_tuple)
            
    c.executemany('INSERT INTO turk_response(turk_response_text, prompt_type, assignment, identifier, comment) VALUES (?,?,?,?,?) ', final_data_tuples)
    db.commit()
            
filenames = find_csv_filenames("./turk_data/")

for name in filenames:
  turk_csv = os.path.join(os.getcwd(), 'turk_data', name)
  insert_csv_data_into_db(turk_csv)

db.close()
