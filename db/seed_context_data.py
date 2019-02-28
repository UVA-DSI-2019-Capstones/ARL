import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
from os import listdir, path
import os

context_file_path = './turk_data/context_data/context_data_master_1.csv'

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return(conn)
    except Error as e:
        print(e)

def diff(first, second):
    second = set(second)
    return([item for item in first if item not in second])

db = create_connection("database.db")
c = db.cursor()

ratings_map = {
    1: {
        1: 'a',
        2: 'b',
        3: 'c'
    },
    2: {
        1: 'c',
        2: 'b',
        3: 'a'
    },
    3: {
        1: 'b',
        2: 'c',
        3: 'a'
    },
    4: {
        1: 'a',
        2: 'b',
        3: 'c'
    },
    5: {
        1: 'b',
        2: 'a',
        3: 'c'
    },
    6: {
        1: 'c',
        2: 'b',
        3: 'a'
    },
    7: {
        1: 'b',
        2: 'c',
        3: 'a'
    },
    8: {
        1: 'b',
        2: 'c',
        3: 'a'
    },
    9: {
        1: 'b',
        2: 'c',
        3: 'a'
    },
    10: {
        1: 'd',
        2: 'a',
        3: 'c',
        4: 'b'
    },
    11: {
        1: 'a',
        2: 'b'
    },
    12: {
        1: 'b',
        2: 'a',
        3: 'c'
    },
    13: {
        1: 'b',
        2: 'a'
    },
    14: {
        1: 'c',
        2: 'a',
        3: 'b'
    },
    15: {
        1: 'd',
        2: 'a',
        3: 'c'
    },
    16: {
        1: 'a',
        2: 'c'
    },
    17: {
        1: 'd',
        2: 'b',
        3: 'a'
    },
    18: {
        1: 'd',
        2: 'c',
        3: 'a'
    },
    19: {
        1: 'c',
        2: 'a'
    }
}


def get_correct_mapping(section, label):
    return(str(section) +  ratings_map[section][label])

def insert_csv_data_into_db(file_name):
    turk_data = pd.read_csv(file_name)
    data_to_insert = []
    for index, row in turk_data.iterrows():
        try:
            prompt_type = 'context'
            assignmentID = row['ID']
            response = row['Unlabeled Text']
            section = str(row['Section'])
            identifier = get_correct_mapping(int(section), int(row['Mode']))
            data_to_insert.append((response, prompt_type, assignmentID, identifier, ''))
        except:
            pass

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


insert_csv_data_into_db(context_file_path)
db.close()
