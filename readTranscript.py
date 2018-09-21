# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import docx
import os

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

#%%
    
files_dir = os.path.join(os.getcwd(),'Desktop\Journal\Capstone Papers\Labeled Transcripts')

files = os.listdir(files_dir)

for file in files:
    conversation = getText(os.path.join(files_dir, file))
    file = file[:-9]
    with open("{}.csv".format(os.path.join(files_dir,file)), "w") as f:
        f.write(conversation)
#%%