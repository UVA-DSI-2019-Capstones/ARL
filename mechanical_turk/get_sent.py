import os
import csv

#Move to the directory of the dataset
os.chdir("..")
dir = os.getcwd()
os.chdir(dir + '\\data_processing\\unlabeled_data')

#Function for reading the paragraphs of the document
from docx import Document
def getText(filename):
    doc = Document(filename)
    fullText = []
    for para in doc.paragraphs:
          fullText.append(para.text)
    return fullText

#function for creating a flat list of sentences. The string of dialogue
# can be split into sentences using string.split('.'), but if there are
#any questions, you will get entries like: "Hello? Everyone." in the list 
#flat_chunks splits these entries and creates a flat list containing all
#sentences
def flat_chunks(list_of_sents):
	for sent in list_of_sents:
		sent = sent.strip()
		# Don't add sentences that are just periods
		if sent != '':
			if '?' in sent:
				entry = sent.split('?', 1)
				entry[0] = entry[0] + "?"
				f_chunks.append(entry[0])
				testee.append(post_file[i][:4])
				flat_chunks(entry[1:])
			else:
				entry = sent + '.'
				f_chunks.append(entry)
				testee.append(post_file[i][:4])

#Get the names of the text files
file = open('correct_listwav.txt', 'r')
post_file = []
for line in file:
    line = line.split('\n')[0]
    if 'post' in line.casefold():
        post_file.append(line)

#Create the overall list
f_chunks = []
testee = []
#Loop over files
for i in range(0,len(post_file)):
	testee_text = []
	#For each file read the text
	fultext = getText(post_file[i]+'.docx')
	#Loop over paragraphs in a doc
	for j in range(0,len(fultext)):
		#Check if the paragraph is dialoge from the testee
		if fultext[j][0] == 'T':
			#This section locates docs that do not correctly use the format
			#'T:' to mark testee text
			try:
				cur_text = fultext[j].split(':')[1]
			except:
				print(post_file[i])
			#Append the paragraph to a list
			testee_text.append(cur_text)
	for lines in testee_text:
		#Split each paragraph 
		lines = lines.split('.')
		#Use flat_chunks to make sure questions are split (see above)
		flat_chunks(lines)



#Save to a text file containing a list of every sentence
#Turn our list into a list of tuples to write to the columns of the csv
rows = zip(testee, f_chunks)
os.chdir("..")
with open('sent.csv', 'w', newline = '') as csvf:
	writer = csv.writer(csvf, delimiter=',', quoting = csv.QUOTE_ALL)
	for row in rows:
		writer.writerow(row)
