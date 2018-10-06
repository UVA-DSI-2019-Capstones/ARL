import os

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
		if '?' in sent:
			entry = sent.split('?', 1)
			entry[0] = entry[0] + "?"
			f_chunks.append(entry[0])
			flat_chunks(entry[1:])
		else:
			entry = sent + '.'
			f_chunks.append(entry)
	return(f_chunks)

#Get the names of the text files
file = open('correct_listwav.txt', 'r')
post_file = []
for line in file:
    line = line.split('\n')[0]
    if 'post' in line.casefold():
        post_file.append(line)

#Create the overall list
f_chunks = []
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
		f_chunks = flat_chunks(lines)

#Remove sentences that are just periods
f_chunks = [x for x in f_chunks if x != '.' and x != ' .']

#Save to a text file containing a list of every sentence
os.chdir("..")
f = open('sent.txt', '+w')
for line in f_chunks:
	f.write(line + '\n')
f.close()
