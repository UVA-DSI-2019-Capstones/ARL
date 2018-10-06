string = 'This is a string. It contains multiple sentences. Do you understand them? Really? I think that is great.'
chunks = string.split('.')[:-1]
#print(chunks)
f_chunks = []
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

f_chunks = flat_chunks(chunks)
print(f_chunks)

