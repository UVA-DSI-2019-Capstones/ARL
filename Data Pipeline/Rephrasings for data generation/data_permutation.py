import pandas as pd
import os
import time

print(os.getcwd())

df = pd.read_csv('question_1.csv')

exception_cases = [2]

print(df.head())
num_cols = df.shape[1] -1
index = 0

non_nan = df.index[pd.notnull(df.iloc[:,0])].tolist()
print(non_nan)
first_col = [''] + list(df.iloc[non_nan,0].transpose())
print(first_col)
master_labels = []
master_data = []
for i in range(len(first_col)):
	label = 0
	if i != 0:
		label = 2**(num_cols-1)
	print('The label is: {}'.format(label))
	# time.sleep(5)
	for p in range(len(exception_cases)):
			label += 2**(num_cols - exception_cases[p])
	new_data = [first_col[i]]
	new_labels = [label]
	print(new_labels)
	print('current i: {}'.format(i))
	for j in range(1,num_cols+1):
		print('current row {}'.format(i))
		print('current column: {}'.format(j))
		cur_len = len(new_data)
		while not(pd.isnull(df.iloc[index,j])):
			next_value = df.iloc[index,j]
			index += 1
			print('current length of new_data: {}'.format(len(new_data)))
			for k in range(cur_len):
				new_data.append(new_data[k] + ' ' +  next_value)
				#This part is unique to this example
				if j > 1:
					if j not in exception_cases:
						new_labels.append(new_labels[k]+ 2**(num_cols-j))
					else:
						new_labels.append(new_labels[k]- 2**(num_cols-j))
				else: 
					new_labels.append(new_labels[k])
		index = 0
	print(len(master_data))
	master_data += new_data
	master_labels += new_labels

bin_labels = [bin(element)[2:] for element in master_labels]
final_data = pd.DataFrame(list(zip(master_data,bin_labels)), columns = ['data', 'labels'])
final_data.to_csv('output.csv')