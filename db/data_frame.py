import sqlite3
import pandas as pd

def load_dataframe():
	conn = sqlite3.connect("database.db")
	# df = pd.read_sql_query("select * from trainee_response;", conn)
	df = pd.read_sql_query("SELECT * FROM trainee_response INNER JOIN turk_response ON trainee_response.identifier = turk_response.identifier;", conn)

	def score_round(score):
	    if score < 2.67:
	        return 1
	    elif score >= 2.67 and score < 3.67:
	        return 3
	    return 5

	df['response_round_score'] = df['response_score'].apply(score_round)

	print(df.columns)

	from sklearn.model_selection import StratifiedShuffleSplit
	stratSplit = StratifiedShuffleSplit(n_splits =1 , test_size=0.3,random_state=42)
	stratSplit.get_n_splits(df['response_text'], df['response_round_score'])
	for train_idx,test_idx in stratSplit.split(df['response_text'], df['response_round_score']):
	    train_df=df.iloc[train_idx,:]
	    test_df=df.iloc[test_idx,:]
	   

	# test_df = df[~train_df.index.isin(df.index)]
	train_df.to_csv('train_set.csv')
	test_df.to_csv('test_set.csv')

load_dataframe()