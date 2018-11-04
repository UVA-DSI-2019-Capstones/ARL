import pandas as pd
file = "Context_DME.xlsx"
ctxt_df = pd.read_excel(open(file, 'rb'))

j = 16
a = ctxt_df[ctxt_df['Identifier'] == j]['Context'][j-1]

print(a)