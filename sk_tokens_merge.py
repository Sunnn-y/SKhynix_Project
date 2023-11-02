import pandas as pd

df = pd.read_csv('sk_output.csv')
df2 = pd.read_csv('sk_output2.csv')
df3 = pd.read_csv('sk_output3.csv')

all_df = pd.concat([df, df2, df3])

all_df = all_df.drop('morphs', axis = 'columns')
all_df = all_df.drop('pos', axis = 'columns')

all_df.to_csv('sk_output_merge.csv', index=False)