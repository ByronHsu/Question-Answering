'''
Fill the empty cell in testing data
'''

import pandas as pd
import math

df = pd.read_excel('data/TestingData.xlsx')

for i in range(df.shape[0]):
    if(df.iloc[i, 1] != df.iloc[i, 1]): # checknan
        df.iloc[i, 1] = df.iloc[i - 1, 1]
    # print(type(df.iloc[i, 1]))
df.to_excel('data/TestingData_fill.xlsx', index = False)