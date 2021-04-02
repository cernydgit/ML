import numpy as np
import pandas as pd


#%% RUN0

list('AB')

#%% RUN1


df = pd.DataFrame(data=[[1,6], [3,8]], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df.append(df2)
   


#%% RUN2

#load data into a DataFrame object:
columns = ['A', 'B']
a = []
b = []
df1 = pd.DataFrame([[1],[2],[3]], columns = columns)
df2 = pd.DataFrame([[1],[2],[66666]], columns = columns)
df1.append(df2)
res = res.set_value(len(res), 'qty1', 10.0)
print(df1)