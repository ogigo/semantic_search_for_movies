import pandas as pd

df=pd.read_csv(r"data\train.csv")

df.dropna(inplace=True)
df.drop_duplicates(subset=['synopsis'],inplace=True)
