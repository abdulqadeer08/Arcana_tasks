import pandas as pd
from pathlib import Path

# CSV files live in the week's data/ folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# df = pd.DataFrame({
#     'name': ['Qadeer', 'Yousf', 'Umed', 'Usman'],
#     'age': [22, 23, 20, 24],
#     'city': ['Karachi', 'Lahore', 'Islamabad', 'Peshawar']})
# print(df)

df = pd.read_csv(DATA_DIR / "iris.csv")
# print(df.head())
# print(df.describe())
# print(df.info())
# print(df.head())
# print(df.dropna())
df = df.fillna(0, inplace = True)
# print (df.head())
df.rename(columns= {"sepal.length":"SL"}, inplace = True)
print(df)
df.to_csv(DATA_DIR / "export.csv")