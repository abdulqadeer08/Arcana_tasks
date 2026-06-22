import pandas as pd
# df = pd.DataFrame({
#     'name': ['Qadeer', 'Yousf', 'Umed', 'Usman'],
#     'age': [22, 23, 20, 24],
#     'city': ['Karachi', 'Lahore', 'Islamabad', 'Peshawar']})
# print(df)

df = pd.read_csv("iris.csv")
# print(df.head())
# print(df.describe())
# print(df.info())
# print(df.head())
# print(df.dropna())
df = df.fillna(0, inplace = True)
# print (df.head())
df.rename(columns= {"sepal.length":"SL"}, inplace = True)
print(df)
df.to_csv("export.csv")