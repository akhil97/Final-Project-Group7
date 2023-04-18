import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Read dataset
df = pd.read_csv('../Data/covtype.csv')

#Printing data types
print(df.dtypes)

# Prrinting the first 5 rows of the dataset
print(df.head())

# About the dataset
print(df.info())

# Statistical understanding of the data
df.describe()

print("Columns of the dataset:", df.columns.tolist())
print("Number of rows in dataset:", len(df))

# Checking for any null values
print("To check if any null values are present in the dataset", df.isnull().sum())

# Unique values present in each feature
print("The unique values in each feature are: {}".format(df.nunique()))

#Check for NA
print("Number of NAs\n: {}".format(df.isna().sum()))\

# Heatmap for correlation
correlation_matrix = df.corr()
k = 55 # number of variables for heatmap
cols = correlation_matrix.nlargest(k,'Cover_Type')['Cover_Type'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(30,30))  # Sample figsize in inches
hm = sns.heatmap(cm, cbar=True, annot=True, cmap = "Blues",
                square=True, fmt='.01f',
                annot_kws={'size': 10},
                yticklabels=cols.values,
                xticklabels=cols.values,ax=ax)
plt.title("Correlation Matrix")
plt.show()


