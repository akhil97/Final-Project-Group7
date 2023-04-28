import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

X = df.iloc[:,0:-1]
Y = df.iloc[:,-1: ]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)
pca = PCA(n_components=54)
pca.fit(X_scaled)

import matplotlib.pyplot as plt
plt.plot(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum())
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Get the explained variance ratio of each principal component
explained_var_ratio = pca.explained_variance_ratio_

# Create a dictionary with column names as keys and explained variance ratios as values
var_dict = dict(zip(X.columns, explained_var_ratio))

# Sort the dictionary in descending order of explained variance ratios
sorted_var_dict = {k: v for k, v in sorted(var_dict.items(), key=lambda item: item[1], reverse=True)}

# Print the sorted dictionary
print(sorted_var_dict)


plt.hist(df['Cover_Type'], bins=7, edgecolor='black');
plt.title('Before oversampling')
plt.show()

sm = SMOTE()
X_train_SMOTE, y_train_SMOTE = sm.fit_resample(X_train, y_train)

resampled_df = pd.concat([X_train_SMOTE, y_train_SMOTE], axis = 1)

plt.hist(resampled_df['Cover_Type'], bins=7, edgecolor='black');
plt.title('After oversampling')
plt.show()

