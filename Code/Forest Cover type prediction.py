dafcdcdcd
dvbg


# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression

# %%
df = pd.read_csv('/content/drive/MyDrive/Covertype.csv')

df.head()

# %%
df['Cover_Type'].value_counts()

"""As we can see that classes are imbanaced and we have good number of observations for cover type 2 and 1. For cover type 4 and 5, we have less number of observations."""

# %%
len(df)

# %%
X = df.drop('Cover_Type', axis = 1)
y = df['Cover_Type']

y.head()

# %%
"""Checking null values"""

df.isnull().sum()

"""There is no missing values in the dataset

EDA(Univariate)
"""
# %%
df_new = df.filter(['Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'], axis=1)

df_new.head(2)

# %%
df_new.describe()

fig,ax = plt.subplots(1,3,figsize = (15,5))

ax[0].violinplot(df_new['Hillshade_9am'])
ax[0].title.set_text('Violin plot for Hillshade_9am')
ax[0].set_ylabel('Hillshade_9am')
ax[1].violinplot(df_new['Hillshade_Noon'])
ax[1].title.set_text('Violin plot for Hillshade_Noon')
ax[1].set_ylabel('Hillshade_Noon')
ax[2].title.set_text('Violin plot for Hillshade_3pm')
ax[2].violinplot(df_new['Hillshade_3pm'])
ax[2].set_ylabel('Hillshade_3pm')

# %%
fig,ax = plt.subplots(1,2,figsize = (10,5))

ax[0].violinplot(df_new['Horizontal_Distance_To_Roadways'])
ax[0].title.set_text('Violin plot for Horizontal_Distance_To_Roadways')
ax[0].set_ylabel('Horizontal_Distance_To_Roadways')
ax[1].violinplot(df_new['Horizontal_Distance_To_Fire_Points'])
ax[1].title.set_text('Violin plot for Horizontal_Distance_To_Fire_Points')
ax[1].set_ylabel('Horizontal_Distance_To_Fire_Points')

# import plotly.express as px
# fig = px.scatter_3d(df, x='Hillshade_9am', y='Hillshade_Noon', z='Hillshade_3pm',
#               color='Cover_Type')
# fig.show()

# %%

"""**Soil type mapping**

Based on the soil hierarchy, we can classify the the all 40 soil types to main order in soil hierarchy.
Based on that, we are able to classify the features into 7 different soil order types.
"""

soil_orders = pd.read_excel('/content/Soil orders according to family.xlsx')

# %%
soil_orders.head()

# %%
soil_orders.Order.value_counts()

# %%
df[['Inceptisols','Mollisols','Spodosols','Alfisols','Entisols','Unknown','Histosols']] = 0

df.columns

"""**Using the soil order mapping, classifying 40 soil types into 7 main soil orders.**


"""
# %%
for i in range(0, len(df)):
  mol = ['Soil_Type1','Soil_Type3','Soil_Type4','Soil_Type7','Soil_Type8','Soil_Type1','Soil_Type14','Soil_Type16','Soil_Type17','Soil_Type18']
  alf = ['Soil_Type2','Soil_Type5','Soil_Type6','Soil_Type9','Soil_Type26']
  ent = ['Soil_Type12','Soil_Type34']
  hist = ['Soil_Type19']
  enc = ['Soil_Type10','Soil_Type11','Soil_Type32','Soil_Type28','Soil_Type13','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type33','Soil_Type27','Soil_Type25','Soil_Type38','Soil_Type31','Soil_Type29','Soil_Type30']
  spod = ['Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type39','Soil_Type40']
  Unknown = ['Soil_Type15']
  flag = 0

  if i%50000 == 0:
    print(i)

  for x1 in mol:
    if df[x1][i]== 1:
      df['Mollisols'][i] = 1
      flag = 1
      break
      
  if flag == 1:
    next
  
  for x1 in alf:
    if df[x1][i]== 1:
      df['Alfisols'][i] = 1
      flag = 1
      break

  if flag == 1:
    next

  for x1 in ent:
    if df[x1][i]== 1:
      df['Entisols'][i] = 1
      flag = 1
      break

  if flag == 1:
    next

  for x1 in hist:
    if df[x1][i]== 1:
      df['Histosols'][i] = 1
      flag = 1
      break

  if flag == 1:
    next

  for x1 in enc:
    if df[x1][i]== 1:
      df['Inceptisols'][i] = 1
      flag = 1
      break
  
  if flag == 1:
    next

  for x1 in spod:
    if df[x1][i]== 1:
      df['Spodosols'][i] = 1
      flag = 1
      break

  if flag == 1:
    next

  for x1 in Unknown:
    if df[x1][i]== 1:
      df['Unknown'][i] = 1
      break


# %%
df.head()

# %%
df.columns

# %%
updated_df = df[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Cover_Type', 'Inceptisols', 'Mollisols',
       'Spodosols', 'Alfisols', 'Entisols', 'Unknown', 'Histosols']]

# %%
updated_df.head()

# %%
"""2. Finding most impacting feature:
Some of the soil types are contributing more in some of the forest cover types.
Those features are really strong to classify the cover types.
"""

df[df['Histosols'] == 1].groupby(['Cover_Type'])['Histosols'].count()

soil_contri = pd.DataFrame(data = {'Cover_Type':[1,2,3,4,5,6,7]})

soil_contri

df[df['Soil_Type2']==1].groupby(['Cover_Type'])['Soil_Type2'].count().rename("Soil_Type2").transform(lambda x: (x/x.sum())*100).reset_index()

# %%
col = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40']

# %%
for i in col:
  soil_contri = soil_contri.merge(df[df[i]==1].groupby(['Cover_Type'])[i].count().rename(i).reset_index(), on='Cover_Type', how='left')

pd.set_option('display.max_columns', None)
soil_contri

# %%
"""category wise soil distribution"""

df['Cover_Type'].value_counts().reset_index().rename(columns = {'index':'Cover_Type','Cover_Type':'Total_count'})

soil_contri = soil_contri.merge(df['Cover_Type'].value_counts().reset_index().rename(columns = {'index':'Cover_Type','Cover_Type':'Total_count'}), on='Cover_Type', how='left')

# %%
soil_contri.head()

# %%
def my_func(x,a):
    try:
        return ((x[a]) / (x['Total_count']))* 100
    except (ZeroDivisionError, ValueError):
        return 0

col = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40']
for i in col:
  soil_contri[i] = my_func(soil_contri,i)

# %%

soil_contri

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(50,50))
k = soil_contri.drop(['Total_count'], axis = 1).plot(x='Cover_Type', kind='bar', stacked=True,title='Stacked Bar Graph by dataframe')
plt.show()

# %%
maxValues = soil_contri.drop(['Cover_Type','Total_count'],axis = 1).max(axis=1)

# %%
maxValues

# %%
"""based on these observations, we can say that Soil_Type29,Soil_Type10,Soil_Type3,Soil_Type30,Soil_Type38 are important in cover prediction"""

# %%
df.columns

# %%
wild_area = pd.DataFrame(data = {'Cover_Type':[1,2,3,4,5,6,7]})

df[df['Wilderness_Area1'] == 1].groupby(['Cover_Type'])['Wilderness_Area1'].count().rename("Wilderness_Area1")

col = ['Wilderness_Area1','Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']

# %%
for i in col:
  wild_area = wild_area.merge(df[df[i]==1].groupby(['Cover_Type'])[i].count().rename(i).reset_index(), on='Cover_Type', how='left')

wild_area

# %%

col = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']
for i in col:
  wild_area[i] = my_func(wild_area,i)


# %%
wild_area.drop('total_count',axis = 1,inplace = True)

# %%
wild_area

# %%
c = wild_area.T.reset_index()
c.columns = c.iloc[0]
c

# %%
c= c.drop(0)

# %%
wild_area.plot(x='Cover_Type', kind='bar', stacked=True,title='Stacked Bar For wilderness area')

# %%
num_df = df.iloc[:,[*range(0,10)] + [54]]
num_df

# %%
"""#working on this part with better color palette"""

sns.color_palette("hls", 7)
sns.pairplot(num_df,hue = 'Cover_Type',palette='tab10')

# from PIL import Image

# # Open the image file
# img = Image.open('/content/Untitled.png')

# # Display the image using Matplotlib
# plt.imshow(img)
# plt.show()
# %%
df.columns

# %%
from sklearn.preprocessing import StandardScaler
def standard_scaling(dfa,column_list):
  for i in column_list:
    tf = StandardScaler()
    dfa[i] = tf.fit_transform(dfa[i].values.reshape(-1,1))
  return dfa

# %%
df.to_csv('cover_type_updated_soil features.csv')