# -*- coding: utf-8 -*-

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

# %%
"""Feature Engineering on Numerical data"""

# %%
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(df['Aspect'].values.reshape(-1, 1))

# %%
# Generate a unimodal distribution using the means and standard deviations of the component distributions
means, stds = gmm.means_, np.sqrt(gmm.covariances_).reshape(-1)
unimodal_data = norm.rvs(loc=means.mean(), scale=stds.mean(), size=581012)

# %%
plt.hist(unimodal_data)

# %%
unimodal_data.shape

# %%
df['Aspect_unimodal'] = unimodal_data

df['Aspect_unimodal']

# %%
from sklearn.preprocessing import StandardScaler
def standard_scaling(df,column_list):
  for i in column_list:
    tf = StandardScaler()
    df[i] = tf.fit_transform(df[i].values.reshape(-1,1))
  return df

df = standard_scaling(df,['Aspect_unimodal'])

# %%
plt.hist(df['Aspect_unimodal'])

# %%
from sklearn.preprocessing import StandardScaler
def standard_scaling(dfa,column_list):
  for i in column_list:
    tf = StandardScaler()
    dfa[i] = tf.fit_transform(dfa[i].values.reshape(-1,1))
  return dfa

# %%
df = standard_scaling(df,['Horizontal_Distance_To_Hydrology','Elevation','Slope','Vertical_Distance_To_Hydrology','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'])

# %%
df.head()

# %%
df = standard_scaling(df,['Aspect'])

# %%
df.columns

# %%
"""**Baseline model**"""

# %%
from sklearn.model_selection import train_test_split
def split_data(df,y,test_size):
  xtrain,x_test,y_train,y_test = train_test_split(df.drop(y,axis = 1),df[y],test_size = test_size)
  return xtrain,x_test,y_train,y_test

# %%
base_features = df[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40', 'Cover_Type']]

# %%
x_train,x_test,y_train,y_test = split_data(df,'Cover_Type',0.3)

# %%
"""building based on base features"""




mod = LogisticRegression(multi_class='multinomial')
mod.fit(x_train,y_train)

c = mod.predict(x_test)
print(classification_report(y_test, c))

# %%
"""**Feature importance**"""
# %%
df.shape

# %%
df.columns

# %%
all_features = df.drop('Aspect',axis = 1)
x_train,x_test,y_train,y_test = split_data(all_features,'Cover_Type',0.3)

# %%
tf = RFE(RandomForestClassifier(), n_features_to_select=30, verbose=1)
Xt = tf.fit_transform(x_train,y_train)
print("Shape =", Xt.shape)

# %%
tf.get_feature_names_out()

# %%
top_30_features = df[['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
       'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type2', 'Soil_Type4', 'Soil_Type10', 'Soil_Type12',
       'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type29',
       'Soil_Type32', 'Soil_Type33', 'Soil_Type38', 'Inceptisols',
       'Mollisols', 'Spodosols', 'Alfisols', 'Entisols',
       'Aspect_unimodal','Cover_Type']]

# %%
x_train,x_test,y_train,y_test = split_data(top_30_features,'Cover_Type',0.3)

x_train,x_cv,y_train,y_cv = train_test_split(x_train,y_train,test_size = 0.2)

# %%
"""**PCA**"""

pca = PCA(n_components=30)
pca.fit(top_30_features)

# %%
import matplotlib.pyplot as plt
plt.plot(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum())
plt.title("Principal Component Analysis(PCA)")
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# %%
# Get the explained variance ratio of each principal component
explained_var_ratio = pca.explained_variance_ratio_

# %%
# Create a dictionary with column names as keys and explained variance ratios as values
var_dict = dict(zip(x_train.columns, explained_var_ratio))

# %%
# Sort the dictionary in descending order of explained variance ratios
sorted_var_dict = {k: v for k, v in sorted(var_dict.items(), key=lambda item: item[1], reverse=True)}

# %%
# Print the sorted dictionary
print(sorted_var_dict)

# %%
explained_var_ratio

c = [2.07854152e-01, 1.71050935e-01, 1.49486485e-01, 1.27517647e-01,
       8.49656862e-02, 6.13519661e-02, 4.36782210e-02, 3.31694728e-02,
       2.65631493e-02, 1.73204038e-02, 1.26614608e-02, 8.59851891e-03,
       7.24828091e-03, 6.98442936e-03, 6.36531354e-03, 5.88123111e-03,
       5.14033141e-03, 4.29604179e-03, 3.98308916e-03, 3.48602133e-03,
       3.28142783e-03, 2.50895995e-03, 2.20774187e-03, 1.53838986e-03,
       9.82157977e-04, 8.12520082e-04, 5.39350224e-04, 2.83709007e-04,
       1.58312642e-04, 8.45939743e-05]

sum  = 0
for i in c:
  sum += i
sum

# %%
"""**Top 30 features are explaining around 99% of variance**

**Smote**
"""
# %%
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_train_smote, Y_train_smote = oversample.fit_resample(x_train, y_train)

# %%
Y_train_smote.value_counts()

# %%
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_cv = le.fit_transform(y_cv)
y_test = le.fit_transform(y_test)
Y_train_smote = le.fit_transform(Y_train_smote)

# %%
"""**Gridsearch**"""

param = { 
    'n_estimators': [100,200,300,400,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [5,10,15,20,25,50],
    'criterion' :['gini', 'entropy']
}

# grid = GridSearchCV(RandomForestClassifier(), param_grid = param_grid,cv = 5, verbose = 10,n_jobs=-1)

#random = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param, cv=3,verbose=1, random_state=42,n_jobs=-1, return_train_score=True)

# random.fit(X_train_smote,Y_train_smote)

# best_model = random.best_estimator_

# rand_predictions = best_model.predict(x_test)

# print(classification_report(x_test, rand_predictions))

# from xgboost import XGBClassifier

# estimator = XGBClassifier(
#     objective= 'binary:logistic',
#     nthread=4,
#     seed=42
# )

# parameters = {
#     'max_depth': range(2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05]
# }

# grid_search = GridSearchCV(
#     estimator=estimator,
#     param_grid=parameters,
#     scoring = 'f1_macro',
#     n_jobs = 10,
#     cv = 10,
#     verbose=True
# )

# grid_search.fit(X_train_smote,Y_train_smote)

# best_model_xg = grid_search.best_estimator_

# grid_predictions_xg = best_model_xg.predict(x_test)

# print(classification_report(y_test, grid_predictions_xg))

# grid.best_params_

# grid_search.best_params_

"""**As the dataset is hugh, GridsearchCV is not working on local system as well as google colab.**

**Hyperparameter tuning using for loop**
"""

# %%
from sklearn.metrics import f1_score
depth = [3,5,10,20,50,100]
cv_f1_score = []
train_f1_score = []
for i in depth:
  model = RandomForestClassifier(max_depth = i,n_jobs = -1,n_estimators = 10)
  model.fit(X_train_smote, Y_train_smote)
  CV = CalibratedClassifierCV(model,method = 'sigmoid')
  CV.fit(X_train_smote, Y_train_smote)
  predicted = CV.predict(x_cv)
  train_predicted = CV.predict(x_train)
  cv_f1_score.append(f1_score(y_cv,predicted,average = 'macro'))
  train_f1_score.append(f1_score(y_train,train_predicted,average = 'macro'))
  print('depth {0} is finished'.format(i))
for i in range(0,len(cv_f1_score)):
  print('f1 value score for depth =' + str(depth[i]) + ' is ' + str(cv_f1_score[i]))
plt.plot(depth,cv_f1_score,c='r')
plt.plot(depth,train_f1_score,c='b')
plt.xlabel('depth(depth of the tree)')
plt.ylabel('f1 score(train and test)')
plt.title('depth vs df1 score')
for i,score in enumerate(cv_f1_score):
  plt.annotate((depth[i],np.round(score,4)),(depth[i],np.round(cv_f1_score[i],4)))
for i,score1 in enumerate(train_f1_score):
  plt.annotate((depth[i],np.round(score1,4)),(depth[i],np.round(train_f1_score[i],4)))
index = cv_f1_score.index(max(cv_f1_score))
best_dpt = depth[index]
print('best max depth is ' + str(best_dpt))
model = RandomForestClassifier(max_depth = best_dpt,n_jobs = -1)
model.fit(X_train_smote, Y_train_smote)
predict_train = model.predict(x_train)
print('f1 score on train data ' + str(f1_score(y_train,predict_train,average = 'macro')))
train_mat = confusion_matrix(y_train,predict_train)
predict_cv = model.predict(x_cv)
print('f1 score on cv data ' + str(f1_score(y_cv,predict_cv,average = 'macro')))
cv_mat = confusion_matrix(y_cv,predict_cv)
predict_test = model.predict(x_test)
print('f1 score on test data ' + str(f1_score(y_test,predict_test,average = 'macro')))
test_mat = confusion_matrix(y_test,predict_test)
fig,ax = plt.subplots(1,3,figsize = (15,5))
sns.heatmap(ax = ax[0],data = train_mat,annot=True,fmt='g',cmap="YlGnBu")
ax[0].set_xlabel('predicted')
ax[0].set_ylabel('actual')
ax[0].title.set_text('confusion matrix for train data')
sns.heatmap(ax = ax[1],data = cv_mat,annot=True,fmt='g')
ax[1].set_xlabel('predicted')
ax[1].set_ylabel('actual')
ax[1].title.set_text('confusion matrix for CV data')
sns.heatmap(ax = ax[2],data = test_mat,annot=True,fmt='g')
ax[2].set_xlabel('predicted')
ax[2].set_ylabel('actual')
ax[2].title.set_text('confusion matrix for test data')


# %%
fig,ax = plt.subplots(1,3,figsize = (15,5))
sns.heatmap(ax = ax[0],data = train_mat,annot=True,fmt='g',cmap="Blues")
ax[0].set_xlabel('predicted')
ax[0].set_ylabel('actual')
ax[0].title.set_text('confusion matrix for train data')
sns.heatmap(ax = ax[1],data = cv_mat,annot=True,fmt='g',cmap="Blues")
ax[1].set_xlabel('predicted')
ax[1].set_ylabel('actual')
ax[1].title.set_text('confusion matrix for CV data')
sns.heatmap(ax = ax[2],data = test_mat,annot=True,fmt='g',cmap="Blues")
ax[2].set_xlabel('predicted')
ax[2].set_ylabel('actual')
ax[2].title.set_text('confusion matrix for test data')

# %%
from sklearn.metrics import f1_score
estimators = [10,20,50,100]
cv_f1_score = []
train_f1_score = []
for i in estimators:
  model = RandomForestClassifier(n_estimators = i,max_depth = 50,n_jobs = -1)
  model.fit(X_train_smote, Y_train_smote)
  CV = CalibratedClassifierCV(model,method = 'sigmoid')
  CV.fit(X_train_smote, Y_train_smote)
  predicted = CV.predict(x_cv)
  train_predicted = CV.predict(x_train)
  cv_f1_score.append(f1_score(y_cv,predicted,average = 'macro'))
  train_f1_score.append(f1_score(y_train,train_predicted,average = 'macro'))
  print('Estimator {0} is finished'.format(i))
for i in range(0,len(cv_f1_score)):
  print('f1 value score n_estimators =' + str(estimators[i]) + ' is ' + str(cv_f1_score[i]))
plt.plot(estimators,cv_f1_score,c='r')
plt.plot(estimators,train_f1_score,c='b')
plt.xlabel('estimators(n_estimators)')
plt.ylabel('f1 score(train and test)')
plt.title('estimators vs df1 score')
for i,score in enumerate(cv_f1_score):
  plt.annotate((estimators[i],np.round(score,4)),(estimators[i],np.round(cv_f1_score[i],4)))
for i,score1 in enumerate(train_f1_score):
  plt.annotate((estimators[i],np.round(score1,4)),(estimators[i],np.round(train_f1_score[i],4)))
index = cv_f1_score.index(max(cv_f1_score))
best_est = estimators[index]
print('best estimator is ' + str(best_est))
model = RandomForestClassifier(n_estimators = best_est,max_depth = 50,n_jobs = -1)
model.fit(X_train_smote, Y_train_smote)
predict_train = model.predict(X_train_smote)
print('f1 score on train data ' + str(f1_score(Y_train_smote,predict_train,average = 'macro')))
train_mat = confusion_matrix(Y_train_smote,predict_train)
predict_cv = model.predict(x_cv)
print('f1 score on cv data ' + str(f1_score(y_cv,predict_cv,average = 'macro')))
cv_mat = confusion_matrix(y_cv,predict_cv)
predict_test = model.predict(x_test)
print('f1 score on test data ' + str(f1_score(y_test,predict_test,average = 'macro')))
test_mat = confusion_matrix(y_test,predict_test)
fig,ax = plt.subplots(1,3,figsize = (15,5))
sns.heatmap(ax = ax[0],data = train_mat,annot=True,fmt='g',cmap="YlGnBu")
ax[0].set_xlabel('predicted')
ax[0].set_ylabel('actual')
ax[0].title.set_text('confusion matrix for train data')
sns.heatmap(ax = ax[1],data = cv_mat,annot=True,fmt='g')
ax[1].set_xlabel('predicted')
ax[1].set_ylabel('actual')
ax[1].title.set_text('confusion matrix for CV data')
sns.heatmap(ax = ax[2],data = test_mat,annot=True,fmt='g')
ax[2].set_xlabel('predicted')
ax[2].set_ylabel('actual')
ax[2].title.set_text('confusion matrix for test data')

# %%
for i in range(0,len(cv_f1_score)):
  print('f1 value score n_estimators =' + str(estimators[i]) + ' is ' + str(cv_f1_score[i]))
plt.ylim(ymin=0)
plt.plot(estimators,cv_f1_score,c='r')
plt.plot(estimators,train_f1_score,c='b')
plt.xlabel('estimators(n_estimators)')
plt.ylabel('f1 score(train and test)')
plt.title('estimators vs df1 score')
for i,score in enumerate(cv_f1_score):
  plt.annotate((estimators[i],np.round(score,4)),(estimators[i],np.round(cv_f1_score[i],4)))
for i,score1 in enumerate(train_f1_score):
  plt.annotate((estimators[i],np.round(score1,4)),(estimators[i],np.round(train_f1_score[i],4)))

# %%
fig,ax = plt.subplots(1,3,figsize = (15,5))
sns.heatmap(ax = ax[0],data = train_mat,annot=True,fmt='g',cmap="Blues")
ax[0].set_xlabel('predicted')
ax[0].set_ylabel('actual')
ax[0].title.set_text('confusion matrix for train data')
sns.heatmap(ax = ax[1],data = cv_mat,annot=True,fmt='g',cmap="Blues")
ax[1].set_xlabel('predicted')
ax[1].set_ylabel('actual')
ax[1].title.set_text('confusion matrix for CV data')
sns.heatmap(ax = ax[2],data = test_mat,annot=True,fmt='g',cmap="Blues")
ax[2].set_xlabel('predicted')
ax[2].set_ylabel('actual')
ax[2].title.set_text('confusion matrix for test data')

"""**We got best tree depth as 50 and number of estimetors as 100**"""

# %%

"""Fitting best model"""

best_rf_model = RandomForestClassifier(n_estimators=100,criterion='gini', max_depth=50,max_features = 'auto')

best_rf_model.fit(X_train_smote, Y_train_smote)

pred=best_rf_model.predict(x_test)

print(classification_report(y_test, pred))

pred_cv=best_rf_model.predict(x_cv)

print(classification_report(y_cv, pred_cv))

# %%

"""**Tried number of estimeter = 200 on separate model and got better F1 macro score.(200 estimator will be really heavy value for hyperparameter tuning so tried with it in a separate model)**"""

best_rf_model = RandomForestClassifier(n_estimators=200,criterion='entropy', max_depth=50,max_features = 'auto')

best_rf_model.fit(X_train_smote, Y_train_smote)

pred=best_rf_model.predict(x_test)
print(classification_report(y_test, pred))


# %%
"""**MLP classifer**"""

from sklearn.neural_network import MLPClassifier
mlp_gs = MLPClassifier(max_iter=5)
parameter_space = {
    'hidden_layer_sizes': [(50,),(100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
# %%
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train_smote, Y_train_smote) # X is train samples and y is the corresponding labels

# %%
clf.best_params_

# %%
mlp_gs = MLPClassifier(activation= 'relu',
 alpha= 0.0001,
 hidden_layer_sizes= (100,),
 learning_rate= 'constant',
 solver= 'adam')
mlp_gs.fit(X_train_smote, Y_train_smote)

# %%
pred=mlp_gs.predict(x_test)

# %%
print(classification_report(y_test, pred))

# %%
pred_cv=mlp_gs.predict(x_cv)

# %%
print(classification_report(y_cv, pred_cv))