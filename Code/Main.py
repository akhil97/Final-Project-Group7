
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
import EDA_forest_cover as eda
import Feature_Engineering as FE
import Preprocessing as pp
import Baseline_model as bm
import Feature_reduction_RFE as RFE
import Smote_oversampling as SMO
import Split_data as tts
import model_building as mb


##############################loading data####################
# %%
df = pd.read_csv('../Data/covtype.csv')

df.head()

####################class values########################
# %%
df['Cover_Type'].value_counts()

#%%
"""As we can see that classes are imbanaced and we have good number of observations for cover type 2 and 1. For cover type 4 and 5, we have less number of observations."""

# %%
len(df)

# %%
X = df.drop('Cover_Type', axis = 1)
y = df['Cover_Type']


# %%
"""Checking null values"""

df.isnull().sum()

"""There is no missing values in the dataset"""

#%%
"""
EDA(Univariate)
"""
# %%
df_new = df.filter(['Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'], axis=1)

df_new.head(2)

# %%
df_new.describe()

#########################EDA on numerical features, Hill shade,horizontal distance to roadways and fire points#################
# %%
eda.plot_violin(df_new)
eda.plot_violin_2(df_new)

#####################EDA categorical feature########################
#%%
eda.wilderness_area(df)


#####################Soil order type mapping###########################
# %%

"""**Soil type mapping**

Based on the soil hierarchy, we can classify the the all 40 soil types to main order in soil hierarchy.
Based on that, we are able to classify the features into 7 different soil order types.
"""
# %%
df = FE.soil_order_mapping(df)
#this function will take more time as it is mapping orders to 5.5 lacs rows

# %%
df.head()

#######################important soil features########################3
# %%
FE.soil_importance(df)


#%%
#######preprocessing###########
df = pp.sqrt_transform(df)
df = pp.gaussian_transform(df)
df = pp.scaling_features(df)

#%%
df.head()


#############Baseline Model###############
# %%
bm.baseline_model(df)


#####################feature Reduction#################
# %%
RFE.select_best_feature_RFE(df)

##########################data split#######################

# %%
x_train,y_train,x_cv,y_cv,x_test,y_test = tts.data_division(df)

#######################SMOTE###########################

#%%
X_train_smote, Y_train_smote = SMO.Oversampling(x_train,y_train)

#################Random Forest Model##############################
#%%
#hyper parameter tuning based on depth
mb.random_forest_model_tuning_depth(X_train_smote, Y_train_smote,x_train,y_train,x_cv,y_cv,x_test,y_test)


#%%
#hyper parameter tuning based on estimator
mb.random_forest_model_tuning_estimator(X_train_smote, Y_train_smote,x_train,y_train,x_cv,y_cv,x_test,y_test)


#%%
#best RF model
mb.best_random_forest_model(X_train_smote, Y_train_smote,x_train,y_train,x_cv,y_cv,x_test,y_test)


# %%

#%%
mb.xgboost_model_tuning_depth(X_train_smote, Y_train_smote, x_train, y_train, x_cv, y_cv, x_test, y_test)

#%%

#%%
mb.xgboost_model_tuning_estimator(X_train_smote, Y_train_smote, x_train, y_train, x_cv, y_cv, x_test, y_test)

#%%

#%%
mb.best_xgboost_model(X_train_smote, Y_train_smote, x_train, y_train, x_cv, y_cv, x_test, y_test)

#%%