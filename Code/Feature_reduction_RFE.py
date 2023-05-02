from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def split_data(df,y,test_size):
  xtrain,x_test,y_train,y_test = train_test_split(df.drop(y,axis = 1),df[y],test_size = test_size)
  return xtrain,x_test,y_train,y_test

def select_best_feature_RFE(df):
    all_features = df.drop('Aspect',axis = 1)
    x_train,x_test,y_train,y_test = split_data(all_features,'Cover_Type',0.3)

    tf = RFE(RandomForestClassifier(), n_features_to_select=30, verbose=1)
    Xt = tf.fit_transform(x_train,y_train)
    print("Shape =", Xt.shape)

    return tf.get_feature_names_out()

