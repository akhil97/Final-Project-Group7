from sklearn.model_selection import train_test_split

def split_data(df,y,test_size):
  xtrain,x_test,y_train,y_test = train_test_split(df.drop(y,axis = 1),df[y],test_size = test_size)
  return xtrain,x_test,y_train,y_test


def data_division(df):
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

    x_train,x_test,y_train,y_test = split_data(top_30_features,'Cover_Type',0.3)

    x_train,x_cv,y_train,y_cv = train_test_split(x_train,y_train,test_size = 0.2)

    return x_train,y_train,x_cv,y_cv,x_test,y_test