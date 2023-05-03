
import pandas as pd
import matplotlib.pyplot as plt

def soil_importance(df):
    """2. Finding most impacting feature:
    Some of the soil types are contributing more in some of the forest cover types.
    Those features are really strong to classify the cover types.
    """

    

    soil_contri = pd.DataFrame(data = {'Cover_Type':[1,2,3,4,5,6,7]})


    #df[df['Soil_Type2']==1].groupby(['Cover_Type'])['Soil_Type2'].count().rename("Soil_Type2").transform(lambda x: (x/x.sum())*100).reset_index()

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
        soil_contri = soil_contri.merge(df[df[i]==1].groupby(['Cover_Type'])[i].count().rename(i).reset_index(), on='Cover_Type', how='left')


    """category wise soil distribution"""

    df['Cover_Type'].value_counts().reset_index().rename(columns = {'index':'Cover_Type','Cover_Type':'Total_count'})

    soil_contri = soil_contri.merge(df['Cover_Type'].value_counts().reset_index().rename(columns = {'index':'Cover_Type','Cover_Type':'Total_count'}), on='Cover_Type', how='left')



    for i in col:
        soil_contri[i] = my_func(soil_contri,i)


    plt.figure(figsize=(50,50))
    k = soil_contri.drop(['Total_count'], axis = 1).plot(x='Cover_Type', kind='bar', stacked=True,title='Stacked Bar Graph by dataframe')
    plt.show()

    maxValues = soil_contri.drop(['Cover_Type','Total_count'],axis = 1).max(axis=1)
    return maxValues




"""**Using the soil order mapping, classifying 40 soil types into 7 main soil orders.**"""
def soil_order_mapping(df):

    df[['Inceptisols','Mollisols','Spodosols','Alfisols','Entisols','Unknown','Histosols']] = 0
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

    return df


def my_func(x,a):
    try:
        return ((x[a]) / (x['Total_count']))* 100
    except (ZeroDivisionError, ValueError):
        return 0
    

def Yash():
    print('Yash')