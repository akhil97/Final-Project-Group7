import matplotlib.pyplot as plt
import pandas as pd

def plot_violin(df_new):
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

def my_func(x,a):
    try:
        return ((x[a]) / (x['Total_count']))* 100
    except (ZeroDivisionError, ValueError):
        return 0


def plot_violin_2(df_new):
    fig,ax = plt.subplots(1,2,figsize = (10,5))

    ax[0].violinplot(df_new['Horizontal_Distance_To_Roadways'])
    ax[0].title.set_text('Violin plot for Horizontal_Distance_To_Roadways')
    ax[0].set_ylabel('Horizontal_Distance_To_Roadways')
    ax[1].violinplot(df_new['Horizontal_Distance_To_Fire_Points'])
    ax[1].title.set_text('Violin plot for Horizontal_Distance_To_Fire_Points')
    ax[1].set_ylabel('Horizontal_Distance_To_Fire_Points')

def wilderness_area(df):
    wild_area = pd.DataFrame(data = {'Cover_Type':[1,2,3,4,5,6,7]})
    df[df['Wilderness_Area1'] == 1].groupby(['Cover_Type'])['Wilderness_Area1'].count().rename("Wilderness_Area1")
    col = ['Wilderness_Area1','Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']

    for i in col:
        wild_area = wild_area.merge(df[df[i]==1].groupby(['Cover_Type'])[i].count().rename(i).reset_index(), on='Cover_Type', how='left')

    total_obs = df.Cover_Type.value_counts().reset_index().rename(columns = {'index':'Cover_Type','Cover_Type':'Total_count'})
    wild_area = wild_area.merge(total_obs, on='Cover_Type', how='left')
    col = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']
    for i in col:
        wild_area[i] = my_func(wild_area,i)
    wild_area.drop('Total_count',axis = 1,inplace = True)
    cover_types = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
    ax = wild_area.plot(x='Cover_Type', kind='bar', stacked=True,title='Stacked Bar For wilderness area',figsize=(30, 10))
    ax.legend(['Rawah','Neota','Comanche Peak','Cache la Poudre​'])
    ax.set_xticklabels(cover_types)