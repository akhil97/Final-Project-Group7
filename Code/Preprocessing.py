from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def standard_scaling(dfa,column_list):
  for i in column_list:
    tf = StandardScaler()
    dfa[i] = tf.fit_transform(dfa[i].values.reshape(-1,1))
  return dfa

def sqrt_transform(df):
   df['Horizontal_Distance_To_Roadways'] = np.sqrt(df['Horizontal_Distance_To_Roadways'])
   df['Horizontal_Distance_To_Fire_Points'] = np.sqrt(df['Horizontal_Distance_To_Fire_Points'])
   return df


def scaling_features(df):
    df = standard_scaling(df,['Aspect','Aspect_unimodal','Horizontal_Distance_To_Hydrology','Elevation','Slope','Vertical_Distance_To_Hydrology','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'])
    return df
def gaussian_transform(df):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(df['Aspect'].values.reshape(-1, 1))

    # Generate a unimodal distribution using the means and standard deviations of the component distributions
    means, stds = gmm.means_, np.sqrt(gmm.covariances_).reshape(-1)
    unimodal_data = norm.rvs(loc=means.mean(), scale=stds.mean(), size=581012)
    plt.hist(unimodal_data)
    df['Aspect_unimodal'] = unimodal_data
    return df

   