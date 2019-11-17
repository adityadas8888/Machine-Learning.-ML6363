import numpy as np
import pandas as pd
import copy
import math
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def read_replace():
    dir_path = os.path.dirname(os.path.realpath(__file__));                 
    data = pd.read_csv(dir_path+"/IRIS.csv");                                                           # reads the dataset from the current working directory  
    data.species.replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1, 2, 3], inplace=True);  # replacing the nominal values
    data = data.sample(frac=1).reset_index(drop=True)
    return data                                                  # randomizing the dataset and resetting the index

def set_random_centers(data):
    rndc = data.sample(n=len(data.species.unique()))
    centers = {
     i+1: [float(rndc.sepal_length.iloc[i]),float(rndc.sepal_width.iloc[i]),float(rndc.petal_length.iloc[i]),float(rndc.petal_width.iloc[i])]
     for i in range(len(data.species.unique()))                                                 # the number of labels of the flowers has to be changed here.
 }
    return centers

def center_assignment(df, centers):
    colmap = {1: 'r', 2: 'g', 3: 'b',4:'y',5:'pink',6:'brown'}
    for i in centers.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['sepal_length'] - centers[i][0]) ** 2                              ## the dimensions of the iris data has to be put here
                + (df['sepal_width'] - centers[i][1]) ** 2
                + (df['petal_length'] - centers[i][2]) ** 2                              ## the dimensions of the iris data has to be put here
                + (df['petal_width'] - centers[i][3]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centers.keys()]
    df['predicted_cluster'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['predicted_cluster'] = df['predicted_cluster'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['predicted_cluster'].map(lambda x: colmap[x])
    return df

def kmeans(data,center):
    data = center_assignment(data,center)                                   ## Checked, this is working fine for the first iteration
    iterations=0
    norm = True
    while iterations<1000 and norm:
        old_centers = copy.deepcopy(center)                                     ## deep copy of the old centers
        center = update_center(center,data)
        data = center_assignment(data,center)  
        dw=dx=dy=dz=0.0                                   ## new initialized centers
        flag =0
        for i in old_centers.keys():
            old_w = old_centers[i][0]
            old_x = old_centers[i][1]
            old_y = old_centers[i][2]
            old_z = old_centers[i][3]
            dw = (center[i][0] -old_w)**2
            dx = (center[i][1] - old_x)**2
            dy = (center[i][2] - old_y)**2
            dz = (center[i][3] - old_z)**2
            temp = np.sqrt(dw+dx+dy+dz)
            if(temp<0.001):
                flag+=1
        if flag ==3:
            norm =False
        iterations+=1
    print('centers', center)
    print('iterations    :', iterations)           
    return(data,center)

def update_center(center,data):
    for i in center.keys():
        center[i][0] = data.loc[(data['predicted_cluster'] == i),'sepal_length'].mean()
        center[i][1] = data.loc[(data['predicted_cluster'] == i),'sepal_width'].mean()
        center[i][2] = data.loc[(data['predicted_cluster'] == i),'petal_length'].mean()
        center[i][3] = data.loc[(data['predicted_cluster'] == i),'petal_width'].mean()
    return center

def main():
  
  data = read_replace();  # reading function
  center = set_random_centers(data)                                 ## this is working fine
  data,center = kmeans(data,center)
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')

  x = data['sepal_length'],
  y = data['sepal_width'],
  z = data['petal_length'],
  c = data['color']
  img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
  for i in center.keys():
    x = float(center[i][0])
    y = float(center[i][1])
    z = float(center[i][2])
    c = int(center[i][3])
    ax.scatter(x,y,z, color='black')
  plt.show()

if __name__== "__main__":
  main()