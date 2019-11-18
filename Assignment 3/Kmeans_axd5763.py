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
    data = pd.read_csv(dir_path+"/IRIS.csv");                                                           # reads the dataset csv from the current working directory  
    data.species.replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1, 2, 3], inplace=True); # replacing the nominal values
    data = data.sample(frac=1).reset_index(drop=True)                                                   # randomizing the dataset and resetting the index
    return data                                                  

def set_random_centers(data):                                                                           # randomly selecting k number of centroids. The k depends on the 
    rndc = data.sample(n=len(data.species.unique()))
    centers = {
     i+1: [float(rndc.sepal_length.iloc[i]),float(rndc.sepal_width.iloc[i]),float(rndc.petal_length.iloc[i]),float(rndc.petal_width.iloc[i])]
     for i in range(len(data.species.unique()))                                                         # the number of labels of the flowers has to be changed here.
 }
    return centers

def center_assignment(df, centers):                                                                     # assigning the centers of the centroids
    colmap = {1: 'r', 2: 'g', 3: 'b',4:'y',5:'pink',6:'brown'}
    for i in centers.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['sepal_length'] - centers[i][0]) ** 2                              
                + (df['sepal_width'] - centers[i][1]) ** 2
                + (df['petal_length'] - centers[i][2]) ** 2                              
                + (df['petal_width'] - centers[i][3]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centers.keys()]
    df['predicted_cluster'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['predicted_cluster'] = df['predicted_cluster'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['predicted_cluster'].map(lambda x: colmap[x])
    return df

def kmeans(data,center):                                                                                # kmeans actual function
    data = center_assignment(data,center)                                   
    iterations=0
    norm = True
    j=1
    print ("...............................Clustering the points...............................\n")
    while iterations<1000 and norm:
        old_centers = copy.deepcopy(center)                                     
        center = update_center(center,data)
        data = center_assignment(data,center)  
        dw=dx=dy=dz=0.0                                   
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
    print('No of iterations    : %d\n'% iterations)           
    return(data,center)

def update_center(center,data):                                                                         #updating the centroids
    for i in center.keys():
        center[i][0] = data.loc[(data['predicted_cluster'] == i),'sepal_length'].mean()
        center[i][1] = data.loc[(data['predicted_cluster'] == i),'sepal_width'].mean()
        center[i][2] = data.loc[(data['predicted_cluster'] == i),'petal_length'].mean()
        center[i][3] = data.loc[(data['predicted_cluster'] == i),'petal_width'].mean()
    return center

def plot_clusters(data,center):                                                                         # plotting the clusters
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')
  print('Centroids of the clusters')
  for i in center.keys():
      print('Center %d'% int(i))
      print(center[i],'\n')
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

def main():
  
  data = read_replace();  
  center = set_random_centers(data)                                 
  data,center = kmeans(data,center)
  eval = data.groupby('species')['predicted_cluster'].value_counts()
  maxvals = eval.max(level='species')
  minvals= eval.min(level='species')
  incorrect=0
  for i in range(len(data.species.unique())):
      if maxvals[i+1]!=minvals[i+1]:
          incorrect+=minvals[i+1]
  print('Accuracy is %f percent\n'%(((len(data.index)-incorrect)/len(data.index))*100))
  plot_clusters(data,center)
if __name__== "__main__":
  main()