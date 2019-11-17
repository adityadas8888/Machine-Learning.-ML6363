import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import copy
import os
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
    df['predicted'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['predicted'] = df['predicted'].map(lambda x: int(x.lstrip('distance_from_')))
    return df

def kmeans(data,center):
    data = center_assignment(data,center)               ## this is working fine
    dw=dx=dy=dz=1
    iterations=0
    while iterations<1000:# and (dw>=0.001 and dx>= 0.001 and dy>=0.001 and dz>= 0.001):
        closest_centers = data['predicted'].copy(deep=True)                     ## guess this is working
        old_centers = copy.deepcopy(center)                                     ## deep copy of the old centers
        center = update_center(center,data)                                     ## new initialized centers
        for i in old_centers.keys():
            old_w = old_centers[i][0]
            old_x = old_centers[i][1]
            old_y = old_centers[i][2]
            old_z = old_centers[i][3]
            dw = abs(center[i][0] - old_centers[i][0])
            dx = abs(center[i][1] - old_centers[i][1])
            dy = abs(center[i][2] - old_centers[i][2])
            dz = abs(center[i][3] - old_centers[i][3])
        data = center_assignment(data, center)
        print(data)
        iterations+=1
        if closest_centers.equals(data['predicted']):
            return(data)
    return(data)

def update_center(center,data):
    for i in center.keys():
        center[i][0] = np.mean(data[data['predicted'] == i]['sepal_length'])
        center[i][1] = np.mean(data[data['predicted'] == i]['sepal_width'])
        center[i][2] = np.mean(data[data['predicted'] == i]['petal_length'])
        center[i][3] = np.mean(data[data['predicted'] == i]['petal_width'])
    return center


def main():
  data = read_replace();
  print(data.to_string())
  center = set_random_centers(data)                                 ## this is working fine
  data = kmeans(data,center)
  data['correct'] = np.where((data['species'] == data['predicted']) , 1, 0)
  print(data.to_string())
  print(data.correct.sum())
if __name__== "__main__":
  main()































# iter = 1 
# mean1 = [1,0]
# mean2 = [0,1.5]

# Sigma1 = [[0.9,0.4],[0.4,0.9] ]
# Sigma2 = [[0.9,0.4],[0.4,0.9] ]
# #c=[[10,10],[-10,-10]]                                                          #centers for Part 2 of Question 1
# c=[[10,10],[-10,-10],[10,-10],[-10,10]]                                         #centers for Part 3 of Question 1
# #k=2                                                                            # cluster for Part 2 of Question 1
# k=4                                                                             # cluster for Part 3 of Question 1
# colmap = {1: 'r', 2: 'g', 3: 'b',4:'y',5:'pink',6:'brown'}

# x, y = np.random.multivariate_normal(mean1, Sigma1, 500).T
# p, q = np.random.multivariate_normal(mean2, Sigma2, 500).T

# x=np.concatenate((x, p))
# y=np.concatenate((y, q))

# d = {
#     'x': x,
#     'y': y
# }
# df=pd.DataFrame(d)

# centers = {
#      i+1: [c[i][0],c[i][1]]
#      for i in range(len(c))                                                 # the number of labels of the flowers has to be changed here.
#  }

# df = center_assignment(df, centers)
# dx=1
# dy=1

# while iter<=10000 or (dx>=0.001 and dy>= 0.001):
   
#     closest_centers = df['closest'].copy(deep=True)
#     old_centers = copy.deepcopy(centers)
   
#     for i in old_centers.keys():
#         old_x = old_centers[i][0]
#         old_y = old_centers[i][1]
#         dx = (centers[i][0] - old_centers[i][0])
#         dy = (centers[i][1] - old_centers[i][1])

#     centers = update_center(centers)
#     df = center_assignment(df, centers)
#     iter=iter+1
#     if closest_centers.equals(df['closest']):
#         break

# fig = plt.figure(figsize=(10, 10))
# plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k',marker="x")
# for i in centers.keys():
#     plt.scatter(*centers[i], color='black')
# plt.xlim(-3, 5)
# plt.ylim(-3, 5)
# print(centers)
# print(iter)
# print(df)
# plt.show()
