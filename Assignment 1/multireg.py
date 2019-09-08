# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import copy

# def center_assignment(df, centers):
#     for i in centers.keys():
#         df['distance_from_{}'.format(i)] = (
#             np.sqrt(
#                 (df['x'] - centers[i][0]) ** 2
#                 + (df['y'] - centers[i][1]) ** 2
#             )
#         )
#     centroid_distance_cols = ['distance_from_{}'.format(i) for i in centers.keys()]
#     df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
#     df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
#     df['color'] = df['closest'].map(lambda x: colmap[x])
#     return df

# def update_center(center):
#     for i in centers.keys():
#         centers[i][0] = np.mean(df[df['closest'] == i]['x'])
#         centers[i][1] = np.mean(df[df['closest'] == i]['y'])
#     return center


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
#      for i in range(len(c))
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