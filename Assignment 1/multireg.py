import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os

def read_rand():
    dir_path = os.path.dirname(os.path.realpath(__file__));
    data = pd.read_csv(dir_path+"/IRIS.csv");
    data.species.replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1, 2, 3], inplace=True);
    train_set = data.sample(frac=0.75, random_state=0)
    test_set = data.drop(train_set.index).sample(frac =1.0);
    data2 = data.drop('species',axis = 1);
    data2= data2.values;
    betacap = np.linalg.inv((data2.transpose().dot(data2))).dot(data2.transpose().dot(data.species));
    print(betacap);
    print(data);
read_rand();