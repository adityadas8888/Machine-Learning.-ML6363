import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os

def read_replace():
    dir_path = os.path.dirname(os.path.realpath(__file__));
    data = pd.read_csv(dir_path+"/IRIS.csv");
    data.species.replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1, 2, 3], inplace=True);
    return data;


def kfold(data,k=3):
    chunks = chunkify(data, k);
    acc = [];
    for i in range(k):
        test_data = chunks[i];
        train_data = data.drop(test_data.index);
        betacap = train_model(train_data);
        accuracy = test_model(betacap,test_data);
        acc.append(accuracy);
    print('Accuracy for a multivariate regression with {} folds is {}'.format(k,np.mean(acc))); 

def train_model(train_data):
    train_data = train_data.values;
    betacap = np.linalg.inv((train_data.transpose().dot(train_data))).dot(train_data.transpose().dot(train_data.species));
    return betacap;

def test_model(betacap,test_data):
    


    # train_set = data.sample(frac=0.75, random_state=0)
    # test_set = data.drop(train_set.index).sample(frac =1.0);
    # data2 = data.drop('species',axis = 1);
    # data2= data2.values;
    # betacap = np.linalg.inv((data2.transpose().dot(data2))).dot(data2.transpose().dot(data.species));
    # # print(betacap);
    # print(data);
    
def chunkify(lst,n):
    return [ lst[i::n] for i in range(n) ]
# def kfold(k=10):
#     bins = []
#     a = []
#     #random.shuffle(data)
#     k = len(data)//7
#     for i in range(0, len(data)):
#         a.append(data[i])
#         if len(a) % k == 0:
#             bins.append(a)
#             a = []
#     bins.append(a)
#     efficiency = 0
#     for i in range(0, len(bins)):
#         testing_set = bins[i]
#         training_set = []
#         for j in range(0, len(bins)):
#             if i is not j:
#                 for b in bins[j]:
#                     training_set.append(b)
#         tree = id3(training_set, target_attribute, attributes)
#         acccr = (accuracy(testing_set, tree)[1])
#         efficiency += acccr
#         print("Bin", i, " accuracy =", acccr)
#     print("Average accuracy =", efficiency / len(bins))

def main():
  data = read_replace();
  kfold(data,k=10);  
if __name__== "__main__":
  main()