import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
pd.set_option('display.max_rows', None);

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
        # accuracy = 
        print(i);
        test_model(betacap,test_data);
        # acc.append(accuracy);
    # print('Accuracy for a multivariate regression with {} folds is {}'.format(k,np.mean(acc))); 

def train_model(train_data):
    data = train_data
    train_data = data.drop('species',axis = 1);
    train_data= train_data.values;
    betacap = np.linalg.inv((train_data.transpose().dot(train_data))).dot(train_data.transpose().dot(data.species));
    return betacap;

def test_model(betacap,test_data):
    foo=[];
    predicted =(test_data.iloc[:,0:4].values*betacap);
    for i in range(len(predicted)):
        foo.append(round(sum(predicted[i])));
    test_data['predicted']=foo;
    test_data['correct'] = np.where(test_data['species']==test_data['predicted'], 1, 0);
    accuracy=test_data['correct'].sum()/len(test_data);
    return accuracy;

def chunkify(lst,n):
    return [ lst[i::n] for i in range(n) ]

def main():
  data = read_replace();
  kfold(data,k=15);  
if __name__== "__main__":
  main()

















   # train_set = data.sample(frac=0.75, random_state=0)
    # test_set = data.drop(train_set.index).sample(frac =1.0);
    # 
   