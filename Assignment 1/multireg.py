import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#read_replace function. This function is used to read the dataset from the folder and replace the Nominal values with some quantitative values.
def read_replace():
    dir_path = os.path.dirname(os.path.realpath(__file__));                 
    data = pd.read_csv(dir_path+"/IRIS.csv");                                                          # reads the dataset from the current working directory     
    data.species.replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1, 2, 3], inplace=True);  # replacing the nominal values
    data = data.sample(frac=1).reset_index(drop=True);                                                  # randomizing the dataset and resetting the index
    return data;

#kfold function. This function is used to implement the k-fold cross validation. It calls the train_model and test_model functions, finds the average of the accuracies and prints it to the user.
def kfold(data,k):
    if k>1 and k<len(data):                                                                             # checking if k is valid, else exit
        chunks = chunkify(data, k);                                                                     # dividing the dataset into chunks    
        acc = [];
        for i in range(k):
            test_data = chunks[i];
            train_data = data.drop(test_data.index);                                                    # separating the test and training dataset
            betacap = train_model(train_data);  
            accuracy = test_model(betacap,test_data);
            acc.append(accuracy);
        print('Accuracy for a multivariate regression with {} folds is {}%'.format(k,(np.mean(acc)*100))); 
    else:
        print('The entered k value is invalid. Exiting!!!');
        exit;
# train_model function. This function is used to train the model. It returns the coeffecients of the trained multivariate regression model.
def train_model(train_data):
    data = train_data
    train_data = data.drop('species',axis = 1);
    train_data= train_data.values;
    betacap = np.linalg.inv((train_data.transpose().dot(train_data))).dot(train_data.transpose().dot(data.species));             # beta = inverse((transp(X).X)).(transp(X).Y)
    return betacap;

#test_model function. This function is used to check the accuracy of the trained model. 
def test_model(betacap,test_data):
    foo=[];
    predicted =(test_data.iloc[:,0:4].values*betacap);                                                                          # multiplying the coeffecients with the test data to get the predicted values
    for i in range(len(predicted)):
        foo.append(round(sum(predicted[i])));
    test_data.loc[:,'predicted']=foo;
    test_data.loc[:,'correct'] = np.where(test_data['species']==test_data['predicted'], 1, 0);
    accuracy=test_data['correct'].sum()/len(test_data);
    return accuracy;

#chunkify function. Divides a particular pandas dataframe into chunks of n.
def chunkify(lst,n):
    return [ lst[i::n] for i in range(n) ]

# main function. You can change the k value to change the k fold cross validation
def main():
  data = read_replace();
  kfold(data,k=5);  
if __name__== "__main__":
  main()
