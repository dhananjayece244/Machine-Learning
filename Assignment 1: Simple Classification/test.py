
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import sys,csv

def read_test_file():
    test = pd.read_csv('testSold.csv')
    return test
def read_gt_file():
    gt = pd.read_csv('gt.csv')
    #print(gt.iloc[:,3])
    return (gt.iloc[:,3])
    
def pre_processing_test_data(test):
    del test['Id']
   

    train = pd.read_csv('trainSold.csv')
    del train['Id']
    train=train.fillna(train.median())
    labels = train['SaleStatus']
    del train['SaleStatus']
    
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels=le.transform(labels)
    
    train=pd.get_dummies(train) 
    for column in train.columns[0:]:
          train[column] = (train[column] - train[column].mean()) / (train[column].max() - train[column].min())
    
    
    test=test.fillna(test.median())    
    test=pd.get_dummies(test)
       
    for column in test.columns[0:]:
          test[column] = (test[column] - test[column].mean()) / (test[column].max() - test[column].min())
             
    for column in list(train.columns.values):
        if(column not in list(test.columns.values)):
            test[column]=0
            test[column]=test[column].astype(np.uint8)
    
    return (test,le)

def predict_output(test,le,labels):
    
    
    
    try:
        pickle_path = 'grad_boosting_pickle.pkl'
        unpickle = open(pickle_path, 'rb')
    except:
        pickle_path = 'finalModel1.pkl'
        unpickle = open(pickle_path, 'rb')
    
    grd_bst = pickle.load(unpickle)
    pred_grd = grd_bst.predict(test)
    pred_grd =le.inverse_transform(pred_grd)
    accuracy_grd=accuracy_score(pred_grd,labels)
    print('accuracy using gradient boosting model:',accuracy_grd)
    
    pred=pred_grd
    accuracy=accuracy_grd
    

    try:
        pickle_path = 'random_forest_pickle.pkl'
        unpickle = open(pickle_path, 'rb')
    except:
        pickle_path = 'finalModel2.pkl'
        unpickle = open(pickle_path, 'rb')
    
    
    grd_rdf = pickle.load(unpickle)
    pred_rdf = grd_rdf.predict(test)
    pred_rdf =le.inverse_transform(pred_rdf)
    accuracy_rdf=accuracy_score(pred_rdf,labels)
    print('accuracy using random model           :',accuracy_rdf)
    if(accuracy<accuracy_rdf):
        accuracy=accuracy_rdf
        pred=pred_rdf
    
    
    return (pred)
    
def out_csv(pred):
    file = 'gt.csv'
    out = pd.read_csv(file)
    
    df = pd.DataFrame(data={"Id": out['Id'], "SaleStatus":pred })
    df.to_csv("./out.csv", sep=',',index=False)

if __name__ == '__main__':
    test=read_test_file()
    
    out_labels=read_gt_file()    
    test,le=pre_processing_test_data(test)
    pred=predict_output(test,le,out_labels)
    
    out_csv(pred)
   
   

