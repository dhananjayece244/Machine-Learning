
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import sys,os,csv,glob


def read_train_files():
    train = pd.read_csv('trainSold.csv')
    return train
    
def pre_processing(train):
    del train['Id']
    labels = train['SaleStatus']
    del train['SaleStatus']
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels=le.transform(labels)
    
    train=train.fillna(train.median())
    train=pd.get_dummies(train)

    for column in train.columns[0:]:
          train[column] = (train[column] - train[column].mean()) / (train[column].max() - train[column].min())
    
    return (train,labels)     

def split_train_test(train,labels):
    X_train, X_test, Y_train, Y_test = train_test_split(train,labels, test_size=0.2,random_state=32)
    return (X_train, X_test, Y_train, Y_test)
    
def SVM_model(X_train, X_test, Y_train, Y_test):
    
    #C parameter 
    parameter_C=[]
    parameter_gamma=[]
    acc=[]

    for c in range(80,200,10):
        clf = svm.SVC(C= c, gamma=0.01)
        clf.fit(X_train,Y_train)
        pred_label=clf.predict(X_test)
        accuracy=accuracy_score(Y_test, pred_label)
        parameter_C.append(c)
        acc.append(accuracy)

    #line graph 
    plt.plot(parameter_C, acc, label='Parameter C vs Accuracy in SVM')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Parameter C vs Accuracy in SVM')
    plt.savefig('svm_C.jpeg')
  

    #gamma
    acc1=[]
    G=[0.000001, 0.00001,0.0001,0.001, 0.01,0.1]
    for g in G:
        clf = svm.SVC(gamma = g, C=160)
        clf.fit(X_train,Y_train)
        pred_label=clf.predict(X_test)
        accuracy=accuracy_score(Y_test, pred_label)
        parameter_gamma.append(g)
        acc1.append(accuracy)

    #line graph 
    plt.plot(parameter_gamma, acc1, label='gamma vs Accuracy in SVM')
    plt.xlabel('gamma')
    plt.ylabel('Accuracy')
    plt.title('gamma vs Accuracy in SVM')
    plt.savefig('SVM_gamma.jpeg')
  
    
    clf = svm.SVC(gamma = 0.006, C= 501)
    clf.fit(X_train,Y_train)
    pred_label=clf.predict(X_test)
    accuracy=accuracy_score(Y_test, pred_label)
    print("accuracy ",accuracy,'\n')

    #make pickle file for SVM model.
    list_pickle_path = 'svm_pickle.pkl'
    # Create an variable to pickle and open it in write mode
    list_pickle = open(list_pickle_path, 'wb')
    pickle.dump(clf, list_pickle)
    list_pickle.close()

    
def gradient_boosting_model(X_train, X_test, Y_train, Y_test):
    
    
    #n_estimator tuning
    n_estimator=[]
    acc=[]

    for n_estm in range(50,500,50):
        clf_g = ensemble.GradientBoostingClassifier(n_estimators=n_estm,learning_rate=0.1)
        clf_g.fit(X_train,Y_train)
        pred_grd= clf_g.predict(X_test)
        accuracy=accuracy_score(Y_test, pred_grd)
        n_estimator.append(n_estm)
        acc.append(accuracy)

    #line graph 
    plt.plot(n_estimator, acc)
    plt.xlabel('n_estimator')
    plt.ylabel('Accuracy')
    plt.title('n_estimator vs Accuracy in Gradient boosting')
    plt.savefig('grd_boosting_n_estimator.jpeg')
   
    
    #learning rate
    learning_rate=[]
    acc1=[]
    learn_rt=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    for l_rt in learn_rt:
        clf_g = ensemble.GradientBoostingClassifier(n_estimators=150,learning_rate=l_rt)
        clf_g.fit(X_train,Y_train)
        pred_grd= clf_g.predict(X_test)
        accuracy=accuracy_score(Y_test, pred_grd)
        learning_rate.append(l_rt)
        acc1.append(accuracy)

    #line graph 
    plt.plot(learning_rate, acc1)
    plt.xlabel('learning rate')
    plt.ylabel('Accuracy')
    plt.title('learning-rate vs Accuracy in Gradient boosting')

    plt.savefig('grd_boosting_learning_rate.jpeg')
    
    
    
    parameter={'n_estimators':150,'max_depth': 3, 'subsample': 1.0,
          'learning_rate': 0.15,'min_samples_leaf': 1, 'random_state': 3}
    clf_g = ensemble.GradientBoostingClassifier(**parameter)
    clf_g.fit(X_train,Y_train)
    pred_grd= clf_g.predict(X_test)
    accuracy=accuracy_score(Y_test, pred_grd)
    print("accuracy ", accuracy,'\n')
    
    #make pickle file
    list_pickle_path = 'grad_boosting_pickle.pkl'
    # Create an variable to pickle and open it in write mode
    list_pickle = open(list_pickle_path, 'wb')
    pickle.dump(clf_g, list_pickle)
    list_pickle.close()
    
    
def decision_tree(X_train, X_test, Y_train, Y_test):
    clf_dtree = tree.DecisionTreeClassifier(random_state=2)
    clf_dtree.fit(X_train,Y_train)
    clf_dtree = clf_dtree.fit(X_train,Y_train)
    pred_dtree= clf_dtree.predict(X_test)
    accuracy=accuracy_score(Y_test, pred_dtree)
    print('accuracy',accuracy,'\n' )
    
    #make pickle file
    list_pickle_path = 'decision_tree_pickle.pkl'
    # Create an variable to pickle and open it in write mode
    list_pickle = open(list_pickle_path, 'wb')
    pickle.dump(clf_dtree, list_pickle)
    list_pickle.close()
    
    
    
def random_forest(X_train, X_test, Y_train, Y_test):
    
    #n_estimator
    n_estimator=[]
    acc=[]
    for n_estm in range(1,25,2):
        clf_R = RandomForestClassifier(max_depth=14, n_estimators=n_estm)
        clf_R.fit(X_train,Y_train)
        pred_R= clf_R.predict(X_test)
        accuracy=accuracy_score(Y_test, pred_R)
        n_estimator.append(n_estm)
        acc.append(accuracy)

    #line graph 
    plt.plot(n_estimator, acc)
    plt.xlabel('n_estimator')
    plt.ylabel('Accuracy')
    plt.title('n_estimator vs Accuracy in Random Forest')
    plt.savefig('Random_forest_n_estimator.jpeg')
  
    
    #maximum depth
    max_depth=[]
    acc1=[]
    for md in range(1,25,1):
        clf_R = RandomForestClassifier(max_depth=md, n_estimators=14)
        clf_R.fit(X_train,Y_train)
        pred_R= clf_R.predict(X_test)
        accuracy=accuracy_score(Y_test, pred_R)
        max_depth.append(md)
        acc1.append(accuracy)

    #line graph 
    plt.plot(max_depth, acc1)
    plt.xlabel('max-depth')
    plt.ylabel('Accuracy')
    plt.title('max-depth vs Accuracy in Random Forest')

    plt.savefig('Random_forest_max_depth.jpeg')
    

    
    
    clf_R = RandomForestClassifier(max_depth=14, n_estimators=14)
    clf_R.fit(X_train,Y_train)
    pred_R= clf_R.predict(X_test)
    accuracy=accuracy_score(Y_test, pred_R)
    print('accuracy ',accuracy,'\n')
    
    #make pickle file
    list_pickle_path = 'random_forest_pickle.pkl'
    # Create an variable to pickle and open it in write mode
    list_pickle = open(list_pickle_path, 'wb')
    pickle.dump(clf_R, list_pickle)
    list_pickle.close()
    
    

    
    
if __name__ == '__main__':
    train=read_train_files()
    train,labels=pre_processing(train)
    X_train, X_test, Y_train, Y_test=split_train_test(train,labels)
    
    print("SVM model---------------------\n")
    SVM_model(X_train, X_test, Y_train, Y_test)
    
    print("Gradient boositing-------------------\n")
    gradient_boosting_model(X_train, X_test, Y_train, Y_test)
    
    print("Random Forest ---------------------\n")
    random_forest(X_train, X_test, Y_train, Y_test)
    
    print("Decision Tree ----------------------\n")
    decision_tree(X_train, X_test, Y_train, Y_test)
    
    
    
    
  





