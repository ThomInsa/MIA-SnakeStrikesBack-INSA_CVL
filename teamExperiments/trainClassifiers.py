import numpy as np
from sklearn import model_selection
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import (
    KNeighborsClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
)
import xgboost
from mlxtend.classifier import (
    StackingClassifier
)

import pandas as pd
import datetime as dt

def newClassifiersDatabase():
    classifiersDict = {'Classifier':[], 'Name':[], 'f1':[], 'precision':[], 'confusionMatrix':[], 'AUC':[], 'STD':[]}
    return pd.DataFrame(classifiersDict)

def trainModels(modelsToUse, X_train, y_train):
    classifiersCollection = []
    for model in modelsToUse:
        print("Training model : " + str(model.__name__))
        cls = model()
        cls.fit(X_train.values, y_train.values)
        classifiersCollection.append(cls)
    return classifiersCollection

def trainStackingClassifier(classifiersCollection, X_train, y_train):
    stack = StackingClassifier(
        classifiers=classifiersCollection,
        meta_classifier=LogisticRegression(),
    )
    print("Training model : StackingClassifier")
    stack.fit(X_train.values, y_train.values)
    return stack

def appendStackingClassifier(classifiersCollection, modelsCollection, X_train, y_train):
    stack = StackingClassifier(
        classifiers=classifiersCollection,
        meta_classifier=LogisticRegression(),
    )
    modelsCollection.append(stack)
    stack.fit(X_train.values, y_train.values)
    classifiersCollection.append(stack)
    return classifiersCollection

def createCollection_Predictions(classifiersCollection, X_test):
    predictionsCollection = []
    for model in classifiersCollection:
        prediction = pd.DataFrame(model.predict(X_test))
        prediction.rename(columns={"0": "isMember"})
        predictionsCollection.append(prediction)
    return predictionsCollection

def createCollection_PrecisionScore(y_test, predictionsCollection):
    precisionCollection = []
    index = 0
    for prediction in predictionsCollection:
        precisionCollection.append(precision_score(y_test.values, predictionsCollection[index]))
        index += 1
    return precisionCollection

def createCollection_ROC(classifiersCollection, X, y, X_train, y_train, value = "mean"):
    rocCollection = []
    index = 0
    for classifier in classifiersCollection:
        s = model_selection.cross_val_score(classifier, X, y, scoring='roc_auc', cv=model_selection.KFold(n_splits=10, random_state=42, shuffle=True))
        if value == "mean":
            rocCollection.append(s.mean())
        elif value == "std":
            rocCollection.append(s.std())
        else:
            print("Invalid value, choose mean or std for ROC computation")
            return 1
        index += 1
    return rocCollection

def createCollection_ConfusionMatrices(y_test, predictionsCollection):
    matricesCollection = []
    for prediction in predictionsCollection:
        matrix = confusion_matrix(y_test, prediction)
        matrix = np.round(matrix / len(prediction), decimals=2)
        matricesCollection.append(matrix)
    return matricesCollection

def createClassifiersMetricsDB(classifiersCollection, X, y, precisionsCollection, matricesCollection, ROCMCollection, ROCSCollection):
    classifiersDB = newClassifiersDatabase()
    index = 0
    for classifier in classifiersCollection:
        newRow = (classifier , classifier.__class__.__name__, classifier.score(X,y), precisionsCollection[index], str((matricesCollection[index]).tolist()), ROCMCollection[index], ROCSCollection[index])
        classifiersDB.loc[len(classifiersDB)] = newRow
        index += 1
    return classifiersDB

def saveClassifiersMetricsDB(classifiersDB, directory_path, file_name):
    now = dt.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    new_file_name = file_name + f"classifiersPerfs_{now}.csv"
    file_path = f"{directory_path}{new_file_name}"
    (classifiersDB.drop(columns=['Classifier'])).to_csv(file_path, index=False)