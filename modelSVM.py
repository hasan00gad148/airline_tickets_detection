import pickle

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from main import preproc
#dataSet = pd.read_csv("data.csv")
dataSet = preproc("airline-price-classification.csv")

tmp = dataSet["TicketCategory"]
dataSet.drop(['TicketCategory'],axis=1,inplace=True)
dataSet["TicketCategory"] = tmp


def featureNormalization(X: pd.DataFrame):
    for c in X.columns:
        m = X[c].mean()
        std = X[c].std()
        # max = (X[c]).max()
        # min = (X[c]).min()
        if std == 0:
            std = 1
        X[c] = (X[c] - m) / (std)

# def handleData(X:np.ndarray):
#     m = X.shape[0]
#     X = np.hstack((X,np.ones((m,1))))  
#     tmp = X[:,0].copy()
#     X[:,0] =  X[:,-1]
#     X[:,-1] = tmp

#     return X

#print(dataSet.columns)
# ------------------------------------------------------------------------
corr = dataSet.corr()
# Top n% Correlation training features with the Value
corrPercent = 0.2
top_feature = corr.index[abs(corr['TicketCategory']) > corrPercent]
top_feature = top_feature.delete(-1)

X = dataSet[top_feature]
Y = dataSet["TicketCategory"]  # Label
# ------------------------------------------------------------------------

X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)
# -----------------------------------------------------------------------------------------
print("\n" + str(X_train.shape) + "\n")
print(str(type(X_train)))
featureNormalization(X_train)

# ----------------------
m = X_train.shape[0]
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.9, C=0.01)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=0.01)
lin_svc = svm.LinearSVC(C=0.1, multi_class="ovr")
svc = lin_svc.fit(X_train, y_train)
# svc = OneVsRestClassifier(lin_svc).partial_fit(X_train[0:int(m*0.02),:], y_train[0:int(m*0.02),:],np.unique(y_train))
# ----------------------

predictionsTrain = svc.predict(X_train)
accuracy1 = np.mean(predictionsTrain == y_train)
print("\ntrainingAccuracy " + str(accuracy1) + "\n")
# ------------------------------------------------------------------------------

featureNormalization(X_test)
predictionsTest = svc.predict(X_test)
accuracy2 = np.mean(predictionsTest == y_test)

print("\ntestingAccuracy " + str(accuracy2) + "\n")
pickle.dump(svc, open('modelSVM.pkl', 'wb'))
pickle.dump(top_feature, open('top_feature.pkl', 'wb'))