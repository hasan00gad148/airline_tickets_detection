import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pickle
from main import preproc
# from preprocessing import *

# dataSet = getCleanData()

# dataSet.drop(["route"],axis=1,inplace=True)


# dataSet.to_csv("C:\\Users\\Hassan\\Desktop\\mlPro\\data.csv")
dataSet = preproc("airline-price-classification.csv")
#dataSet = pd.read_csv("data.csv")
tmp = dataSet["TicketCategory"]
dataSet.drop(['TicketCategory'],axis=1,inplace=True)
dataSet["TicketCategory"] = tmp
def featureNormalization(X:pd.DataFrame):
    for c in range(X.shape[1]):
        m=X[:,c].mean()
        std=X[:,c].std()
        # max = (X[c]).max()
        # min = (X[c]).min()
        if std == 0:
            std = 1
        X[:,c] =  (X[:,c]-m)/(std)
 


print(dataSet.columns)
#------------------------------------------------------------------------Index(['airline', 'ch_code', 'num_code', 'stop', 'type', 'time_taken_minutes',

corr = dataSet.corr()
    #Top n% Correlation training features with the Value
corrPercent=0.2
top_feature = corr.index[abs(corr['TicketCategory'])>corrPercent]
top_feature = top_feature.delete(-1)

X = dataSet[top_feature]
Y=dataSet["TicketCategory"] #Label
#------------------------------------------------------------------------

X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)
    
poly_features = PolynomialFeatures(degree=3, include_bias=True)
#-----------------------------------------------------------------------------------------



X_train_poly = poly_features.fit_transform(X_train)
featureNormalization(X_train_poly)
# X_train_poly = handleData(X_train_poly)

#----------------------
m = X_train.shape[0]

logistic =LogisticRegression(multi_class="ovr",C=5).fit(X_train_poly, y_train)
#svc = OneVsRestClassifier(lin_svc).partial_fit(X_train[0:int(m*0.02),:], y_train[0:int(m*0.02),:],np.unique(y_train))
#----------------------


predictionsTrain = logistic.predict(X_train_poly)
accuracy1 = np.mean(predictionsTrain == y_train)
print("\ntrainingAccuracy " + str(accuracy1)+"\n")
#------------------------------------------------------------------------------
X_test_poly = poly_features.fit_transform(X_test)
featureNormalization(X_test_poly)
# X_test_poly = handleData(X_test_poly)

predictionsTest = logistic.predict(X_test_poly)
accuracy2 = np.mean(predictionsTest == y_test)


print("\ntestingAccuracy " + str(accuracy2)+"\n")

pickle.dump(logistic, open('modelLOGISTIC.pkl', 'wb'))
pickle.dump(top_feature, open('top_feature.pkl', 'wb'))