import numpy as np
import pandas as pd
import math
import threading
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pickle
from main import preproc
# from preprocessing import *

# dataSet = preproc("airline-price-classification.csv")
dataSet = pd.read_csv("data.csv")
#dataSet = preproc("airline-price-classification.csv")
tmp = dataSet["TicketCategory"]
dataSet.drop(['TicketCategory'],axis=1,inplace=True)
dataSet["TicketCategory"] = tmp

# dataSet.to_csv("C:\\Users\\Hassan\\Desktop\\mlPro\\data.csv")



def featureNormalization(X: pd.DataFrame):
    for c in range(X.shape[1]):
        m = X[:, c].mean()
        std = X[:, c].std()
        # max = (X[c]).max()
        # min = (X[c]).min()
        if std == 0:
            std = 1
        X[:, c] = (X[:, c] - m) / (std)


def handleData(X: np.ndarray):
    m = X.shape[0]
    X = np.hstack((X, np.ones((m, 1))))
    tmp = X[:, 0].copy()
    X[:, 0] = X[:, -1]
    X[:, -1] = tmp

    return X


def sigmoid(Z: np.ndarray) -> np.ndarray:
    sig = np.zeros((Z.shape[0], 1))
    for i in range(0, Z.shape[0]):
        sig[i] = 1 / (1 + math.exp(-Z[i]))
    return sig


def sigmoid2(Z: np.ndarray) -> np.ndarray:
    sig = np.zeros((Z.shape))
    for i in range(0, Z.shape[0]):
        for j in range(0, Z.shape[1]):
            sig[i, j] = 1 / (1 + math.exp(-Z[i, j]))
    return sig


theatas = list()


def minFunc(X: np.ndarray, Y: np.ndarray, theata: np.ndarray, epochs, a, l2, m, i):
    for i in range(0, epochs):
        z = np.dot(X, theata)
        hx = sigmoid(z)
        e = np.subtract(hx, Y)
        thetaReg = theata.copy()
        thetaReg[0] = 0
        grad = (1 / m) * ((X.transpose()).dot(e))
        grad += (l2 / m) * thetaReg
        theata -= a * grad
    theatas.append(theata)


def gradiant(X: np.ndarray, Y: np.ndarray, theata: np.ndarray, epochs, a, l2, C):
    m = X.shape[0]
    threads = []
    theata = np.random.random((X_train_poly.shape[1], 1))
    for i in range(0, C):
        Y = (np.array(Y).reshape((m, 1)) == i)
        t = threading.Thread(target=minFunc, args=[X, Y, theata, epochs, a, l2, m, i])
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    theatas1 = np.array(theatas)
    theatas1 = theatas1.reshape((theatas1.shape[1], C))

    return theatas1


#print(dataSet.columns)
corr = dataSet.corr()
# Top n% Correlation training features with the Value
corrPercent = 0.2
top_feature = corr.index[abs(corr['TicketCategory']) > corrPercent]
top_feature = top_feature.delete(-1)

X = dataSet[top_feature]
Y = dataSet["TicketCategory"]  # Label

# print(X.iloc[0:10 ,:])

poly_features = PolynomialFeatures(degree=2, include_bias=True)
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)

X_train, y_train = shuffle(X_train, y_train)
X_train_poly = poly_features.fit_transform(X_train)
# cols = ["num_code", "time_taken_minutes"]
featureNormalization(X_train_poly)
X_train_poly = handleData(X_train_poly)
# print(X_train_poly[0:10,:])

theata = np.random.random((X_train_poly.shape[1], 1))
a = 0.005
epochs = 500
l2 = 0.1
theatas = gradiant(X_train_poly, y_train, theata, epochs, a, l2, 4)

# result = numpy.where(arr == numpy.amax(arr))
tmp = np.transpose(X_train_poly.dot(theatas))
result = np.argmax(tmp, axis=0)
# np.transpose()
accuracy1 = np.mean(result == y_train)
print("\nMSE1 =" + str(accuracy1) + "\n")
# ___________________________________________________________________________
X_test, y_test = shuffle(X_test, y_test)
X_test_poly = poly_features.fit_transform(X_test)
featureNormalization(X_test_poly)
X_test_poly = handleData(X_test_poly)

tmp2 = np.transpose(X_test_poly.dot(theatas))
result2 = np.argmax(tmp2, axis=0)
# np.transpose()
accuracy2 = np.mean(result2 == y_test)
print("\nMSE2 =" + str(accuracy2) + "\n")


# pickle.dump(theatas, open('modelLOGISTIC_Theatas.pkl', 'wb'))