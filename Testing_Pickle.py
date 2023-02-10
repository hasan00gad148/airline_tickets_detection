from statistics import mode
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from main import preproc
import pandas as pd
import pickle

def featureNormalization(X:np.ndarray):
    z = np.array(X)
    for c in range(z.shape[1]):
        m=z[:,c].mean()
        std=z[:,c].std()
        # max = (X[c]).max()
        # min = (X[c]).min()
        if std == 0:
            std = 1
        z[:,c] =  (z[:,c]-m)/(std)
    return z
 


data = preproc("airline-test-samples.csv") 
top_feature = pickle.load(open("top_feature.pkl","rb"))

X = data[top_feature]
Y = data["TicketCategory"]  # Labe
poly_features = PolynomialFeatures(degree=3, include_bias=True)



svm = pickle.load(open("modelSVM.pkl","rb"))

logistic = pickle.load(open("modelLOGISTIC.pkl","rb"))

Tree = pickle.load(open("modelTREE.pkl","rb"))


print("================================================================================")
print("\n model: svm\n")
svmX=featureNormalization(X[top_feature])
p = svm.predict(svmX)
accuracy2 = np.mean(p == Y)
print("\ntestingAccuracy " + str(accuracy2) + "\n")
print("--------------------------------------------------------------")

print("================================================================================")
print("\n model: logistic\n")

logistcX = poly_features.fit_transform(X[top_feature])
logistcX=featureNormalization(logistcX)
p = logistic.predict(logistcX)
accuracy2 = np.mean(p == Y)
print("\ntestingAccuracy " + str(accuracy2) + "\n")
print("--------------------------------------------------------------")

print("================================================================================")
print("\n model: tree\n")
p = Tree.predict(X)
accuracy2 = np.mean(p == Y)
print("\ntestingAccuracy " + str(accuracy2) + "\n")
print("--------------------------------------------------------------")

# tmp2 = np.transpose(X.dot(logistic2Theatas))
# result2 = np.argmax(tmp2, axis=0)
# accuracy2 = np.mean(result2 == Y)
# print("\ntestingAccuracy for from scratch logistic " + str(accuracy2) + "\n")

#pickled_model = pickle.load(open('model.pkl', 'rb'))
#pickled_model.predict(data)
