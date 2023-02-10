from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from main import preproc
import pickle
# dataSet = pd.read_csv("data.csv")
dataset =  preproc("airline-price-classification.csv")
tmp = dataset["TicketCategory"]
dataset.drop(['TicketCategory'],axis=1,inplace=True)
dataset["TicketCategory"] = tmp


# ------------------------------------------------------------------------
corr = dataset.corr()
# Top n% Correlation training features with the Value
corrPercent = 0.2
top_feature = corr.index[abs(corr['TicketCategory']) > corrPercent]
top_feature = top_feature.delete(-1)

X = dataset[top_feature]
Y = dataset["TicketCategory"]  # Label
# ------------------------------------------------------------------------



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X_train,y_train)
y_prediction = clf.predict(X_test)
accuracy=np.mean(y_prediction == y_test)*100
print ("The achieved accuracy using Decision Tree is " + str(accuracy))
pickle.dump(clf, open('modelTREE.pkl', 'wb'))