import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

clf=RandomForestClassifier(random_state=0,class_weight='balanced',n_estimators=7,max_depth=7,criterion='entropy')

X_train = np.load('/gpfs/hpchome/gagand87/project/Data/processed_data/final_dataset/rf/pca_Xtrain.npy')
X_test = np.load('/gpfs/hpchome/gagand87/project/Data/processed_data/final_dataset/rf/pca_Xtest.npy')
y_train = np.load('/gpfs/hpchome/gagand87/project/Data/processed_data/final_dataset/rf/pca_ytrain.npy')
y_test = np.load('/gpfs/hpchome/gagand87/project/Data/processed_data/final_dataset/rf/pca_ytest.npy')

mskt = (y_train == 'Brodmann area 1')|(y_train == 'Brodmann area 11')|(y_train == 'Brodmann area 46')

X_train = X_train[mskt]
y_train = y_train[mskt]

msk = (y_test == 'Brodmann area 1')|(y_test == 'Brodmann area 11')|(y_test == 'Brodmann area 46')
X_test = X_test[msk]
y_test = y_test[msk]

X = np.append(X_train,X_test,axis=0)
Y=np.append(y_train,y_test,axis=0)

print(clf.fit(X, Y))
y_trainp = clf.predict(X)

cftrain=confusion_matrix(Y,y_trainp)
from sklearn.metrics import f1_score
print(cftrain)
print(f1_score(Y, y_trainp, average='micro'))
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(clf, X, Y,cv=5)
print(confusion_matrix(Y,y_pred))
print(f1_score(Y, y_pred, average='macro'))