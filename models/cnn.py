from __future__ import division, print_function, absolute_import
import tflearn
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn.layers.normalization import batch_normalization

###################################
### Import data
###################################
import numpy as np
import pandas as pd
# Data
X = np.load('/gpfs/hpchome/gagand87/project/models/cnn/test_models/cv_test_models/new_tests_23_04/dataset/X.npy')
for i in range(X.shape[0]):
    X[i,:,:,:] = X[i,:,:,:].astype('float32') / X[i,:,:,:].max().astype('float32')
y = np.load('/gpfs/hpchome/gagand87/project/models/cnn/test_models/cv_test_models/new_tests_23_04/dataset/Y.npy')
y=pd.get_dummies(y)
y = y.as_matrix()

from sklearn import cross_validation
ypred_all = np.zeros((y.shape[0],15))
cv = cross_validation.KFold(X.shape[0], n_folds=5)

for  traincv, testcv in cv:
    trset = X[traincv]
    ytrn = y[traincv]

    testset = X[testcv]
    ytest = y[testcv]

    network = input_data(shape=[None, 74, 20, 43])
    conv_1 = conv_2d(network,64, 3, activation='elu', name='conv_1')
    network = max_pool_2d(conv_1, 3)
    conv_2 = conv_2d(network,64, 3, activation='elu', name='conv_2')
    network = max_pool_2d(conv_2, 3)
    conv_3 = conv_2d(network,64, 3, activation='elu', name='conv_3')
    network = max_pool_2d(conv_3, 3)

    network = fully_connected(network,256, activation ='elu')
    network = fully_connected(network, 15, activation='softmax')
    # Configure how the network will be trained
    acc = Accuracy(name="Accuracy")
    network = regression(network, optimizer='adagrad',
                             loss='categorical_crossentropy',
                             learning_rate=0.01, metric=acc)
    # Wrap the network in a model object
    model = tflearn.DNN(network,tensorboard_verbose = 3, tensorboard_dir='tmp/')
    ###################################
    # Train model
    ###################################
    model.fit(trset,ytrn, validation_set=(testset,ytest), batch_size=500,n_epoch=10, run_id='CV_15c_basic_64_adagrad_elu', show_metric=True)
    ypred_test = model.predict(testset)
    ypred_all[testcv] = ypred_test
    tf.reset_default_graph()

from sklearn.metrics import confusion_matrix
print("Test CF is:")
print(confusion_matrix(y.argmax(axis=1), ypred_all.argmax(axis=1)))
from sklearn.metrics import f1_score
print('F1 score micro:')
print(f1_score(y.argmax(axis=1), ypred_all.argmax(axis=1), average='micro'))
print('F1 score macro:')
print(f1_score(y.argmax(axis=1), ypred_all.argmax(axis=1), average='macro'))