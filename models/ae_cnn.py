classes = ['Anterior Commissure',
           'Brodmann area 11',
           'Brodmann area 10',
           'Brodmann area 1',
           'Brodmann area 19',
           'Brodmann area 21',
           'Brodmann area 3',
           'Brodmann area 38',
           'Brodmann area 4',
           'Brodmann area 42',
           'Brodmann area 45',
           'Brodmann area 46',
           'Lateral Dorsal Nucleus',
           'Midline Nucleus',
           'Optic Tract']

import numpy as np

X = np.load('../../dataset/X.npy')
X = X.astype('float32') / X.max().astype('float32')
y = np.load('../../dataset/Y.npy')
aes = {}
for clas in classes:
    msk = y == clas
    x_train = X[msk]
    # x_train = x_train.astype('float32') / x_train.max().astype('float32')
    y_train = y[msk]

    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
    from keras.models import Model

    input_img = Input(shape=(74, 20, 43))
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='r')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (4, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(43, (2, 2), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_train, x_train))
    aes[clas] = autoencoder
    autoencoder.save(clas + '.h5')
X_train = np.zeros((X.shape[0], 10, 3, 32))
for ex in range(X.shape[0]):
    cur_clas = y[ex]
    cur_ae = aes[cur_clas]
    encoder = Model(inputs=cur_ae.input,
                    outputs=cur_ae.get_layer('r').output)
    e = np.zeros((1, 74, 20, 43))
    e[0, :, :, :] = X[ex]
    X_train[ex] = encoder.predict(e)
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
import tensorflow as tf
from sklearn import cross_validation
import pandas as pd

y = pd.get_dummies(y)
y = y.as_matrix()
ypred_all = np.zeros((y.shape[0], 15))
cv = cross_validation.KFold(X.shape[0], n_folds=5)
i = 1
###################################
# Define network architecture
###################################
# Input is a 74x20 image with 43 color channels (red, green and blue)
for traincv, testcv in cv:
    print("iteration", i, ":")
    i = i + 1
    trset = X_train[traincv]
    testset = X_train[testcv]
    ytrn = y[traincv]
    ytest = y[testcv]
    network = input_data(shape=[None, 10, 3, 32])
    conv_1 = conv_2d(network, 128, 3, activation='relu', name='conv_1')
    network = max_pool_2d(conv_1, 2)
    conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')
    network = max_pool_2d(conv_2, 2)
    conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')
    network = max_pool_2d(conv_3, 2)
    conv_4 = conv_2d(conv_2, 128, 3, activation='relu', name='conv_4')
    network = max_pool_2d(conv_4, 2)

    network = fully_connected(network, 128, activation='softmax')
    network = dropout(network, 0.3)
    network = fully_connected(network, 15, activation='softmax')
    acc = Accuracy(name="Accuracy")
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.01, metric=acc)
    model = tflearn.DNN(network,
                        tensorboard_verbose=3, tensorboard_dir='tmp/tflearn_logs/')
    model.fit(trset, ytrn, validation_set=(testset, ytest), batch_size=256,
              n_epoch=5, run_id='ae_cnn', show_metric=True)
    yp = model.predict(testset)
    ypred_all[testcv] = yp
    tf.reset_default_graph()
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y.argmax(axis=1), ypred_all.argmax(axis=1)))
from sklearn.metrics import f1_score

print('F1 score micro:')
print(f1_score(y.argmax(axis=1), ypred_all.argmax(axis=1), average='micro'))
print('F1 score macro:')
print(f1_score(y.argmax(axis=1), ypred_all.argmax(axis=1), average='macro'))