classes = [  # 'Anterior Commissure',
    'Brodmann area 11',
    # 'Brodmann area 10',
    'Brodmann area 1',
    # 'Brodmann area 19',
    # 'Brodmann area 21',
    # 'Brodmann area 3',
    # 'Brodmann area 38',
    # 'Brodmann area 4',
    # 'Brodmann area 42',
    # 'Brodmann area 45',
    'Brodmann area 46']
# 'Lateral Dorsal Nucleus',
# 'Midline Nucleus',
# 'Optic Tract']

import numpy as np

X = np.load('../dataset/X.npy')
X = X.astype('float32') / X.max().astype('float32')
y = np.load('../dataset/Y.npy')
aes = {}
from keras.models import load_model

for clas in classes:
    # msk = y == clas
    # x_train = X[msk]
    ##x_train = x_train.astype('float32') / x_train.max().astype('float32')
    # y_train = y[msk]

    # from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
    # from keras.models import Model

    # input_img = Input(shape=(74, 20, 43))
    # x = Conv2D(128, (2, 2), activation='relu', padding='same')(input_img)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # print(x.get_shape())
    # x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # print(x.get_shape())
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # print(x.get_shape())
    # encoded = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(128, (4, 3), activation='relu')(x)
    # x = UpSampling2D((2, 2))(x)
    # decoded = Conv2D(43, (2, 2), activation='sigmoid', padding='same')(x)
    # print(decoded.get_shape())
    # autoencoder = Model(input_img, decoded)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    # if clas == 'Brodmann area 10':
    #	epochs = 5
    # else:
    #	epochs = 1
    # autoencoder.fit(x_train, x_train,
    #        epochs=10,
    #        batch_size=64,
    #        shuffle=True,
    #        validation_data=(x_train, x_train))
    aes[clas] = load_model(clas + '.h5')

# aes_avg_error = {}

# for clas in aes:
#	msk = y == clas
#	x_train = X[msk]
#	x_train = x_train.astype('float32') / x_train.max().astype('float32')
#	y_train = y[msk]
#	rit = aes[clas].predict(x_train)
#	srit = 0
#	rcerrorstrain = np.zeros(rit.shape[0])
#	for j in range(rit.shape[0]):
#       	srit = srit+np.sum(np.sqrt((x_train[j,:,:,:]-rit[j,:,:,:])**2))
#       	rcerrorstrain[j] = np.sum(np.sqrt((x_train[j,:,:,:]-rit[j,:,:,:])**2))
#	print(clas,"'s average rc error is:")
#	print(srit/rit.shape[0])
#	aes_avg_error[clas] = srit/rit.shape[0]
#	print(clas,' has below min and max rc errors:')
#	print(rcerrorstrain.min(),rcerrorstrain.max())
#	print("++++++++++++++++++++++++++++++++++++++++++++++")
# for clas in classes:
#	print(clas)
#	if clas == 'Brodmann area 10':
#		msk = y == clas
#		x_train = X[msk]
#		x_train = x_train.astype('float32') / x_train.max().astype('float32')
#		y_train = y[msk]
#		for ae in aes:
#			print(aes[ae])
#			rit = aes[ae].predict(x_train)
#			print("For class",clas,"ae",ae,"gives below error")
#			rerr = np.sum((x_train[1,:,:,:]-rit[1,:,:])**2)
#			print(rerr)
#			print(aes_avg_error[ae] - rerr)
# msk =: (y == 'Brodmann area 1')| (y == 'Brodmann area 10')| (y == 'Brodmann area 11') |( y == 'Brodmann area 46')
msk = (y == 'Brodmann area 1') | (y == 'Brodmann area 11') | (y == 'Brodmann area 46')
x_test = X[msk]
x_test = x_test.astype('float32') / x_test.max().astype('float32')
print(x_test.shape)
y_test = y[msk]
y_pred = [0 for i in range(y_test.shape[0])]
for i in range(x_test.shape[0]):
    xt = np.zeros((1, 74, 20, 43))
    xt[0, :, :, :] = x_test[i]
    rerrs = {}
    for ae in aes:
        rit = aes[ae].predict(xt)
        rerr = np.sum(np.sqrt((xt[0, :, :, :] - rit[0, :, :]) ** 2))
        # err = abs(aes_avg_error[ae] - rerr)
        rerrs[ae] = rerr
    # print("orig class is",y_test[i])
    pclas = min(rerrs, key=rerrs.get)
    # print("pred_class is" ,pclas)
    y_pred[i] = pclas
y_pred = np.array(y_pred)
# np.save('ypred',y_pred)
# np.save('ytest',y_test)
from sklearn.metrics import confusion_matrix

print("CF")
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import f1_score

print("F1 Score is")
print(f1_score(y_test, y_pred, average='macro'))