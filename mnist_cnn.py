# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(20160723) # シード値を固定

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import matplotlib.pyplot as plt

# MNIST データセットを取り込む
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 変換前：28 x 28 の2次元配列 x 60,000
# 変換後：1 x 28 x 28 の3次元配列 x 60,000
#         - 今回はグレースケールなので1チャネル（RGB などであれば3チャネル）
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32') / 255
X_test  = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32') / 255

# 変換前：0 〜 9 の数字 x 60,000
# 変換後：10要素の1次元配列（one-hot 表現） x 60,000
#         - 0 : [1,0,0,0,0,0,0,0,0,0]
#         - 1 : [0,1,0,0,0,0,0,0,0,0]
#         ...
Y_train = np_utils.to_categorical(y_train, 10)
Y_test  = np_utils.to_categorical(y_test, 10)

# シーケンシャルモデル
model = Sequential()

# 畳み込み層 1
# - フィルタ数：32
# - カーネルサイズ：3 x 3
# - 活性化関数：relu
# - 出力：26 x 26 の2次元配列 x 32
model.add(Convolution2D(32, 3, 3,
                        border_mode='valid',
                        input_shape=(1, 28, 28)))
model.add(Activation('relu'))

# 畳み込み層 2
# - フィルタ数：32
# - カーネルサイズ：3 x 3
# - 活性化関数：relu
# - 出力：24 x 24 の2次元配列 x 32
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))

# プーリング層
# - プールサイズ：2 x 2
# - ドロップアウト比率：0.25
# - 出力：12 x 12 の2次元配列 x 32
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 1次元配列に変換
# - 入力：12 x 12 の2次元配列 x 32
# - 出力：4608要素の1次元配列
model.add(Flatten())

# 隠れ層
# - ノード数：128
# - 活性化関数：relu
# - ドロップアウト比率：0.5
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 出力層
# - ノード数：10
# - 活性化関数：softmax
model.add(Dense(10))
model.add(Activation('softmax'))

# モデルの要約を出力
model.summary()

# 学習過程の設定
# - 目的関数：categorical_crossentropy
# - 最適化アルゴリズム：adadelta
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# 学習
# - バッチサイズ：128
# - 学習の繰り返し回数：12
history = model.fit(X_train, Y_train,
                    batch_size=128,
                    nb_epoch=12,
                    verbose=1,
                    validation_data=(X_test, Y_test))

# 評価
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 学習過程をグラフで出力
loss     = history.history['loss']
val_loss = history.history['val_loss']
nb_epoch = len(loss)
plt.plot(range(nb_epoch), loss, marker='.', label='loss')
plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
