# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(20160715) # シード値を固定

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

import matplotlib.pyplot as plt

# MNIST データセットを取り込む
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 変換前：28 x 28 の2次元配列 x 60,000
# 変換後：784要素の1次元配列 x 60,000（256階調を 0 〜 1 に正規化）
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test  = X_test.reshape(10000, 784).astype('float32') / 255

# 変換前：0 〜 9 の数字 x 60,000
# 変換後：10要素の1次元配列（one-hot 表現） x 60,000
#         - 0 : [1,0,0,0,0,0,0,0,0,0]
#         - 1 : [0,1,0,0,0,0,0,0,0,0]
#         ...
Y_train = np_utils.to_categorical(y_train, 10)
Y_test  = np_utils.to_categorical(y_test, 10)

# シーケンシャルモデル
model = Sequential()

# 隠れ層 1
# - ノード数：512
# - 入力：784次元
# - 活性化関数：relu
# - ドロップアウト比率：0.2
model.add(Dense(512, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 隠れ層 2
# - ノード数：512
# - 活性化関数：relu
# - ドロップアウト比率：0.2
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 出力層
# - ノード数：10
# - 活性化関数：softmax
model.add(Dense(10))
model.add(Activation('softmax'))

# モデルの要約を出力
model.summary()

# 学習過程の設定
# - 目的関数：categorical_crossentropy
# - 最適化アルゴリズム：rmsprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 学習
# - バッチサイズ：128
# - 学習の繰り返し回数：20
history = model.fit(X_train, Y_train,
          batch_size=128,
          nb_epoch=20,
          verbose=1,
          validation_data=(X_test, Y_test))

# 評価
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

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
