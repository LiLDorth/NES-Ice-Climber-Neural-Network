from __future__ import print_function
from keras.layers import Dense, Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
import numpy as np


def training():
    train_data = np.load('ICE_Train5.npy')
    train = train_data[::7]
    test = train_data[-3::]
    x_train = np.array([i[0] for i in train]).reshape(-1,80,60,1)
    x_test = np.array([i[0] for i in test]).reshape(-1,80,60,1)
    y_train = np.asarray([i[1] for i in train])
    y_test = np.asarray([i[1] for i in test])

    #Viewing Training Data
    #i = 0
    # while (True):
    #
    #     cv2.imshow('AIBOX', train_data[i][0])
    #     cv2.waitKey(20)
    #     i += 1
    #     if (i > 2500):
    #         i = 0


def main():
    model = Sequential()
    model.add(Dense(100, input_shape=(200,140, 1), activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4, activation='linear'))
    model.compile(lr=.2, loss='mse', optimizer='adam', metrics=['mae'])
    model.save('base.model')
    model.summary()
if __name__ == "__main__":
    main()
