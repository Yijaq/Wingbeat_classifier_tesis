import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage
import keras
from keras.utils import image_dataset_from_directory
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Rescaling
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

!unzip -q '/content/drive/MyDrive/Colab Notebooks/Spectrogram80x80.zip'
!mv Spectrogram80x80 /content/Data

train_valid_frame = pd.read_csv('/content/Data/name_label.txt',sep=' ',header=None)
train_valid_frame.columns = ['files','labels']

X = train_valid_frame.iloc[:,0]
y = train_valid_frame.iloc[:,1]

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=7)

cvscores = []
for train , test in kfold.split(X, y):
    train_frame = train_valid_frame.iloc[train]
    valid_frame = train_valid_frame.iloc[test]

    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_iter = train_datagen.flow_from_dataframe(train_frame,
                                         x_col='files',
                                         y_col='labels',
                                         target_size=(80,80),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=256,
                                         shuffle=True)

    valid_iter = valid_datagen.flow_from_dataframe(valid_frame,
                                         x_col='files',
                                         y_col='labels',
                                         target_size=(80,80),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=256,
                                         shuffle=False)

    model = Sequential()
    model.add(Input(shape=(80, 80, 1)))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D( pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D( pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D( pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D( pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D( pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(learning_rate=1e-5, weight_decay=5e-4)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=10,
                                   monitor="val_loss",
                                   restore_best_weights=True)

    history = model.fit(train_iter,
                        epochs=100,
                        validation_data=valid_iter,
                        verbose=0,
                        callbacks=[early_stopping])

    best_epoch = early_stopping.best_epoch
    best_accuracy = history.history['val_accuracy'][best_epoch]
    print('Accuracy: {:.2f}% -- Época: {}'.format(best_accuracy*100, best_epoch))
    cvscores.append(best_accuracy*100)

print("{:.2f}% (+/- {:.2f})".format(np.mean(cvscores), np.std(cvscores)))