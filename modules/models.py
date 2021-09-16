import tensorflow as tf
import numpy as np
import keras

from keras.models import Sequential
from keras import layers

class phiDNN:
    def __init__(self, save_path='models/phiDNN') -> None:
        self.save_path = save_path
        self.makeModel()
        pass

    def makeModel(self, units = (512,512,256,128,2), input_dim=(4*3+4,), dropout=0., batch_norm=True, ) -> Sequential:
        input_dim = (4*3+4,)

        model = Sequential()
        model.add(layers.Input(shape=input_dim))
        # model.add(layers.BatchNormalization()) # Might be (a lot) better to do this in my preprocessor
        for unit in units:
            model.add(layers.Dense(unit, activation='relu'))
            if batch_norm:
                model.add(layers.BatchNormalization())
            if dropout:
                model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile('adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

        self.model = model
        return self.model


    def fit(self, X_train, y_train, batch_size=16, epochs=100, validation_split=1/3, validation_data=None) -> Sequential:
        #, X_validation=None, y_validation=None
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=self.save_path, 
                    save_freq='epoch', period=10),#, save_best_only=True),
            # keras.callbacks.EarlyStopping(monitor='val_loss', patience=16, restore_best_weights=True),
            # keras.callbacks.TensorBoard()
        ]

        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data)

        return self.model


    def save(self, name="custom_save") -> None:
        self.model.save(self.save_path+name)