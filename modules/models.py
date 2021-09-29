import tensorflow as tf
import numpy as np
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam

def hpModel(hp) -> Sequential:
        input_dim = (4*3+4*1,)

        LR = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        units   = hp.Choice('units', [96, 128, 192, 256, 384, 512, 1024]) #[32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024])
        layers  = hp.Choice('layers', [2,4,6]) #[1,2,3,4,6,8,10]
        act     = 'relu' #['relu', 'elu', 'selu', 'sigmoid'][hp.Choice('act', [0, 1, 2, 3])] # add leakyrelu later
        norm    = hp.Choice('norm', ['none', 'batch']) #['none', 'batch', 'layer'][hp.Choice('norm_type', [0, 1, 2])]
        dropout = hp.Choice('dropout', [0., .4]) #[0., .2, .4, .6, .8])

        # final_act = 'softmax'

        model = Sequential()
        model.add(Input(shape=input_dim))

        for l in range(layers):
            model.add(Dense(units, activation=act))

            if norm == 'batch':
                model.add(BatchNormalization())
            elif norm == 'layer':
                model.add(LayerNormalization())

            if dropout:
                model.add(Dropout(dropout))

        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=LR)
        model.compile('adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

        return model


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
        model.add(layers.Dense(1, activation='sigmoid')) # returns ~probabilities~

        model.compile('adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

        self.model = model
        return self.model

    # Need to add VAE
    # Need to add GAN


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