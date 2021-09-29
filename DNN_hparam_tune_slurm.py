import numpy as np
import tensorflow as tf
from tensorflow import keras
from modules.models import hpModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# from modules.data import preprocess
# preprocess('data/raw/DVPhiWagon_ntuple_inb.root', 'data/raw/skim8_epkpkm.root', rsmp=100)

with open('data/processed/X_train.npy', 'rb') as f:
    X_train = np.load(f)
with open('data/processed/y_train.npy', 'rb') as f:
    y_train = np.load(f)



def slurmModel(ID=-1) -> Sequential:
        input_dim = (4*3+4*1,)

        LR      = 0.01#[1e-2, 1e-3, 1e-4][ID]
        units   = 128#[32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024][ID]
        layers  = 6#[1,2,3,4,6,8,10][ID]
        act     = 'relu'#['relu', 'elu', 'selu', 'sigmoid'][ID] # add leakyrelu later
        norm    = 'batch'#['none', 'batch', 'layer'][ID]
        dropout = 0.#[0., .2, .4, .6, .8][ID]

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
        model.compile(opt, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

        return model




val_idx = int(2/3*len(y_train))

X_train, y_train, X_val, y_val = X_train[:val_idx], y_train[:val_idx], X_train[val_idx:], y_train[val_idx:]

callbacks = [
    tf.keras.callbacks.TensorBoard(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
]

model = slurmModel()
model.fit(X_train, y_train, batch_size=64, validation_data=(X_val, y_val),
                epochs=1000, callbacks=callbacks)


model.save('models/phiDNN2')

# tuner = RandomSearch(hpModel, objective=Objective("val_auc", direction="max"), max_trials=10000,
#                         directory='models/', project_name='DNN_search') # Might take 73 days so no......

# tuner = Hyperband(hpModel, objective=Objective("val_auc", direction="max"), max_epochs=1000,
#                     directory='models/', project_name='DNN_search')

# tuner.search(X_train, y_train, validation_data=(X_val, y_val), 
#                 batch_size=128, callbacks=callbacks) # very large batch size, usually <=16...


# tuner.results_summary()