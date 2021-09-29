import tensorboard
import numpy as np
import keras
from keras_tuner import RandomSearch, Hyperband, Objective
from modules.models import hpModel

# from modules.data import preprocess
# preprocess('data/raw/DVPhiWagon_ntuple_inb.root', 'data/raw/skim8_epkpkm.root', rsmp=100)

with open('data/processed/X_train.npy', 'rb') as f:
    X_train = np.load(f)
with open('data/processed/y_train.npy', 'rb') as f:
    y_train = np.load(f)

val_idx = int(2/3*len(y_train))

X_train, y_train, X_val, y_val = X_train[:val_idx], y_train[:val_idx], X_train[val_idx:], y_train[val_idx:]

callbacks = [
    keras.callbacks.TensorBoard(),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
]

tuner = RandomSearch(hpModel, objective=Objective("val_auc", direction="max"), max_trials=10000,
                        directory='models/', project_name='DNN_search')

# tuner = Hyperband(hpModel, objective=Objective("val_auc", direction="max"), max_epochs=1000,
#                     directory='models/', project_name='DNN_search')

tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), 
                batch_size=128, callbacks=callbacks) # very large batch size, usually <=16...


tuner.results_summary()

tuner.get_best_models()[0].save('models/phiDNN_opt2')