import numpy as np
from modules.models import phiDNN
from modules.data import preprocess

# preprocess("data/raw/DVPhiWagon_ntuple_inb.root", "data/raw/skim8_epkpkm.root")

with open('data/processed/X_train.npy', 'rb') as f:
    X_train = np.load(f)
with open('data/processed/y_train.npy', 'rb') as f:
    y_train = np.load(f)

val_idx = int(2/3*len(y_train))

X_train, y_train, X_val, y_val = X_train[:val_idx], y_train[:val_idx], X_train[val_idx:], y_train[val_idx:]

layers = [(512, 512, 256, 128, 2),
          (512, 512, 512, 512, 512),
          (128, 128, 128),
          (64, 64, 64, 64) ]
dropout = [0.0, 0.2, 0.4]
batch_norm = [True, False]

models = []

for l in layers:
    for bn in batch_norm:
        for do in dropout:
            dnn_model = phiDNN()
            dnn_model.makeModel(units=l, dropout=do, batch_norm=bn)
            models.append(dnn_model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_val, y_val)))

print("\n\n")
print("| Layers | Activation | Batch Norm | Dropout | Train Loss | Train Acc. | Train AUC | Val Loss | Val Acc. | Val AUC |")
print("| ------ | ---------- | ---------- | ------- | ---------- | ---------- | --------- | -------- | -------- | ------- |")

for i, model in enumerate(models):
    l = layers[int(i/len(batch_norm)/len(dropout))%len(layers)]
    bn = batch_norm[int(i/len(dropout))%len(batch_norm)]
    do = dropout[i%len(dropout)]

    ht = model.evaluate(X_train, y_train, verbose=0)
    hv = model.evaluate(X_val, y_val, verbose=0)

    print(f"| {l} | relu | {bn} | {do} | {ht[0]:.3f} | {ht[1]:.3f} | {ht[2]:.3f} | {hv[0]:.3f} | {hv[1]:.3f} | {hv[2]:.3f} |")


