import numpy as np
import ROOT
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

'''
I won't be able to fit all 174 files into RAM so I'll need to build a custom generator such as 
https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
'''

random_state = 42

def preprocess(fPhi, fTotal, rsmp=3/2) -> None:
    phi_df = ROOT.RDataFrame("clas12",fPhi)
    raw_df = ROOT.RDataFrame("clas12",fTotal)

    print("num of phi cand:", phi_df.Count().GetValue())
    print("num of all data:", raw_df.Count().GetValue())

    print(phi_df.GetColumnNames())
    print(raw_df.GetColumnNames())

    features = [f for f in raw_df.GetColumnNames()]
    features.remove("evid")
    features.remove("run")

    phi_np = phi_df.AsNumpy()
    raw_np = raw_df.AsNumpy()

    phi_np2 = {}
    for run in np.unique(phi_np["run"]):
        phi_np2[run] = {}
        iloc1 = phi_np["run"] == run
        iloc2 = np.argsort(phi_np["evid"][iloc1])
        for key in phi_np:
            if key != "run":
                phi_np2[run][key] = phi_np[key][iloc1][iloc2]

    raw_np2 = {}
    for run in np.unique(raw_np["run"]):
        raw_np2[run] = {}
        iloc1 = raw_np["run"] == run
        iloc2 = np.argsort(raw_np["evid"][iloc1])
        for key in raw_np:
            if key != "run":
                raw_np2[run][key] = raw_np[key][iloc1][iloc2]

    phi = np.zeros((len(phi_np['run']), len(features)))

    ii_phi = 0
    for run in phi_np2:
        ids = np.searchsorted(raw_np2[run]["evid"], phi_np2[run]["evid"])
        for id in ids:
            phi[ii_phi] = [raw_np2[run][f][id] for f in features]
            ii_phi+=1
        for k in raw_np2[run]:
            raw_np2[run][k] = np.delete(raw_np2[run][k], ids)

    bg = np.zeros((1, len(features)))
    for run in raw_np2:
        bg = np.vstack((bg, np.array([raw_np2[run][f] for f in features]).T))

    bg = bg[1:]

    if rsmp > 0 and rsmp < len(bg)/len(phi):
        bg = resample(bg, n_samples=int(len(phi)*rsmp), random_state=random_state)
    print(f'num of phi cand = {len(phi)}')
    print(f'num of bg noise = {len(bg)}')
    print(f'      imbalance = {len(bg)/len(phi):.0f} bg/phi')

    X = np.vstack((phi, bg))
    y = np.append(np.ones(len(phi)), np.zeros(len(bg)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, random_state=random_state)
    print()
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    joblib.dump(scl, 'data/scaler.gz')

    with open('data/processed/X_train.npy', 'wb') as f:
        np.save(f, X_train)
    with open('data/processed/y_train.npy', 'wb') as f:
        np.save(f, y_train)
    with open('data/processed/X_test.npy', 'wb') as f:
        np.save(f, X_test)
    with open('data/processed/y_test.npy', 'wb') as f:
        np.save(f, y_test)




# def preprocess(fPhi, fTotal):# -> np.array, np.array, np.array, np.array:


# class ROOTParser:
#     def __init__(self) -> None:
#         pass

# class HIPOParser:
#     def __init__(self) -> None:
#         pass

# class DataParser(ROOTParser, HIPOParser):
#     def __init__(self) -> None:
#         super().__init__()

# def preprocess(flist) -> list:
