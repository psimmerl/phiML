
# phiML

## Dataset

* skim8 (should switch to GEMC and use skim8 as the test dataset)
* phi candidates - 18168 (0.05%)
* Total num of events - 36,988,680 (too small... missing factor of 10 or something)

|  Type  |  Count  | Percent |
| ------ | ------- | ------- |
| Phi    | 18,168  | 0.005% |
| epkpkm | 298,978 | 0.081% |
| Total  | 369,885,938 |    |

*required at least 1 epkpkm and FD



### Preprocessing

* Require 1 *e p kp km*
    * should probably do this because we want to look at ***exclusive*** phi

* Require forward detector
    * should probably also do this because the different calibration metrics will make the convergence take longer and probably be less accurate

* Do I need to downsample to account for the unequal priors
    * currently have P(phi)=2/3 * P(backgroung)


### Features

* ~~vx, vy,~~ vz, px, py, pz, ~~E,~~
* ~~Q^2, t, xb~~
* ~~permutation over all invariant masses?~~

~~If I want to replicate the previous cuts:~~
* ~~hadron vertex difference, missing energy and masses, coplanarity, etc~~


## Architectures to test:

* Deep Neural Network
* Random Forest
* AdaBoost
* Gradient Boost -> XGBoost

---

---

# Preliminary Results


* Data: **skim8**
    * Used wagon to find phi
    * background:
        * required at least 1 epkpkm
            * if there are multiple epkpkm I take the last set (need to fix/decide how to fix)
        * required all in FD
* Features:
    * px, py, pz, vz for each e, p, kp, km (16 feats total)
    * StandardScaler
    * required at least 1 epkpkm
        * if there are multiple epkpkm I take the last set (need to fix/decide how to fix)
    * required all in FD

## Ensemble Methods:
* 3-fold cross validation on 
    * 18145 background
    * 12135 phi
* Still need to try XGBoost
* Need to hyperparameter tune 

| Model | Acc Ave | Acc Std Dev | AUC Ave | AUC Std Dev |
| ----- | ------- | ----------- | ------- | ----------- |
| AdaBoost | 0.838 | 0.001 | 0.905 | 0.001 |
| Gradient Boost | 0.846 | 0.003 | 0.917 | 0.003 |
| Random Forest | 0.854 | 0.002 | 0.922 | 0.002 |



## Deep Neural Net:
* epochs = 100
* batch size = 16
* hidden activation: relu
* final activation: sigmoid
* Train
    * 12063 background
    * 8123 phi
* Validation:
    * 6082 background
    * 4012 phi

| Layers | Batch Norm | Dropout | Train Loss | Train Acc. | Train AUC | Val Loss | Val Acc. | Val AUC |
| ------ | ---------- | ------- | ---------- | ---------- | --------- | -------- | -------- | ------- |
| (512, 512, 256, 128, 2) | True | 0.0 | 0.197 | 0.925 | 0.970 | 0.296 | 0.881 | 0.934 |
| (512, 512, 256, 128, 2) | True | 0.2 | 0.233 | 0.909 | 0.960 | 0.269 | 0.894 | 0.942 |
| (512, 512, 256, 128, 2) | True | 0.4 | 0.262 | 0.897 | 0.954 | 0.281 | 0.891 | 0.942 |
| (512, 512, 256, 128, 2) | False | 0.0 | 0.306 | 0.953 | 0.974 | 0.990 | 0.860 | 0.910 |
| (512, 512, 256, 128, 2) | False | 0.2 | 0.231 | 0.929 | 0.959 | 0.468 | 0.890 | 0.930 |
| (512, 512, 256, 128, 2) | False | 0.4 | 0.387 | 0.897 | 0.949 | 0.414 | 0.870 | 0.925 |
| (512, 512, 512, 512, 512) | True | 0.0 | 0.244 | 0.919 | 0.967 | 0.287 | 0.879 | 0.937 |
| (512, 512, 512, 512, 512) | True | 0.2 | 0.222 | 0.913 | 0.964 | 0.269 | 0.892 | 0.942 |
| (512, 512, 512, 512, 512) | True | 0.4 | 0.244 | 0.898 | 0.957 | 0.265 | 0.891 | 0.943 |
| (512, 512, 512, 512, 512) | False | 0.0 | 0.530 | 0.957 | 0.974 | 1.887 | 0.870 | 0.911 |
| (512, 512, 512, 512, 512) | False | 0.2 | 0.200 | 0.933 | 0.977 | 0.499 | 0.886 | 0.934 |
| (512, 512, 512, 512, 512) | False | 0.4 | 0.277 | 0.898 | 0.958 | 1.635 | 0.887 | 0.943 |
| (128, 128, 128) | True | 0.0 | 0.207 | 0.923 | 0.970 | 0.276 | 0.893 | 0.941 |
| (128, 128, 128) | True | 0.2 | 0.322 | 0.879 | 0.949 | 0.347 | 0.865 | 0.933 |
| (128, 128, 128) | True | 0.4 | 0.254 | 0.890 | 0.949 | 0.266 | 0.887 | 0.942 |
| (128, 128, 128) | False | 0.0 | 0.300 | 0.952 | 0.975 | 1.034 | 0.853 | 0.904 |
| (128, 128, 128) | False | 0.2 | 0.180 | 0.932 | 0.976 | 0.307 | 0.894 | 0.941 |
| (128, 128, 128) | False | 0.4 | 0.212 | 0.916 | 0.963 | 0.266 | 0.904 | 0.949 |
| (64, 64, 64, 64) | True | 0.0 | 0.247 | 0.910 | 0.960 | 0.271 | 0.889 | 0.938 |
| (64, 64, 64, 64) | True | 0.2 | 0.274 | 0.886 | 0.947 | 0.287 | 0.880 | 0.939 |
| (64, 64, 64, 64) | True | 0.4 | 0.268 | 0.883 | 0.946 | 0.276 | 0.880 | 0.941 |
| (64, 64, 64, 64) | False | 0.0 | 0.346 | 0.948 | 0.971 | 0.999 | 0.865 | 0.911 |
| (64, 64, 64, 64) | False | 0.2 | 0.213 | 0.916 | 0.964 | 0.269 | 0.902 | 0.949 |
| (64, 64, 64, 64) | False | 0.4 | 0.230 | 0.909 | 0.958 | 0.282 | 0.901 | 0.948 |


