from joblib import dump
import numpy as np
from numpy.lib.function_base import append
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
with open('data/processed/X_train.npy', 'rb') as f:
    X_train = np.load(f)
with open('data/processed/y_train.npy', 'rb') as f:
    y_train = np.load(f)

# print(len(y_train[y_train == 0]), len(y_train[y_train == 1]))

models = [AdaBoostClassifier(), GradientBoostingClassifier(), RandomForestClassifier()]
mnames = ['AdaBoost', 'Gradient Boost', 'Random Forest']


param_grid = {
    'AdaBoost' : { 
    #     'adaboost__base_estimator' : [DecisionTreeClassifier(max_depth=1, class_weight='balanced'), DecisionTreeClassifier(max_depth=10, class_weight='balanced'), 
    #                                     DecisionTreeClassifier(max_depth=1, class_weight=None), DecisionTreeClassifier(max_depth=10, class_weight=None)],
        'learning_rate' : [0.01, 0.1, 1], 
        'n_estimators'  : [10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 500, 1000, 2000, 4000, 8000],
    },
    'Gradient Boost' : {
        'learning_rate' : [0.01, 0.1, 1], 
        'n_estimators'  : [10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 500, 1000, 2000, 4000, 8000],
    },
    'Random Forest' : { 
        'n_estimators' : [10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 500, 1000, 2000, 4000, 8000], 
        'max_features' : [1, 3, 6, 8, 10, 12, 14, 16],
    }
}

scores = []

gsms = []

print(models[0].get_params().keys())
if True:
    for model, mname in zip(models, mnames):
        m = GridSearchCV(model, param_grid[mname], scoring=['accuracy', 'roc_auc'], n_jobs=12, cv=5, return_train_score=True, verbose=100, refit=False)
        m.fit(X_train, y_train)
        gsms.append(m)

        dump(model, f'models/gs_{mname}')

    print("| Model | params | Train Accuracy | Train AUC | Val Accuracy | Val AUC |")
    print("| ----- | ------ | -------------- | --------- | ------------ | ------- |")

    for mname, model in zip(mnames, gsms):
        scores = model.cv_results_

        iloc = np.argsort(scores['mean_test_roc_auc'])[::-1]
        for i in iloc:
            mtacc = scores['mean_train_accuracy'][i]
            stacc = scores['std_train_accuracy'][i]
            mtauc = scores['mean_train_roc_auc'][i]
            stauc = scores['std_train_roc_auc'][i]
            
            mvacc = scores['mean_train_accuracy'][i]
            svacc = scores['std_test_accuracy'][i]
            mvauc = scores['mean_test_roc_auc'][i]
            svauc = scores['std_test_roc_auc'][i]

            pars = scores['params'][i]
            print(f"| {mname} | {pars} | {mtacc:.3f}±{stacc:.3f} | {mtauc:.3f}±{stauc:.3f} | {mvacc:.3f}±{svacc:.3f} | {mvauc:.3f}±{svauc:.3f} |")


