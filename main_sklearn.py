from joblib import dump
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate
with open('data/processed/X_train.npy', 'rb') as f:
    X_train = np.load(f)
with open('data/processed/y_train.npy', 'rb') as f:
    y_train = np.load(f)

# print(len(y_train[y_train == 0]), len(y_train[y_train == 1]))

models = [AdaBoostClassifier(), GradientBoostingClassifier(), RandomForestClassifier()]
mnames = ['AdaBoost', 'Gradient Boost', 'Random Forest']

scores = []
for model, mname in zip(models, mnames):
    scores.append(cross_validate(model, X_train, y_train, cv=3, n_jobs=12, scoring=['accuracy', 'roc_auc']))
    model.fit(X_train, y_train)
    dump(model, f'models/{mname}')


print("| Model | Acc Ave | Acc Std Dev | AUC Ave | AUC Std Dev |")
print("| ----- | ------- | ----------- | ------- | ----------- |")

for mname, score in zip(mnames, scores):
    acc_ave = np.average(score['test_accuracy'])
    acc_std = np.std(score['test_accuracy'])
    auc_ave = np.average(score['test_roc_auc'])
    auc_std = np.std(score['test_roc_auc'])
    print(f"| {mname} | {acc_ave:.3f} | {acc_std:.3f} | {auc_ave:.3f} | {auc_std:.3f} |")


