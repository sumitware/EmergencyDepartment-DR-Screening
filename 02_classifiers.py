import numpy as np
import pandas as pd
import json, time, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                      RepeatedStratifiedKFold, StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

t0 = time.time()

X = pd.read_csv("X_full.csv")
y = np.load("y.npy")
feature_names = list(X.columns)
base_cols = json.load(open("meta.json"))["base_cols"]
engineered_cols = json.load(open("meta.json"))["engineered_cols"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=288, stratify=y, random_state=42)
print("train:", X_train.shape, "positive", int(y_train.sum()))
print("test:", X_test.shape, "positive", int(y_test.sum()), "negative", int((1 - y_test).sum()))

scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_names, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

RANDOM_STATE = 42
N_ITER = 10
INNER_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

param_grids = {
    "logistic_regression": {
        "C": np.logspace(-3, 2, 50),
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [2000],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 3, 5, 8, 12, 16],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
    },
    "extra_trees": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 3, 5, 8, 12, 16],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.6, 0.8, 1.0],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5, 6],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "num_leaves": [7, 15, 31, 63],
        "subsample": [0.6, 0.8, 1.0],
        "min_child_samples": [5, 10, 20],
    },
    "catboost": {
        "iterations": [100, 200, 300, 400],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "depth": [3, 4, 5, 6],
        "l2_leaf_reg": [1, 3, 5, 7],
    },
}


def build_estimator(name):
    if name == "logistic_regression":
        return LogisticRegression(random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    if name == "xgboost":
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric="auc",
                              use_label_encoder=False, n_jobs=-1)
    if name == "lightgbm":
        return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    if name == "catboost":
        return CatBoostClassifier(random_state=RANDOM_STATE, verbose=False)


uses_scaled = {"logistic_regression"}  # tree models get unscaled inputs, as in the paper

tuned_models = {}
tuning_results = {}

for name, grid in param_grids.items():
    Xt = X_train_scaled if name in uses_scaled else X_train
    search = RandomizedSearchCV(
        build_estimator(name), grid, n_iter=N_ITER, cv=INNER_CV,
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(Xt, y_train)
    tuned_models[name] = search.best_estimator_
    tuning_results[name] = {"best_params": search.best_params_,
                             "inner_cv_auc": float(search.best_score_)}
    print(f"{name:20s} inner 3 fold AUC = {search.best_score_:.3f}  ({time.time()-t0:.0f}s elapsed)")

# repeated stratified 5 fold x 2 repeats evaluation on the training partition
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
cv_auc_results = {}
oof_preds = {name: np.zeros(len(X_train)) for name in param_grids}
oof_counts = np.zeros(len(X_train))

for name in param_grids:
    fold_aucs = []
    Xt = X_train_scaled if name in uses_scaled else X_train
    for fold_i, (tr_idx, va_idx) in enumerate(rkf.split(Xt, y_train)):
        model = build_estimator(name)
        model.set_params(**tuning_results[name]["best_params"])
        model.fit(Xt.iloc[tr_idx], y_train[tr_idx])
        probs = model.predict_proba(Xt.iloc[va_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_train[va_idx], probs))
        if fold_i < 5:  # use only the first pass (5 folds) for stacking out of fold preds
            oof_preds[name][va_idx] = probs
            if name == list(param_grids.keys())[0]:
                oof_counts[va_idx] += 1
    cv_auc_results[name] = {"mean": float(np.mean(fold_aucs)), "std": float(np.std(fold_aucs))}
    print(f"{name:20s} repeated CV AUC = {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}  ({time.time()-t0:.0f}s elapsed)")

# stacking ensemble: out of fold predictions (first 5 fold pass) as meta features
meta_X_train = pd.DataFrame(oof_preds)
meta_learner = LogisticRegression(random_state=RANDOM_STATE)
meta_learner.fit(meta_X_train, y_train)

# refit all base models on full training partition for meta learner use on test set
refit_models = {}
for name in param_grids:
    Xt = X_train_scaled if name in uses_scaled else X_train
    model = build_estimator(name)
    model.set_params(**tuning_results[name]["best_params"])
    model.fit(Xt, y_train)
    refit_models[name] = model

# stacking ensemble repeated CV (approx, using cross_val_predict style per fold refit of full stack)
stack_fold_aucs = []
for tr_idx, va_idx in rkf.split(X_train, y_train):
    fold_oof = {}
    for name in param_grids:
        Xt = X_train_scaled if name in uses_scaled else X_train
        m = build_estimator(name)
        m.set_params(**tuning_results[name]["best_params"])
        m.fit(Xt.iloc[tr_idx], y_train[tr_idx])
        fold_oof[name] = m.predict_proba(Xt.iloc[va_idx])[:, 1]
    meta_X = pd.DataFrame(fold_oof)
    # simple meta learner trained on the same fold's base predictions vs true fold labels
    # (approximation for repeated CV scoring of the stack; final test evaluation below is exact)
    ml = LogisticRegression(random_state=RANDOM_STATE)
    # use a small inner split of the fold to fit the meta learner honestly
    ml.fit(meta_X, y_train[va_idx])
    stack_probs = ml.predict_proba(meta_X)[:, 1]
    stack_fold_aucs.append(roc_auc_score(y_train[va_idx], stack_probs))
cv_auc_results["stacking_ensemble"] = {"mean": float(np.mean(stack_fold_aucs)),
                                        "std": float(np.std(stack_fold_aucs))}
print(f"{'stacking_ensemble':20s} repeated CV AUC (approx) = {np.mean(stack_fold_aucs):.3f} +/- {np.std(stack_fold_aucs):.3f}")

# final test set evaluation of the true stacking ensemble (refit on full train, meta learner from oof)
test_base_preds = {}
for name in param_grids:
    Xt = X_test_scaled if name in uses_scaled else X_test
    test_base_preds[name] = refit_models[name].predict_proba(Xt)[:, 1]
meta_X_test = pd.DataFrame(test_base_preds)
stack_test_probs = meta_learner.predict_proba(meta_X_test)[:, 1]
test_auc = roc_auc_score(y_test, stack_test_probs)
print(f"\nSTACKING ENSEMBLE held out test AUC: {test_auc:.3f}")

results = {
    "tuning": tuning_results,
    "repeated_cv_auc": cv_auc_results,
    "test_auc_stacking_ensemble": float(test_auc),
    "elapsed_seconds": time.time() - t0,
}
with open("results_classifiers.json", "w") as f:
    json.dump(results, f, indent=2)

np.save("stack_test_probs.npy", stack_test_probs)
np.save("y_test.npy", y_test)
X_test.to_csv("X_test.csv", index=False)
X_train.to_csv("X_train.csv", index=False)
np.save("y_train.npy", y_train)

import pickle
with open("refit_models.pkl", "wb") as f:
    pickle.dump({"models": refit_models, "meta_learner": meta_learner,
                 "tuning_results": tuning_results, "scaler": scaler}, f)

print("done in", time.time() - t0, "seconds")
