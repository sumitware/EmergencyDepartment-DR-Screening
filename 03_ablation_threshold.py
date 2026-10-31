import numpy as np, pandas as pd, json, time
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

t0 = time.time()
X_train = pd.read_csv("X_train.csv")
y_train = np.load("y_train.npy")
X_test = pd.read_csv("X_test.csv")
y_test = np.load("y_test.npy")
meta = json.load(open("meta.json"))
base_cols = meta["base_cols"]
all_cols = list(X_train.columns)

RANDOM_STATE = 42
grid = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5, 6],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)

def tune_and_eval(cols, label):
    Xtr = X_train[cols]
    search = RandomizedSearchCV(
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="auc", n_jobs=-1),
        grid, n_iter=10, cv=inner_cv, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(Xtr, y_train)
    fold_aucs = []
    for tr_idx, va_idx in rkf.split(Xtr, y_train):
        m = XGBClassifier(random_state=RANDOM_STATE, eval_metric="auc", n_jobs=-1)
        m.set_params(**search.best_params_)
        m.fit(Xtr.iloc[tr_idx], y_train[tr_idx])
        probs = m.predict_proba(Xtr.iloc[va_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_train[va_idx], probs))
    print(f"{label}: repeated CV AUC = {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}  "
          f"best_params={search.best_params_}  ({time.time()-t0:.0f}s)")
    return {"mean": float(np.mean(fold_aucs)), "std": float(np.std(fold_aucs)),
            "best_params": search.best_params_}

ablation = {}
ablation["base_only"] = tune_and_eval(base_cols, f"base only ({len(base_cols)} features)")
ablation["base_plus_engineered"] = tune_and_eval(all_cols, f"base + engineered ({len(all_cols)} features)")

with open("results_ablation.json", "w") as f:
    json.dump(ablation, f, indent=2)

# threshold calibration on the stacking ensemble's held out test predictions
stack_test_probs = np.load("stack_test_probs.npy")
fpr, tpr, thresholds = roc_curve(y_test, stack_test_probs)
youden_idx = np.argmax(tpr - fpr)
youden_threshold = thresholds[youden_idx]

def metrics_at(threshold):
    preds = (stack_test_probs >= threshold).astype(int)
    sens = np.mean(preds[y_test == 1] == 1)
    spec = np.mean(preds[y_test == 0] == 0)
    acc = np.mean(preds == y_test)
    tp = np.sum((preds == 1) & (y_test == 1))
    fn = np.sum((preds == 0) & (y_test == 1))
    precision = tp / (tp + np.sum((preds == 1) & (y_test == 0)) + 1e-9)
    recall = sens
    f2 = 5 * precision * recall / (4 * precision + recall + 1e-9)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc),
            "f2_score": float(f2), "missed_cases": int(fn), "threshold": float(threshold)}

m05 = metrics_at(0.5)
myouden = metrics_at(youden_threshold)
test_auc = roc_auc_score(y_test, stack_test_probs)

def bootstrap_ci(y_true, probs, metric_fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], probs[idx]))
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

from sklearn.metrics import average_precision_score
auc_ci = bootstrap_ci(y_test, stack_test_probs, roc_auc_score)
prauc_ci = bootstrap_ci(y_test, stack_test_probs, average_precision_score)

threshold_results = {
    "test_auc": float(test_auc),
    "test_auc_bootstrap_ci": auc_ci,
    "test_prauc_bootstrap_ci": prauc_ci,
    "youden_threshold": float(youden_threshold),
    "at_0.5": m05,
    "at_youden": myouden,
    "n_test": len(y_test),
    "n_positive_test": int(y_test.sum()),
    "n_negative_test": int((1 - y_test).sum()),
}
print(json.dumps(threshold_results, indent=2))
with open("results_threshold.json", "w") as f:
    json.dump(threshold_results, f, indent=2)

print("done in", time.time() - t0, "s")
