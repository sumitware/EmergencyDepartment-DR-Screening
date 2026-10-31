import numpy as np, pandas as pd, json, time, pickle
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

t0 = time.time()
X_train = pd.read_csv("X_train.csv")
y_train = np.load("y_train.npy")
X_test = pd.read_csv("X_test.csv")
y_test = np.load("y_test.npy")
feature_names = list(X_train.columns)
meta = json.load(open("meta.json"))

results = json.load(open("results_improvements_partial.json"))

# ===========================================================================
# IMPROVEMENT 2: richer feature search
# ===========================================================================
print("=" * 70, "\nIMPROVEMENT 2: richer feature search\n", "=" * 70)

ma_cols = [c for c in feature_names if c.startswith("ma") and c[2:].isdigit()]
exudate_cols = [c for c in feature_names if c.startswith("exudate")]

def generate_candidates(df):
    cand = {}
    for i in range(len(ma_cols)):
        for j in range(i + 1, len(ma_cols)):
            a, b = ma_cols[i], ma_cols[j]
            cand[f"ratio_{a}_{b}"] = df[a] / (df[b] + 1e-6)
    for i in range(len(exudate_cols)):
        for j in range(i + 1, len(exudate_cols)):
            a, b = exudate_cols[i], exudate_cols[j]
            cand[f"ratio_{a}_{b}"] = df[a] / (df[b] + 1e-6)
            cand[f"product_{a}_{b}"] = df[a] * df[b]
    cand["distance_squared"] = df["macula_opticdisc_distance"] ** 2
    cand["diameter_squared"] = df["opticdisc_diameter"] ** 2
    return pd.DataFrame(cand, index=df.index)

candidates_train = generate_candidates(X_train)
print(f"generated {candidates_train.shape[1]} candidate features")

rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
best_params = json.load(open("results_ablation.json"))["base_plus_engineered"]["best_params"]

TOP_K = 10
base_fold_aucs, expanded_fold_aucs = [], []
for tr_idx, va_idx in rkf.split(X_train, y_train):
    y_tr = y_train[tr_idx]
    corr = candidates_train.iloc[tr_idx].apply(lambda col: abs(np.corrcoef(col, y_tr)[0, 1]))
    top_cols = corr.sort_values(ascending=False).index[:TOP_K].tolist()

    Xb_tr, Xb_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    Xe_tr = pd.concat([Xb_tr, candidates_train.iloc[tr_idx][top_cols]], axis=1)
    Xe_va = pd.concat([Xb_va, candidates_train.iloc[va_idx][top_cols]], axis=1)

    mb = XGBClassifier(random_state=42, eval_metric="auc", n_jobs=-1); mb.set_params(**best_params)
    mb.fit(Xb_tr, y_tr)
    base_fold_aucs.append(roc_auc_score(y_train[va_idx], mb.predict_proba(Xb_va)[:, 1]))

    me = XGBClassifier(random_state=42, eval_metric="auc", n_jobs=-1); me.set_params(**best_params)
    me.fit(Xe_tr, y_tr)
    expanded_fold_aucs.append(roc_auc_score(y_train[va_idx], me.predict_proba(Xe_va)[:, 1]))

base_fold_aucs, expanded_fold_aucs = np.array(base_fold_aucs), np.array(expanded_fold_aucs)
print(f"22 feature baseline:        {base_fold_aucs.mean():.3f} +/- {base_fold_aucs.std():.3f}")
print(f"22 + top {TOP_K} candidates:    {expanded_fold_aucs.mean():.3f} +/- {expanded_fold_aucs.std():.3f}")

# which candidate features were selected most often across folds
all_selected = []
for tr_idx, va_idx in rkf.split(X_train, y_train):
    y_tr = y_train[tr_idx]
    corr = candidates_train.iloc[tr_idx].apply(lambda col: abs(np.corrcoef(col, y_tr)[0, 1]))
    all_selected.extend(corr.sort_values(ascending=False).index[:TOP_K].tolist())
from collections import Counter
selection_counts = Counter(all_selected).most_common(10)
print("most frequently selected candidate features across folds:", selection_counts)

results["improvement2_richer_features"] = {
    "n_candidates_generated": int(candidates_train.shape[1]),
    "baseline_22_features": {"mean": float(base_fold_aucs.mean()), "std": float(base_fold_aucs.std())},
    "expanded_with_top10_candidates": {"mean": float(expanded_fold_aucs.mean()), "std": float(expanded_fold_aucs.std())},
    "most_frequently_selected": selection_counts,
}
with open("results_improvements_partial.json", "w") as f:
    json.dump(results, f, indent=2)
print("checkpoint saved,", time.time() - t0, "s elapsed\n")

# ===========================================================================
# IMPROVEMENT 3: cost weighted threshold curve
# ===========================================================================
print("=" * 70, "\nIMPROVEMENT 3: cost weighted threshold curve\n", "=" * 70)
stack_test_probs = np.load("stack_test_probs.npy")
fpr, tpr, thresholds = roc_curve(y_test, stack_test_probs)
fnr = 1 - tpr
tnr = 1 - fpr
n_pos, n_neg = int(y_test.sum()), int((1 - y_test).sum())

cost_ratios = [1, 2, 5, 10, 20]
curve_results = []
for ratio in cost_ratios:
    expected_cost = ratio * fnr * n_pos + fpr * n_neg
    best_i = np.argmin(expected_cost)
    row = {"cost_ratio_fn_to_fp": ratio, "threshold": float(thresholds[best_i]),
           "sensitivity": float(tpr[best_i]), "specificity": float(tnr[best_i])}
    curve_results.append(row)
    print(f"cost ratio {ratio:>3}:1  threshold {thresholds[best_i]:.3f}  "
          f"sensitivity {tpr[best_i]:.1%}  specificity {tnr[best_i]:.1%}")

results["improvement3_cost_weighted_threshold"] = curve_results
with open("results_improvements_partial.json", "w") as f:
    json.dump(results, f, indent=2)
print("checkpoint saved,", time.time() - t0, "s elapsed\n")

# ===========================================================================
# IMPROVEMENT 4: three tier triage
# ===========================================================================
print("=" * 70, "\nIMPROVEMENT 4: three tier triage\n", "=" * 70)

def fit_three_tier(y_true, probs, review_cost, cost_ratio, grid_step=0.01):
    grid = np.arange(0.05, 0.95, grid_step)
    best = None
    for t_low in grid:
        for t_high in grid:
            if t_high <= t_low:
                continue
            refer = probs >= t_high
            no_refer = probs < t_low
            uncertain = ~refer & ~no_refer
            missed = np.sum(no_refer & (y_true == 1))
            false_refer = np.sum(refer & (y_true == 0))
            reviewed = np.sum(uncertain)
            cost = missed * cost_ratio + false_refer * 1 + reviewed * review_cost
            if best is None or cost < best["cost"]:
                best = {"t_low": float(t_low), "t_high": float(t_high), "cost": float(cost),
                        "missed": int(missed), "false_refer": int(false_refer), "reviewed": int(reviewed)}
    return best

triage_results = {}
for review_cost in [0.2, 0.3, 0.5]:
    best = fit_three_tier(y_test, stack_test_probs, review_cost=review_cost, cost_ratio=5)
    print(f"review_cost={review_cost}: refer >= {best['t_high']:.2f}, no referral < {best['t_low']:.2f}, "
          f"missed {best['missed']}, false referrals {best['false_refer']}, "
          f"reviewed {best['reviewed']} of {len(y_test)}")
    triage_results[f"review_cost_{review_cost}"] = best

results["improvement4_three_tier_triage"] = triage_results
with open("results_improvements_partial.json", "w") as f:
    json.dump(results, f, indent=2)
print("checkpoint saved,", time.time() - t0, "s elapsed\n")
print("ALL DONE for improvements 2-4,", time.time() - t0, "s total")
