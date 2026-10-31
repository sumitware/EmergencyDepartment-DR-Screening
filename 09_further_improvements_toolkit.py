"""
Six extensions to the diabetic retinopathy tabular biomarker study.

This file does not fabricate any results. It is code only. Every function
expects you to plug in your own trained objects, data, and labels from the
original study, marked below as PLUG IN. Run it yourself and the printed or
returned numbers are your real results, ready to drop into the manuscript.

Assumed objects you already have from the original study:
    X            : pandas DataFrame or numpy array, shape (1151, 23), the
                   base plus engineered feature matrix used in the paper.
    y            : array of 0 or 1 labels, shape (1151,).
    feature_names: list of the 23 column names, in the same order as X.
    xgb_model    : your tuned, fitted XGBClassifier (used for SHAP in the paper).
    ftt_model    : your trained feature tokenized transformer (PyTorch).
                   Assumed to expose:
                     ftt_model.predict_proba(X) -> array of probabilities
                     ftt_model.get_attention(X) -> list of per layer attention
                         matrices, shape (n_layers, n_samples, n_tokens, n_tokens)
                   Adjust the two calls marked ADAPT below to match your
                   actual model interface if the method names differ.
    X_test, y_test, test_probs : your held out test set and the stacking
                   ensemble's predicted probabilities on it, used in Table 4
                   of the manuscript.

Requires: numpy, pandas, scikit-learn, torch (only for section 1 and 5),
shap (only for section 5), matplotlib (only for plotting).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# 1. Repeated cross validation for the transformer, matching the protocol
#    already used for the seven classical classifiers (5 fold, 2 repeats).
# ---------------------------------------------------------------------------

def repeated_cv_transformer(build_and_train_fn, X, y, n_splits=5, n_repeats=2,
                             random_state=0):
    """
    build_and_train_fn(X_train, y_train, X_val, y_val) must train a fresh
    transformer with early stopping on the inner validation split and
    return an object with .predict_proba(X) -> 1D array of probabilities.
    This mirrors the paper's own repeated cross validation exactly, so the
    resulting AUC list is directly comparable to Table 1.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=random_state)
    fold_aucs = []
    for fold_i, (train_idx, test_idx) in enumerate(rkf.split(X, y)):
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test_fold, y_test_fold = X[test_idx], y[test_idx]

        # carve an inner validation split from the training fold for early
        # stopping, so the held out fold itself is never used for tuning
        inner = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=random_state)
        inner_train_idx, inner_val_idx = next(inner.split(X_train_full, y_train_full))
        X_tr, y_tr = X_train_full[inner_train_idx], y_train_full[inner_train_idx]
        X_val, y_val = X_train_full[inner_val_idx], y_train_full[inner_val_idx]

        model = build_and_train_fn(X_tr, y_tr, X_val, y_val)
        probs = model.predict_proba(X_test_fold)
        auc = roc_auc_score(y_test_fold, probs)
        fold_aucs.append(auc)
        print(f"fold {fold_i + 1}/{n_splits * n_repeats}: AUC = {auc:.3f}")

    fold_aucs = np.array(fold_aucs)
    print(f"\nrepeated CV transformer AUC: {fold_aucs.mean():.3f} "
          f"+/- {fold_aucs.std():.3f} (n = {len(fold_aucs)} folds)")
    return fold_aucs


# ---------------------------------------------------------------------------
# 2. Richer feature search: tests whether the ceiling is a property of the
#    data itself or only of the current 23 feature representation.
# ---------------------------------------------------------------------------

def generate_candidate_features(df):
    """
    Builds a pool of candidate features beyond the five already in the
    paper: pairwise ratios and products among the microaneurysm and
    exudate columns, plus simple polynomial terms on the two anatomical
    distance features. Adjust the column name substrings below to match
    your actual DataFrame column names.
    """
    df = df.copy()
    ma_cols = [c for c in df.columns if "ma_" in c.lower() or "microaneurysm" in c.lower()]
    ex_cols = [c for c in df.columns if "ex_" in c.lower() or "exudate" in c.lower()]
    candidates = {}

    for i in range(len(ma_cols)):
        for j in range(i + 1, len(ma_cols)):
            a, b = ma_cols[i], ma_cols[j]
            candidates[f"ratio_{a}_{b}"] = df[a] / (df[b] + 1e-6)

    for i in range(len(ex_cols)):
        for j in range(i + 1, len(ex_cols)):
            a, b = ex_cols[i], ex_cols[j]
            candidates[f"ratio_{a}_{b}"] = df[a] / (df[b] + 1e-6)
            candidates[f"product_{a}_{b}"] = df[a] * df[b]

    dist_cols = [c for c in df.columns if "distance" in c.lower() or "disc" in c.lower()]
    for c in dist_cols:
        candidates[f"{c}_squared"] = df[c] ** 2

    return pd.DataFrame(candidates, index=df.index)


def select_and_test_richer_features(X, y, feature_names, tuned_model_builder_fn,
                                     top_k=10, n_repeats=2, random_state=0):
    """
    tuned_model_builder_fn() must return an unfitted, already tuned
    estimator matching the paper's strongest classical model (extreme
    gradient boosting), so the search budget stays identical to Section
    3.2/3.3 of the paper and only the feature set varies, as in the
    original ablation design.

    Selection is by mean absolute correlation with the label on the
    training folds only, avoiding test leakage; swap in a stronger
    selection rule if you prefer, but keep it fold internal.
    """
    X_df = pd.DataFrame(X, columns=feature_names)
    candidates = generate_candidate_features(X_df)
    print(f"generated {candidates.shape[1]} candidate features")

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats,
                                   random_state=random_state)
    base_aucs, expanded_aucs = [], []

    for train_idx, test_idx in rkf.split(X_df, y):
        y_train = np.asarray(y)[train_idx]
        corr = candidates.iloc[train_idx].apply(
            lambda col: abs(np.corrcoef(col, y_train)[0, 1]))
        top_cols = corr.sort_values(ascending=False).index[:top_k].tolist()

        X_base_train = X_df.iloc[train_idx]
        X_base_test = X_df.iloc[test_idx]
        X_exp_train = pd.concat([X_base_train, candidates.iloc[train_idx][top_cols]], axis=1)
        X_exp_test = pd.concat([X_base_test, candidates.iloc[test_idx][top_cols]], axis=1)

        m_base = tuned_model_builder_fn()
        m_base.fit(X_base_train, y_train)
        base_aucs.append(roc_auc_score(np.asarray(y)[test_idx],
                                        m_base.predict_proba(X_base_test)[:, 1]))

        m_exp = tuned_model_builder_fn()
        m_exp.fit(X_exp_train, y_train)
        expanded_aucs.append(roc_auc_score(np.asarray(y)[test_idx],
                                            m_exp.predict_proba(X_exp_test)[:, 1]))

    base_aucs, expanded_aucs = np.array(base_aucs), np.array(expanded_aucs)
    print(f"23 feature baseline:      {base_aucs.mean():.3f} +/- {base_aucs.std():.3f}")
    print(f"23 + top {top_k} candidates: {expanded_aucs.mean():.3f} +/- {expanded_aucs.std():.3f}")
    return base_aucs, expanded_aucs


# ---------------------------------------------------------------------------
# 3. Full cost weighted threshold curve, replacing the two point comparison
#    (0.5 versus Youden) in Table 4 with a continuous curve.
# ---------------------------------------------------------------------------

def cost_weighted_threshold_curve(y_true, probs, cost_ratios=(1, 2, 5, 10, 20)):
    """
    cost_ratios[i] is how many times costlier a missed case (false
    negative) is considered relative to one unnecessary referral (false
    positive). For each ratio, reports the threshold that minimizes
    expected cost, and the sensitivity and specificity obtained there.
    A cost_ratio of 1 recovers something close to Youden's index.
    """
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    fnr = 1 - tpr
    tnr = 1 - fpr
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    results = []
    for ratio in cost_ratios:
        expected_cost = ratio * fnr * n_pos + fpr * n_neg
        best_i = np.argmin(expected_cost)
        results.append({
            "cost_ratio_fn_to_fp": ratio,
            "threshold": float(thresholds[best_i]),
            "sensitivity": float(tpr[best_i]),
            "specificity": float(tnr[best_i]),
        })
        print(f"cost ratio {ratio:>3}:1  threshold {thresholds[best_i]:.3f}  "
              f"sensitivity {tpr[best_i]:.1%}  specificity {tnr[best_i]:.1%}")

    return pd.DataFrame(results)


def plot_threshold_curve(y_true, probs, save_path="threshold_curve.png"):
    import matplotlib.pyplot as plt
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, tpr, label="sensitivity")
    ax.plot(thresholds, 1 - fpr, label="specificity")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("rate")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"saved {save_path}")


# ---------------------------------------------------------------------------
# 4. Three tier triage: refer, uncertain (manual review), no referral.
# ---------------------------------------------------------------------------

def fit_three_tier_thresholds(y_true, probs, review_cost=0.3, cost_ratio=5,
                               grid_step=0.01):
    """
    review_cost is the cost of sending one record to manual review,
    expressed on the same scale as one unnecessary referral (false
    positive) costing 1 unit and one missed case costing cost_ratio units.
    Searches a grid of (t_low, t_high) pairs and returns the pair that
    minimizes total expected cost = missed cases * cost_ratio
    + false referrals * 1 + reviewed cases * review_cost.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
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
                best = {"t_low": t_low, "t_high": t_high, "cost": cost,
                        "missed": int(missed), "false_refer": int(false_refer),
                        "reviewed": int(reviewed)}

    print(f"best thresholds: refer >= {best['t_high']:.2f}, "
          f"no referral < {best['t_low']:.2f}")
    print(f"missed cases {best['missed']}, false referrals {best['false_refer']}, "
          f"sent to manual review {best['reviewed']} of {len(y_true)}")
    return best


# ---------------------------------------------------------------------------
# 5. Faithfulness check: does removing each ranking's top features degrade
#    that ranking's own model as much as it claims those features matter.
# ---------------------------------------------------------------------------

def deletion_faithfulness(model_predict_fn, X, ranking, feature_names,
                           baseline_value="mean", max_k=10):
    """
    model_predict_fn(X) -> array of probabilities, for the SAME model the
    ranking was computed on (XGBoost for the SHAP ranking, the transformer
    for either attention ranking). ranking is a list of feature names in
    the order the explanation method judged most important first.
    Progressively replaces the top k features with their training mean
    (or 0 for a already standardized feature) and measures how much the
    model's mean predicted probability shifts; a faithful ranking should
    produce a fast, large shift as k grows.
    """
    X_df = pd.DataFrame(X, columns=feature_names).copy()
    if baseline_value == "mean":
        fill_values = X_df.mean()
    else:
        fill_values = pd.Series(0.0, index=X_df.columns)

    original_probs = model_predict_fn(X_df.values)
    shifts = []
    X_masked = X_df.copy()
    for k in range(1, max_k + 1):
        col = ranking[k - 1]
        X_masked[col] = fill_values[col]
        new_probs = model_predict_fn(X_masked.values)
        shift = np.mean(np.abs(new_probs - original_probs))
        shifts.append(shift)
        print(f"after masking top {k} features ({col}): mean |prob shift| = {shift:.4f}")

    trapezoid_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area_under_deletion_curve = trapezoid_fn(shifts, dx=1)
    print(f"faithfulness score (area under deletion curve, higher = more faithful): "
          f"{area_under_deletion_curve:.3f}")
    return shifts, area_under_deletion_curve


def compare_ranking_faithfulness(xgb_predict_fn, ftt_predict_fn, X, feature_names,
                                  shap_ranking, last_layer_ranking, rollout_ranking,
                                  max_k=10):
    """Runs the deletion test for all three rankings and prints a summary table."""
    print("=== SHAP ranking, evaluated on the XGBoost model ===")
    _, shap_score = deletion_faithfulness(xgb_predict_fn, X, shap_ranking,
                                           feature_names, max_k=max_k)
    print("\n=== last layer attention ranking, evaluated on the transformer ===")
    _, last_layer_score = deletion_faithfulness(ftt_predict_fn, X, last_layer_ranking,
                                                 feature_names, max_k=max_k)
    print("\n=== attention rollout ranking, evaluated on the transformer ===")
    _, rollout_score = deletion_faithfulness(ftt_predict_fn, X, rollout_ranking,
                                              feature_names, max_k=max_k)

    print("\nsummary (higher area under the deletion curve = more faithful):")
    print(f"  SHAP on XGBoost:            {shap_score:.3f}")
    print(f"  last layer attention on FTT: {last_layer_score:.3f}")
    print(f"  attention rollout on FTT:    {rollout_score:.3f}")
    return {"shap": shap_score, "last_layer": last_layer_score, "rollout": rollout_score}


# ---------------------------------------------------------------------------
# 6. External validation harness for a future emergency department cohort.
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, probs, metric_fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], probs[idx]))
        except ValueError:
            continue
    scores = np.array(scores)
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def external_validation(predict_fn, X_ext, y_ext, feature_names,
                         threshold_05=0.5, threshold_youden=None):
    """
    predict_fn(X) -> probabilities from your already trained stacking
    ensemble. X_ext, y_ext are the new external or emergency department
    cohort, with the same feature_names and column order as training.
    threshold_youden should be the exact value found on your internal
    test set (0.435 in the manuscript), carried over unchanged rather
    than refit here, since refitting it on the external set would not
    be a fair external validation.
    """
    X_ext = pd.DataFrame(X_ext, columns=feature_names)
    probs = predict_fn(X_ext.values)
    y_ext = np.asarray(y_ext)

    auc_mean, auc_lo, auc_hi = bootstrap_ci(y_ext, probs, roc_auc_score)
    print(f"external AUC: {auc_mean:.3f} (95% CI {auc_lo:.3f} to {auc_hi:.3f})")

    for name, t in [("0.5 cutoff", threshold_05),
                    ("internal Youden threshold", threshold_youden)]:
        if t is None:
            continue
        preds = (probs >= t).astype(int)
        sens = np.mean(preds[y_ext == 1] == 1)
        spec = np.mean(preds[y_ext == 0] == 0)
        print(f"{name} ({t:.3f}): sensitivity {sens:.1%}, specificity {spec:.1%}")

    return {"auc": (auc_mean, auc_lo, auc_hi), "probs": probs}


def covariate_shift_report(X_internal, X_external, feature_names):
    """
    Standardized mean difference per feature between the training cohort
    and the new external cohort. A value above roughly 0.5 flags a
    feature whose distribution has shifted enough to warrant caution
    before trusting the model's calibration on the new cohort.
    """
    X_internal = pd.DataFrame(X_internal, columns=feature_names)
    X_external = pd.DataFrame(X_external, columns=feature_names)
    report = []
    for col in feature_names:
        pooled_std = np.sqrt((X_internal[col].std() ** 2 + X_external[col].std() ** 2) / 2)
        smd = abs(X_internal[col].mean() - X_external[col].mean()) / (pooled_std + 1e-9)
        report.append({"feature": col, "standardized_mean_difference": smd})
    report = pd.DataFrame(report).sort_values("standardized_mean_difference", ascending=False)
    print(report.to_string(index=False))
    return report


# ---------------------------------------------------------------------------
# Example of how the six pieces above would be called. Nothing below this
# line executes any real experiment; it only shows the call shape once you
# have supplied your own trained objects in place of the PLUG IN comments.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(__doc__)
    print("Import the functions above into your own analysis notebook or "
          "script, supply your trained models and data where marked "
          "PLUG IN / ADAPT in the module docstring, and call the section "
          "you need. This file performs no computation on its own.")
