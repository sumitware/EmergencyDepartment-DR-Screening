import numpy as np, pandas as pd, json, time, pickle
import torch, torch.nn as nn

t0 = time.time()
X_test = pd.read_csv("X_test.csv")
X_train = pd.read_csv("X_train.csv")
feature_names = list(X_test.columns)
results = json.load(open("results_improvements_partial.json"))

with open("refit_models.pkl", "rb") as f:
    saved = pickle.load(f)
xgb_model = saved["models"]["xgboost"]

shap_ranking = pd.read_csv("shap_ranking.csv", index_col=0).iloc[:, 0]
last_layer_ranking = pd.read_csv("last_layer_ranking.csv", index_col=0).iloc[:, 0]
rollout_ranking = pd.read_csv("rollout_ranking.csv", index_col=0).iloc[:, 0]


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model=32, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        h = self.encoder(seq)
        return self.head(h[:, 0, :]).squeeze(-1)

n_features = len(feature_names)
transformer_results = json.load(open("results_transformer.json"))
best_cfg = transformer_results["best_config"]
ftt_model = FTTransformer(n_features, d_model=best_cfg["d_model"], n_heads=best_cfg["n_heads"],
                           n_layers=best_cfg["n_layers"])
ftt_model.load_state_dict(torch.load("best_transformer.pt"))
ftt_model.eval()
mean = X_train.mean(); std = X_train.std() + 1e-8

def xgb_predict(X):
    return xgb_model.predict_proba(pd.DataFrame(X, columns=feature_names))[:, 1]

def ftt_predict(X):
    Xs = (pd.DataFrame(X, columns=feature_names) - mean) / std
    with torch.no_grad():
        return torch.sigmoid(ftt_model(torch.tensor(Xs.values, dtype=torch.float32))).numpy()


def deletion_faithfulness(predict_fn, X, ranking, max_k=10):
    X_df = X.copy()
    fill_values = X_df.mean()
    original_probs = predict_fn(X_df.values)
    shifts = []
    X_masked = X_df.copy()
    ranking_order = ranking.index.tolist()
    for k in range(1, max_k + 1):
        col = ranking_order[k - 1]
        X_masked[col] = fill_values[col]
        new_probs = predict_fn(X_masked.values)
        shift = float(np.mean(np.abs(new_probs - original_probs)))
        shifts.append(shift)
    trapezoid_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    score = float(trapezoid_fn(shifts, dx=1))
    return shifts, score

print("=" * 70, "\nIMPROVEMENT 5: faithfulness check (deletion AUC)\n", "=" * 70)

print("\nSHAP ranking, evaluated on the XGBoost model it was computed on:")
shap_shifts, shap_score = deletion_faithfulness(xgb_predict, X_test, shap_ranking, max_k=10)
for k, s in enumerate(shap_shifts, 1):
    print(f"  top {k}: mean |prob shift| = {s:.4f}")
print(f"  faithfulness score = {shap_score:.3f}")

print("\nlast layer attention ranking, evaluated on the transformer:")
last_shifts, last_score = deletion_faithfulness(ftt_predict, X_test, last_layer_ranking, max_k=10)
for k, s in enumerate(last_shifts, 1):
    print(f"  top {k}: mean |prob shift| = {s:.4f}")
print(f"  faithfulness score = {last_score:.3f}")

print("\nattention rollout ranking, evaluated on the transformer:")
roll_shifts, roll_score = deletion_faithfulness(ftt_predict, X_test, rollout_ranking, max_k=10)
for k, s in enumerate(roll_shifts, 1):
    print(f"  top {k}: mean |prob shift| = {s:.4f}")
print(f"  faithfulness score = {roll_score:.3f}")

print(f"\nSUMMARY (higher = more faithful, deletion of own top ranked features "
      f"shifts that model's own predictions more):")
print(f"  SHAP on XGBoost:              {shap_score:.3f}")
print(f"  last layer attention on FTT:  {last_score:.3f}")
print(f"  attention rollout on FTT:     {roll_score:.3f}")

# a natural reference point: same test, but with a RANDOM feature order, averaged over 20 shuffles
rng = np.random.default_rng(0)
random_scores = []
for _ in range(20):
    shuffled = pd.Series(np.arange(n_features), index=np.random.permutation(feature_names))
    _, s = deletion_faithfulness(xgb_predict, X_test, shuffled, max_k=10)
    random_scores.append(s)
random_baseline_xgb = float(np.mean(random_scores))
print(f"\nrandom order baseline on XGBoost (20 shuffles): {random_baseline_xgb:.3f}")

random_scores_ftt = []
for _ in range(20):
    shuffled = pd.Series(np.arange(n_features), index=np.random.permutation(feature_names))
    _, s = deletion_faithfulness(ftt_predict, X_test, shuffled, max_k=10)
    random_scores_ftt.append(s)
random_baseline_ftt = float(np.mean(random_scores_ftt))
print(f"random order baseline on transformer (20 shuffles): {random_baseline_ftt:.3f}")

results["improvement5_faithfulness"] = {
    "shap_on_xgb": {"shifts": shap_shifts, "score": shap_score},
    "last_layer_on_transformer": {"shifts": last_shifts, "score": last_score},
    "rollout_on_transformer": {"shifts": roll_shifts, "score": roll_score},
    "random_baseline_xgb": random_baseline_xgb,
    "random_baseline_transformer": random_baseline_ftt,
}
with open("results_improvements_partial.json", "w") as f:
    json.dump(results, f, indent=2)
print("\ndone in", time.time() - t0, "s")
