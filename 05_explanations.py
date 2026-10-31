import numpy as np, pandas as pd, json, time, pickle
import torch
import shap

t0 = time.time()

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = np.load("y_test.npy")
feature_names = list(X_train.columns)

with open("refit_models.pkl", "rb") as f:
    saved = pickle.load(f)
xgb_model = saved["models"]["xgboost"]

# --- SHAP on the tuned XGBoost model ---
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_ranking = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
print("SHAP top 10 features:\n", shap_ranking.head(10))

# --- attention from the trained transformer ---
import importlib.util
spec = importlib.util.spec_from_file_location("transformer_module", "04_transformer.py")
# instead of re-running the training script, redefine the model class here to load weights
import torch.nn as nn

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
        self.n_layers = n_layers

    def forward(self, x, return_attention=False):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        attentions = []
        h = seq
        if return_attention:
            for layer in self.encoder.layers:
                attn_out, attn_weights = layer.self_attn(
                    h, h, h, need_weights=True, average_attn_weights=True)
                h2 = layer.norm1(h + layer.dropout1(attn_out))
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h2))))
                h = layer.norm2(h2 + layer.dropout2(ff))
                attentions.append(attn_weights.detach())
        else:
            h = self.encoder(seq)
        cls_out = h[:, 0, :]
        logit = self.head(cls_out).squeeze(-1)
        if return_attention:
            return logit, attentions
        return logit

transformer_results = json.load(open("results_transformer.json"))
best_cfg = transformer_results["best_config"]
n_features = len(feature_names)
model = FTTransformer(n_features, d_model=best_cfg["d_model"], n_heads=best_cfg["n_heads"],
                       n_layers=best_cfg["n_layers"])
model.load_state_dict(torch.load("best_transformer.pt"))
model.eval()

mean = X_train.mean()
std = X_train.std() + 1e-8
X_test_std = (X_test - mean) / std
X_test_t = torch.tensor(X_test_std.values, dtype=torch.float32)

with torch.no_grad():
    _, attentions = model(X_test_t, return_attention=True)
# attentions: list of (batch, seq, seq), seq = 1 (cls) + n_features

# last layer attention: cls token's attention to each feature token, averaged over test instances
last_layer = attentions[-1]  # (batch, seq, seq)
cls_to_features_last = last_layer[:, 0, 1:].mean(dim=0).numpy()  # (n_features,)
last_layer_ranking = pd.Series(cls_to_features_last, index=feature_names).sort_values(ascending=False)
print("\nlast layer attention top 10:\n", last_layer_ranking.head(10))

# attention rollout: compose attention matrices across all layers, accounting for residual connections
seq_len = last_layer.shape[-1]
identity = torch.eye(seq_len).unsqueeze(0)  # (1, seq, seq)
rollout = identity.expand(last_layer.shape[0], -1, -1).clone()
for attn in attentions:
    attn_res = 0.5 * attn + 0.5 * identity  # average with identity for residual connection
    attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)  # renormalize row stochastic
    rollout = torch.bmm(attn_res, rollout)
cls_to_features_rollout = rollout[:, 0, 1:].mean(dim=0).numpy()
rollout_ranking = pd.Series(cls_to_features_rollout, index=feature_names).sort_values(ascending=False)
print("\nattention rollout top 10:\n", rollout_ranking.head(10))


def concentration_ratio(ranking, k=10):
    top_k = ranking.head(k).values
    return float(top_k[0] / (top_k[-1] + 1e-9))


def jaccard(ranking_a, ranking_b, k):
    a = set(ranking_a.head(k).index)
    b = set(ranking_b.head(k).index)
    return len(a & b) / len(a | b)


summary = {
    "shap_top10": shap_ranking.head(10).to_dict(),
    "last_layer_top10": last_layer_ranking.head(10).to_dict(),
    "rollout_top10": rollout_ranking.head(10).to_dict(),
    "concentration_ratio": {
        "shap": concentration_ratio(shap_ranking),
        "last_layer_attention": concentration_ratio(last_layer_ranking),
        "attention_rollout": concentration_ratio(rollout_ranking),
    },
    "jaccard_top5": {
        "last_layer_vs_rollout": jaccard(last_layer_ranking, rollout_ranking, 5),
        "shap_vs_last_layer": jaccard(shap_ranking, last_layer_ranking, 5),
        "shap_vs_rollout": jaccard(shap_ranking, rollout_ranking, 5),
    },
    "jaccard_top10": {
        "last_layer_vs_rollout": jaccard(last_layer_ranking, rollout_ranking, 10),
        "shap_vs_last_layer": jaccard(shap_ranking, last_layer_ranking, 10),
        "shap_vs_rollout": jaccard(shap_ranking, rollout_ranking, 10),
    },
    "top_feature": {
        "shap": shap_ranking.index[0],
        "last_layer_attention": last_layer_ranking.index[0],
        "attention_rollout": rollout_ranking.index[0],
    },
}
print(json.dumps(summary, indent=2))
with open("results_explanations.json", "w") as f:
    json.dump(summary, f, indent=2)

shap_ranking.to_csv("shap_ranking.csv")
last_layer_ranking.to_csv("last_layer_ranking.csv")
rollout_ranking.to_csv("rollout_ranking.csv")

print("done in", time.time() - t0, "s")
