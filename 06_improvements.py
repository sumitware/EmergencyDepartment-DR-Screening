import numpy as np, pandas as pd, json, time, pickle
import torch, torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

t0 = time.time()
torch.manual_seed(42)

X_train_full = pd.read_csv("X_train.csv")
y_train_full = np.load("y_train.npy")
X_test = pd.read_csv("X_test.csv")
y_test = np.load("y_test.npy")
feature_names = list(X_train_full.columns)
n_features = len(feature_names)
meta = json.load(open("meta.json"))

# ---------------------------------------------------------------------
# model definitions (same architecture as 04_transformer.py)
# ---------------------------------------------------------------------
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
    def forward(self, x, return_attention=False):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        attentions = []
        h = seq
        if return_attention:
            for layer in self.encoder.layers:
                attn_out, attn_weights = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
                h2 = layer.norm1(h + layer.dropout1(attn_out))
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h2))))
                h = layer.norm2(h2 + layer.dropout2(ff))
                attentions.append(attn_weights.detach())
        else:
            h = self.encoder(seq)
        cls_out = h[:, 0, :]
        logit = self.head(cls_out).squeeze(-1)
        return (logit, attentions) if return_attention else logit


transformer_results = json.load(open("results_transformer.json"))
best_cfg = transformer_results["best_config"]

def train_transformer(X_tr, y_tr, X_val, y_val, max_epochs=150, patience=15):
    model = FTTransformer(n_features, d_model=best_cfg["d_model"], n_heads=best_cfg["n_heads"],
                           n_layers=best_cfg["n_layers"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    best_val_auc, best_state, wait = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t)).numpy()
        val_auc = roc_auc_score(y_val, val_probs)
        if val_auc > best_val_auc:
            best_val_auc, best_state, wait = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


class TorchWrapper:
    """gives the trained transformer a predict_proba(X) numpy interface"""
    def __init__(self, model, mean, std):
        self.model, self.mean, self.std = model, mean, std
    def predict_proba(self, X):
        Xs = (X - self.mean.values) / self.std.values
        with torch.no_grad():
            probs = torch.sigmoid(self.model(torch.tensor(Xs, dtype=torch.float32))).numpy()
        return probs


results = {}

# ===========================================================================
# IMPROVEMENT 1: repeated cross validation for the transformer
# ===========================================================================
print("=" * 70, "\nIMPROVEMENT 1: repeated CV for the transformer\n", "=" * 70)
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
mean_full = X_train_full.mean()
std_full = X_train_full.std() + 1e-8
X_train_std_full = ((X_train_full - mean_full) / std_full).values

fold_aucs = []
for fold_i, (tr_idx, te_idx) in enumerate(rkf.split(X_train_std_full, y_train_full)):
    X_tr_full, y_tr_full = X_train_std_full[tr_idx], y_train_full[tr_idx]
    X_te_fold, y_te_fold = X_train_std_full[te_idx], y_train_full[te_idx]
    inner_tr_idx, inner_val_idx = train_test_split(
        np.arange(len(X_tr_full)), test_size=0.2, stratify=y_tr_full, random_state=42)
    model = train_transformer(X_tr_full[inner_tr_idx], y_tr_full[inner_tr_idx],
                               X_tr_full[inner_val_idx], y_tr_full[inner_val_idx])
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_te_fold, dtype=torch.float32))).numpy()
    auc = roc_auc_score(y_te_fold, probs)
    fold_aucs.append(auc)
    print(f"  fold {fold_i+1}/10: AUC = {auc:.3f}  ({time.time()-t0:.0f}s elapsed)")

fold_aucs = np.array(fold_aucs)
print(f"\ntransformer repeated CV AUC = {fold_aucs.mean():.3f} +/- {fold_aucs.std():.3f}")
results["improvement1_repeated_cv_transformer"] = {
    "fold_aucs": fold_aucs.tolist(), "mean": float(fold_aucs.mean()), "std": float(fold_aucs.std())}

with open("results_improvements_partial.json", "w") as f:
    json.dump(results, f, indent=2)
print("checkpoint saved,", time.time() - t0, "s elapsed")
