import numpy as np, pandas as pd, json, time
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

t0 = time.time()
torch.manual_seed(42)

X_train_full = pd.read_csv("X_train.csv")
y_train_full = np.load("y_train.npy")
X_test = pd.read_csv("X_test.csv")
y_test = np.load("y_test.npy")
feature_names = list(X_train_full.columns)
n_features = len(feature_names)

# standardize using training statistics only
mean = X_train_full.mean()
std = X_train_full.std() + 1e-8
X_train_std = (X_train_full - mean) / std
X_test_std = (X_test - mean) / std

# carve an inner train/validation split from the training partition for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_std, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test_std.values, dtype=torch.float32)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x):
        # x: (batch, n_features) -> tokens: (batch, n_features, d_model)
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
        self.n_heads = n_heads

    def forward(self, x, return_attention=False):
        tokens = self.tokenizer(x)  # (batch, n_features, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)  # (batch, 1+n_features, d_model)

        attentions = []
        h = seq
        if return_attention:
            for layer in self.encoder.layers:
                attn_out, attn_weights = layer.self_attn(
                    h, h, h, need_weights=True, average_attn_weights=True)
                h2 = layer.norm1(h + layer.dropout1(attn_out))
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h2))))
                h = layer.norm2(h2 + layer.dropout2(ff))
                attentions.append(attn_weights.detach())  # (batch, seq, seq)
        else:
            h = self.encoder(seq)

        cls_out = h[:, 0, :]
        logit = self.head(cls_out).squeeze(-1)
        if return_attention:
            return logit, attentions
        return logit


def train_one(d_model, n_heads, n_layers, lr, max_epochs=150, patience=15, verbose=False):
    model = FTTransformer(n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc, best_state, wait = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).numpy()
            val_auc = roc_auc_score(y_val, val_probs)
        if val_auc > best_val_auc:
            best_val_auc, best_state, wait = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch}: loss {loss.item():.4f} val_auc {val_auc:.3f}")

    model.load_state_dict(best_state)
    return model, best_val_auc


configs = [
    {"d_model": 32, "n_heads": 4, "n_layers": 3, "lr": 0.002, "label": "typical (32,4,3)"},
    {"d_model": 32, "n_heads": 4, "n_layers": 3, "lr": 0.001, "label": "typical, lower lr"},
    {"d_model": 16, "n_heads": 4, "n_layers": 2, "lr": 0.003, "label": "small (16,4,2)"},
    {"d_model": 64, "n_heads": 8, "n_layers": 3, "lr": 0.002, "label": "wide (64,8,3)"},
    {"d_model": 32, "n_heads": 4, "n_layers": 4, "lr": 0.003, "label": "deep (32,4,4)"},
]

search_results = []
best_overall = {"val_auc": -1}
for cfg in configs:
    model, val_auc = train_one(cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["lr"])
    print(f"{cfg['label']:25s} val AUC = {val_auc:.3f}  ({time.time()-t0:.0f}s)")
    search_results.append({**cfg, "val_auc": float(val_auc)})
    if val_auc > best_overall["val_auc"]:
        best_overall = {**cfg, "val_auc": float(val_auc)}
        torch.save(model.state_dict(), "best_transformer.pt")
        best_cfg = cfg

print("\nbest config:", best_overall)

# reload best model and evaluate on the true held out test set
best_model = FTTransformer(n_features, d_model=best_cfg["d_model"],
                            n_heads=best_cfg["n_heads"], n_layers=best_cfg["n_layers"])
best_model.load_state_dict(torch.load("best_transformer.pt"))
best_model.eval()
with torch.no_grad():
    test_logits = best_model(X_test_t)
    test_probs = torch.sigmoid(test_logits).numpy()
test_auc = roc_auc_score(y_test, test_probs)
print(f"best transformer, held out test AUC = {test_auc:.3f}")

results = {
    "search_results": search_results,
    "best_config": best_overall,
    "test_auc": float(test_auc),
    "elapsed_seconds": time.time() - t0,
}
with open("results_transformer.json", "w") as f:
    json.dump(results, f, indent=2)

np.save("transformer_test_probs.npy", test_probs)
print("done in", time.time() - t0, "s")
