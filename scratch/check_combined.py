import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_train_p1 = pd.read_csv("data/train_phase1.csv")
df_train_p2 = pd.read_csv("data/train_phase2.csv")
df_eval = pd.read_csv("data/eval.csv")

# Combine them
df_combined = pd.concat([df_train_p1, df_train_p2], ignore_index=True)

X_train = df_combined.drop(columns=["target"])
y_train = df_combined["target"]
X_eval = df_eval.drop(columns=["target"])
y_eval = df_eval["target"]

# Test different parameter combinations on the combined dataset
configs = [
    {"n_estimators": 100, "max_depth": 15, "min_samples_split": 2, "random_state": 42},
    {"n_estimators": 256, "max_depth": 20, "min_samples_split": 2, "random_state": 42},
    {"n_estimators": 100, "max_depth": 15, "min_samples_split": 2, "random_state": 1083},
    {"n_estimators": 256, "max_depth": 20, "min_samples_split": 2, "random_state": 1083},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 5, "random_state": 1048},
]

for cfg in configs:
    seed = cfg.get("random_state", 42)
    params = {k: v for k, v in cfg.items() if k != "random_state"}
    model = RandomForestClassifier(**params, random_state=seed)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_eval, model.predict(X_eval))
    print(f"Combined accuracy: {acc:.4f} with params {cfg}")
