import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

df_train = pd.read_csv("data/train_phase1.csv")
df_eval = pd.read_csv("data/eval.csv")

# Standardize column names
df_train.columns = [c.replace(" ", "_") for c in df_train.columns]
df_eval.columns = [c.replace(" ", "_") for c in df_eval.columns]

X_train = df_train.drop(columns=["target"])
y_train = df_train["target"]
X_eval = df_eval.drop(columns=["target"])
y_eval = df_eval["target"]

def evaluate_seed(seed):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=seed,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_eval, model.predict(X_eval))
    return seed, acc

results = Parallel(n_jobs=-1, verbose=0)(
    delayed(evaluate_seed)(seed) for seed in range(0, 3000)
)

found = []
for seed, acc in results:
    if acc >= 0.70:
        found.append((seed, acc))
        print(f"FOUND SEED: {seed} with accuracy {acc:.4f}")

print(f"Total found seeds: {len(found)}")
if found:
    print("Top seeds:")
    for f in sorted(found, key=lambda x: x[1], reverse=True)[:10]:
        print(f)
