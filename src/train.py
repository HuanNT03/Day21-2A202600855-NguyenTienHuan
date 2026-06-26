import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


class WineFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            # Standardize column names (replace spaces with underscores)
            X_df.columns = [c.replace(" ", "_") for c in X_df.columns]
            cols = [
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol", "wine_type"
            ]
            for col in cols:
                if col not in X_df.columns:
                    space_col = col.replace("_", " ")
                    if space_col in X.columns:
                        X_df[col] = X[space_col]
                    else:
                        X_df[col] = 0.0
            X_arr = X_df[cols].to_numpy()
        else:
            X_arr = np.asarray(X)

        fixed_acidity = X_arr[:, 0]
        volatile_acidity = X_arr[:, 1]
        citric_acid = X_arr[:, 2]
        free_sulfur_dioxide = X_arr[:, 5]
        total_sulfur_dioxide = X_arr[:, 6]
        density = X_arr[:, 7]
        sulphates = X_arr[:, 9]
        alcohol = X_arr[:, 10]

        # Engineered features
        sulfur_ratio = free_sulfur_dioxide / (total_sulfur_dioxide + 1e-5)
        acid_ratio = volatile_acidity / (fixed_acidity + 1e-5)
        alc_density = alcohol / density
        alc_sulphates = alcohol * sulphates
        vol_alc = volatile_acidity * alcohol
        total_acid = fixed_acidity + volatile_acidity + citric_acid

        new_features = np.column_stack([
            X_arr,
            sulfur_ratio,
            acid_ratio,
            alc_density,
            alc_sulphates,
            vol_alc,
            total_acid
        ])
        return new_features


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Đảm bảo thư mục đầu ra tồn tại trước khi log
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    mlflow.set_experiment("WineQuality")
    with mlflow.start_run():

        mlflow.log_params(params)

        seed = params.pop("random_state", 42)
        rf = RandomForestClassifier(**params, random_state=seed)
        
        # Build scikit-learn Pipeline
        model = Pipeline([
            ("extractor", WineFeatureExtractor()),
            ("classifier", rf)
        ])
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
