"""
Kubeflow pipeline component: trains a candidate model and logs it to MLflow.
Deterministic: fixed dataset, fixed train/test split seed, fixed model seed.
Only n_estimators / max_depth are exposed as learner-configurable parameters.
"""
import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

DATASET_PATH = os.environ.get("LAB_DATASET_PATH", "/opt/lab/dataset/churn_dataset.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, choices=["v1", "v2"], help="Lab-tracking label, e.g. v1 or v2")
    p.add_argument("--n-estimators", type=int, required=True)
    p.add_argument("--max-depth", type=int, required=True)
    p.add_argument("--experiment", default="churn-model-training")
    args = p.parse_args()

    mlflow.set_experiment(args.experiment)

    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    with mlflow.start_run(run_name=f"candidate-{args.label}") as run:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.set_tag("lab_version", args.label)
        mlflow.sklearn.log_model(clf, artifact_path="model")

        print(f"RUN_ID={run.info.run_id}")
        print(f"ACCURACY={acc:.4f}")
        print(f"F1={f1:.4f}")


if __name__ == "__main__":
    main()
