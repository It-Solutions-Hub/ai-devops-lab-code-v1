"""
Kubeflow pipeline / MLflow UI action: sets a registry alias on a model
version. Used for BOTH promotion ("champion" -> new version) and rollback
("champion" -> previous version) -- the operation is identical, only the
target version differs, which mirrors how MLflow aliasing actually works.
"""
import argparse
from mlflow import MlflowClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="churn-classifier")
    p.add_argument("--version", required=True)
    p.add_argument("--alias", default="champion")
    args = p.parse_args()

    client = MlflowClient()
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    print(f"ALIAS_SET alias={args.alias} -> {args.model_name} v{args.version}")


if __name__ == "__main__":
    main()
