"""
Kubeflow pipeline component: registers a trained run's model into the
MLflow Model Registry and copies the lab_version tag onto the new
registered model version.
"""
import argparse
import mlflow
from mlflow import MlflowClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--model-name", default="churn-classifier")
    args = p.parse_args()

    client = MlflowClient()
    run = client.get_run(args.run_id)
    lab_version = run.data.tags.get("lab_version", "unknown")

    model_uri = f"runs:/{args.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    client.set_model_version_tag(args.model_name, mv.version, "lab_version", lab_version)

    print(f"REGISTERED_NAME={args.model_name}")
    print(f"REGISTERED_VERSION={mv.version}")
    print(f"LAB_VERSION={lab_version}")


if __name__ == "__main__":
    main()
