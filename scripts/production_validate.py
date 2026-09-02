"""
Kubeflow pipeline component: production validation.

This is deliberately a SCRIPTED / mocked validation, not a live inference
test, so every learner sees the identical result regardless of machine
performance or randomness:

  - whichever model version currently holds the "champion" alias and is
    tagged lab_version=v2  -> validation FAILS
  - any other champion (v1, or after rollback) -> validation PASSES

This keeps the assessment focused on DevOps mechanics (gates, promotion,
rollback) rather than on real-world ML QA.
"""
import argparse
import sys
from mlflow import MlflowClient

# Facilitator-controlled scenario table. Extend/change here to script
# different failure scenarios without touching the pipeline definition.
FORCED_FAILURES = {"v2"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="churn-classifier")
    p.add_argument("--alias", default="champion")
    args = p.parse_args()

    client = MlflowClient()
    mv = client.get_model_version_by_alias(args.model_name, args.alias)
    lab_version = mv.tags.get("lab_version", "unknown")

    print(f"Validating champion: {args.model_name} v{mv.version} (lab_version={lab_version})")

    if lab_version in FORCED_FAILURES:
        print("PRODUCTION_VALIDATION=FAILED")
        sys.exit(1)
    else:
        print("PRODUCTION_VALIDATION=PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
