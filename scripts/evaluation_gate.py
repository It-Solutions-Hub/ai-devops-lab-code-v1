"""
Kubeflow pipeline component: evaluation gate.
Reads the accuracy metric of a given MLflow run and compares it to a
learner-configured threshold. Fails the pipeline step if the gate fails.
"""
import argparse
import sys
import mlflow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--threshold", type=float, required=True)
    args = p.parse_args()

    run = mlflow.get_run(args.run_id)
    acc = run.data.metrics.get("accuracy")
    if acc is None:
        print("GATE=ERROR no accuracy metric found on run")
        sys.exit(2)

    if acc >= args.threshold:
        print(f"GATE=PASS accuracy={acc:.4f} threshold={args.threshold}")
        sys.exit(0)
    else:
        print(f"GATE=FAIL accuracy={acc:.4f} threshold={args.threshold}")
        sys.exit(1)


if __name__ == "__main__":
    main()
