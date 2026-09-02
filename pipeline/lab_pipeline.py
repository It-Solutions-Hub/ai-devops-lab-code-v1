"""
Kubeflow Pipelines (KFP SDK v2) definition for the AI-Native DevOps lab.

All components share one pre-built image (see prerequisites guide:
"ai-devops-lab:1.0") that already contains the dataset and the scripts in
scripts/. Five parameters are learner-configurable per run: label,
n_estimators, max_depth, eval_threshold, promote_alias.

MLFLOW_TRACKING_URI is a facilitator-only setting (not a learner-visible
run parameter). It is injected as an environment variable on every task
at *compile* time via set_env_variable(), rather than baked into the
Docker image, because the correct URL depends on the facilitator's
minikube setup (the docker-network gateway IP -- see prerequisites guide
section 4.6) and can differ between VMs/cohorts. Update the constant
below and re-run this file to recompile whenever that IP changes; no
image rebuild is needed.
"""
import os
from kfp import dsl, compiler

LAB_IMAGE = "ai-devops-lab:1.0"  # built in the prerequisites/setup guide

# Facilitator: set this to the MLflow server URL reachable from *inside*
# the minikube cluster. For the docker driver this is the gateway IP of
# the "ai-devops-lab" docker network (see prerequisites guide 4.6), e.g.:
#   docker network inspect ai-devops-lab | grep Gateway
# The facilitator setup script (facilitator/setup_lab.sh) fills this in
# automatically before compiling.
MLFLOW_TRACKING_URI = "http://192.168.49.1:5000"


def _with_mlflow_env(task: dsl.PipelineTask) -> dsl.PipelineTask:
    # Caching disabled on every task: several components (most notably
    # production_validate_op, whose only input is the "champion" alias
    # string) take the SAME inputs on every run regardless of which model
    # version is actually behind that alias. With caching on, KFP treats
    # those as identical executions and replays a stale result -- e.g. v2's
    # validation silently reusing v1's PASSED result instead of re-running
    # and correctly FAILING. Disabling caching trades a few seconds of
    # re-execution for correctness, which matters far more in this lab.
    return (
        task
        .set_env_variable(name="MLFLOW_TRACKING_URI", value=MLFLOW_TRACKING_URI)
        .set_caching_options(False)
    )


@dsl.component(base_image=LAB_IMAGE)
def train_op(label: str, n_estimators: int, max_depth: int) -> str:
    import subprocess
    out = subprocess.run(
        ["python", "/opt/lab/scripts/train.py",
         "--label", label,
         "--n-estimators", str(n_estimators),
         "--max-depth", str(max_depth)],
        capture_output=True, text=True, check=True,
    )
    print(out.stdout)
    run_id = [l for l in out.stdout.splitlines() if l.startswith("RUN_ID=")][0].split("=")[1]
    return run_id


@dsl.component(base_image=LAB_IMAGE)
def evaluation_gate_op(run_id: str, threshold: float):
    import subprocess
    subprocess.run(
        ["python", "/opt/lab/scripts/evaluation_gate.py",
         "--run-id", run_id, "--threshold", str(threshold)],
        check=True,
    )


@dsl.component(base_image=LAB_IMAGE)
def register_model_op(run_id: str) -> str:
    import subprocess
    out = subprocess.run(
        ["python", "/opt/lab/scripts/register_model.py", "--run-id", run_id],
        capture_output=True, text=True, check=True,
    )
    print(out.stdout)
    version = [l for l in out.stdout.splitlines() if l.startswith("REGISTERED_VERSION=")][0].split("=")[1]
    return version


@dsl.component(base_image=LAB_IMAGE)
def set_alias_op(version: str, alias: str = "champion"):
    import subprocess
    subprocess.run(
        ["python", "/opt/lab/scripts/set_alias.py", "--version", version, "--alias", alias],
        check=True,
    )


@dsl.component(base_image=LAB_IMAGE)
def production_validate_op(alias: str = "champion"):
    import subprocess
    subprocess.run(
        ["python", "/opt/lab/scripts/production_validate.py", "--alias", alias],
        check=True,
    )


@dsl.pipeline(name="ai-devops-model-lifecycle")
def lab_pipeline(
    label: str = "v1",
    n_estimators: int = 150,
    max_depth: int = 8,
    eval_threshold: float = 0.90,
    promote_alias: str = "champion",
):
    train_task = _with_mlflow_env(
        train_op(label=label, n_estimators=n_estimators, max_depth=max_depth)
    )

    gate_task = _with_mlflow_env(
        evaluation_gate_op(run_id=train_task.output, threshold=eval_threshold)
    )
    gate_task.after(train_task)

    register_task = _with_mlflow_env(register_model_op(run_id=train_task.output))
    register_task.after(gate_task)

    promote_task = _with_mlflow_env(
        set_alias_op(version=register_task.output, alias=promote_alias)
    )
    promote_task.after(register_task)

    validate_task = _with_mlflow_env(production_validate_op(alias=promote_alias))
    validate_task.after(promote_task)


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_pipeline.yaml")
    compiler.Compiler().compile(lab_pipeline, out_path)
    print(f"Compiled successfully -> {out_path}")
    print(f"MLFLOW_TRACKING_URI baked into this compile: {MLFLOW_TRACKING_URI}")
