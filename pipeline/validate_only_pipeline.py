"""
Standalone "validate only" pipeline for the AI-Native DevOps lab.

lab_pipeline.py bundles production_validate_op as the last step of a full
train -> gate -> register -> promote -> validate run. That's fine for
Phases 3 and 6 (candidate-v1/v2), but Phase 9 (re-check v2) and Phase 10
(re-check after rollback to v1) need to validate whichever model version
currently holds the "champion" alias WITHOUT retraining a new candidate.
This pipeline exposes just that one step.

Facilitator: upload this as its own pipeline (name it e.g. "AI DevOps
Production Validate") alongside "AI DevOps Model Lifecycle". Keep
MLFLOW_TRACKING_URI here in sync with pipeline/lab_pipeline.py -- both
must point at the same MLflow server.
"""
from kfp import dsl, compiler

LAB_IMAGE = "ai-devops-lab:1.0"

# Facilitator: keep this in sync with pipeline/lab_pipeline.py's constant
# of the same name (see that file's comment for how to determine it).
MLFLOW_TRACKING_URI = "http://192.168.49.1:5000"


@dsl.component(base_image=LAB_IMAGE)
def production_validate_op(alias: str = "champion"):
    import subprocess
    subprocess.run(
        ["python", "/opt/lab/scripts/production_validate.py", "--alias", alias],
        check=True,
    )


@dsl.pipeline(name="ai-devops-production-validate")
def validate_only_pipeline(alias: str = "champion"):
    validate_task = production_validate_op(alias=alias)
    validate_task.set_env_variable(name="MLFLOW_TRACKING_URI", value=MLFLOW_TRACKING_URI)
    # Same reasoning as lab_pipeline.py: alias is the only input, and it's
    # "champion" on every call, so this MUST run fresh every time or it
    # will just replay whatever the last validation result happened to be.
    validate_task.set_caching_options(False)


if __name__ == "__main__":
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_only_pipeline.yaml")
    compiler.Compiler().compile(validate_only_pipeline, out_path)
    print(f"Compiled successfully -> {out_path}")
    print(f"MLFLOW_TRACKING_URI baked into this compile: {MLFLOW_TRACKING_URI}")
