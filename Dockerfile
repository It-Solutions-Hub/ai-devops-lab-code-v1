# ai-devops-lab:1.0
# Single image used by every KFP component in the lab, so learners never
# install or debug dependencies during their 60 minutes.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    scikit-learn==1.9.0 \
    pandas \
    mlflow==3.15.2

WORKDIR /opt/lab
COPY scripts/ /opt/lab/scripts/
COPY dataset/make_dataset.py /opt/lab/dataset/make_dataset.py

# Bake the fixed, deterministic dataset into the image at build time.
RUN python /opt/lab/dataset/make_dataset.py

# NOTE: MLFLOW_TRACKING_URI is intentionally NOT set here.
# minikube's node IP / docker-network gateway is only known at facilitator
# setup time (and can change between VMs/cohorts), so it is injected as a
# per-task env var by pipeline/lab_pipeline.py at compile time instead of
# being baked into this image. See the prerequisites guide, section 4.6/4.8.
# This also means the image never needs rebuilding if the gateway IP changes
# -- only a pipeline recompile.
