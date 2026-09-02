#!/usr/bin/env bash
# Facilitator-only. Run once per VM/cohort, from the ai-devops-lab-code
# directory (or anywhere -- it locates itself). Assumes docker, kubectl,
# minikube, git and python3.10 are ALREADY installed on this VM (per the
# prerequisites guide) -- this script does not install or upgrade any of
# them. See prerequisites guide section 4 for the manual, step-by-step
# version of everything below.
set -euo pipefail

PROFILE="ai-devops-lab"
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HOME}/ml-flow/mlflow-venv"

echo "== 0. Confirm required tools are present (no install/upgrade performed) =="
docker --version
kubectl version --client
minikube version
python3.10 --version
git --version

echo
echo "== 1. Start (or reuse) the minikube cluster, profile: ${PROFILE} =="
if minikube status -p "${PROFILE}" >/dev/null 2>&1; then
  echo "Profile ${PROFILE} already running -- leaving it as-is."
else
  minikube start -p "${PROFILE}" \
    --driver=docker \
    --cpus=4 \
    --memory=8000mb \
    --disk-size=40g \
    --kubernetes-version=v1.35.1
fi
kubectl config use-context "${PROFILE}"
kubectl get nodes

echo
echo "== 2. Deploy Kubeflow Pipelines (standalone) =="
export PIPELINE_VERSION=2.2.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=${PIPELINE_VERSION}"
echo "Waiting for KFP pods (can take several minutes on first run)..."
kubectl wait pods -n kubeflow -l application-crd-id=kubeflow-pipelines --for condition=Ready --timeout=15m

echo
echo "== 3. Set up the MLflow venv (python3.10, reusing ${VENV_DIR} if present) =="
if [ ! -d "${VENV_DIR}" ]; then
  python3.10 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install --quiet mlflow==3.15.2 kfp==2.17.0
sudo mkdir -p /var/lib/mlflow/artifacts

echo
echo "== 4. Install and start the MLflow systemd service =="
sudo tee /etc/systemd/system/mlflow.service > /dev/null <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=${USER}
ExecStart=${VENV_DIR}/bin/mlflow server \\
  --backend-store-uri sqlite:////var/lib/mlflow/mlflow.db \\
  --default-artifact-root /var/lib/mlflow/artifacts \\
  --host 0.0.0.0 --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now mlflow
sleep 3
curl -sf http://localhost:5000 > /dev/null && echo "MLflow server is up on :5000" \
  || echo "WARNING: MLflow did not respond -- check: sudo systemctl status mlflow"

echo
echo "== 5. Determine the MLflow URL reachable from *inside* the minikube cluster =="
GATEWAY_IP="$(docker network inspect "${PROFILE}" 2>/dev/null | grep -m1 '"Gateway"' | sed -E 's/.*"Gateway": ?"([0-9.]+)".*/\1/')"
if [ -z "${GATEWAY_IP}" ]; then
  echo "Could not read the ${PROFILE} docker network gateway -- falling back to host.minikube.internal"
  MLFLOW_URL="http://host.minikube.internal:5000"
else
  MLFLOW_URL="http://${GATEWAY_IP}:5000"
fi
echo "Pods inside minikube will reach MLflow at: ${MLFLOW_URL}"

echo
echo "== 6. Build the lab image and load it into minikube (no external pulls at session time) =="
cd "${LAB_DIR}"
docker build -t ai-devops-lab:1.0 .
minikube image load ai-devops-lab:1.0 -p "${PROFILE}"

echo
echo "== 7. Confirm the dataset baked in correctly =="
docker run --rm ai-devops-lab:1.0 python -c \
  "import pandas as pd; print(pd.read_csv('/opt/lab/dataset/churn_dataset.csv').shape)"
echo "(expected: (4000, 21))"

echo
echo "== 8. Point both pipelines at this VM's MLflow URL and compile them =="
sed -i "s#^MLFLOW_TRACKING_URI = .*#MLFLOW_TRACKING_URI = \"${MLFLOW_URL}\"#" "${LAB_DIR}/pipeline/lab_pipeline.py"
sed -i "s#^MLFLOW_TRACKING_URI = .*#MLFLOW_TRACKING_URI = \"${MLFLOW_URL}\"#" "${LAB_DIR}/pipeline/validate_only_pipeline.py"
"${VENV_DIR}/bin/python" "${LAB_DIR}/pipeline/lab_pipeline.py"
"${VENV_DIR}/bin/python" "${LAB_DIR}/pipeline/validate_only_pipeline.py"

echo
echo "================================================================"
echo "Setup complete. Remaining MANUAL steps:"
echo ""
echo "1) In a separate terminal/tmux pane, expose the KFP UI to the network:"
echo "     kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 --address 0.0.0.0"
echo ""
echo "2) Open http://<this-vm-ip>:8080 -> Pipelines -> Upload pipeline"
echo "   -> select ${LAB_DIR}/pipeline/lab_pipeline.yaml"
echo "   -> name it 'AI DevOps Model Lifecycle'"
echo "   -> then Upload pipeline again, select"
echo "      ${LAB_DIR}/pipeline/validate_only_pipeline.yaml"
echo "   -> name it 'AI DevOps Production Validate'"
echo "   (used to re-check validation against the current champion in"
echo "    Phases 9/10 without retraining -- see the activity guide)"
echo ""
echo "3) Seed the baseline exploration run (prerequisites guide section 4.9):"
echo "   Experiments -> AI DevOps Model Lifecycle -> Create run"
echo "   label=v1, n_estimators=100, max_depth=6, eval_threshold=0.90, promote_alias=baseline-demo"
echo ""
echo "4) Run the golden-path verification in the prerequisites guide, section 5,"
echo "   before letting any learner start the timed activity."
echo "================================================================"
