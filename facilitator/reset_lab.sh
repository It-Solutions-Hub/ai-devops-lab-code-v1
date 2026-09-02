#!/usr/bin/env bash
# Facilitator-only. Clears MLflow state between learners/cohorts.
# Does NOT touch the minikube cluster, the uploaded KFP pipeline definition,
# or any installed tooling -- only MLflow's runs/registry/artifacts.
set -euo pipefail

echo "This will DELETE all MLflow runs, registered models, and artifacts."
read -p "Continue? [y/N] " CONFIRM
if [ "${CONFIRM}" != "y" ] && [ "${CONFIRM}" != "Y" ]; then
  echo "Aborted."
  exit 0
fi

echo "Stopping MLflow..."
sudo systemctl stop mlflow

echo "Clearing MLflow database and artifacts..."
sudo rm -f /var/lib/mlflow/mlflow.db
sudo rm -rf /var/lib/mlflow/artifacts/*

echo "Restarting MLflow..."
sudo systemctl start mlflow
sleep 3
curl -sf http://localhost:5000 > /dev/null && echo "MLflow reset and back up on :5000" \
  || echo "WARNING: MLflow did not respond after restart -- check: sudo systemctl status mlflow"

echo
echo "Reminder (manual, in the KFP UI):"
echo "  - Delete any runs under 'AI DevOps Model Lifecycle' from the previous cohort"
echo "    (the pipeline definition itself can stay uploaded)."
echo "  - Re-seed the baseline exploration run (prerequisites guide section 4.9)."
