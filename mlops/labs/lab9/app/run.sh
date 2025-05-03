#!/usr/bin/env bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────
# run.sh – build & deploy reddit‑app locally on Minikube
# ─────────────────────────────────────────────────────────────

# 1) Start/ensure Minikube is running
minikube start

# 2) Use Minikube’s internal Docker daemon
eval "$(minikube docker-env)"

# 3) Build the image (replace paths / tag if you need to)
docker build -t reddit-app:latest .

# 4) Apply Kubernetes manifests
kubectl apply -f reddit-deployment.yaml
kubectl apply -f reddit-service.yaml

# 5) Scale deployment to 2 replicas
# kubectl scale deployment reddit-app-deployment --replicas=2

# minikube service reddit-app-service --url