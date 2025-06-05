# scripts/deploy.sh
#!/bin/bash

# Deployment script for Kubernetes

set -e

# Configuration
NAMESPACE=${NAMESPACE:-"rag"}
KUBECONFIG=${KUBECONFIG:-"~/.kube/config"}

echo "Deploying RAG System to Kubernetes..."

# Create namespace if it doesn't exist
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Apply all manifests
kubectl apply -f k8s/ -n ${NAMESPACE}

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/neo4j -n ${NAMESPACE}
kubectl wait --for=condition=available --timeout=300s deployment/redis -n ${NAMESPACE}
kubectl wait --for=condition=available --timeout=600s deployment/rag-system -n ${NAMESPACE}

echo "Deployment completed successfully!"

# Get service information
kubectl get services -n ${NAMESPACE}
