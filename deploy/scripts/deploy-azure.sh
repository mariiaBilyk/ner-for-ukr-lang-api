#!/usr/bin/env bash
# scripts/deploy-azure.sh
#
# One-time Azure resource setup for NER API on Container Apps (free tier).
#
# Azure Container Apps free tier:
#   - 180,000 vCPU-seconds/month
#   - 360,000 GB-seconds/month
#   - Scales to 0 when idle (no charges when not used)
#   - Perfect for a pet project with sporadic traffic
#
# IMPORTANT: Azure Container Apps doesn't support GPU.
#   Ollama must run on a separate VPS (or your local machine with ngrok).
#   This deploys ONLY the FastAPI service; Ollama is external.
#
#
# Prerequisites:
#   brew install azure-cli   (macOS)
#   az login
#   az extension add --name containerapp --upgrade

set -euo pipefail

# ── Load .env so all defaults come from one place ────────────────────────
ENV_FILE="$(dirname "$0")/../../.env"
if [ -f "${ENV_FILE}" ]; then
    set -o allexport
    source "${ENV_FILE}"
    set +o allexport
    echo "Loaded config from .env"
else
    echo "ERROR: .env not found at ${ENV_FILE}"
    echo "Copy .env.example to .env and fill in your values first."
    exit 1
fi

# ── Configuration — change these ────────────────────────────────────────
RESOURCE_GROUP="ner-rg"
LOCATION="westeurope"
CONTAINER_APP_ENV="ner-env"
APP_NAME="ner-api"

DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:?Set DOCKERHUB_USERNAME in .env}"
IMAGE="${DOCKERHUB_USERNAME}/docker-ner-api:latest"

OLLAMA_HOST="${OLLAMA_HOST:?Set OLLAMA_HOST in .env}"
# ────────────────────────────────────────────────────────────────────────


echo "=== Azure Container Apps Deployment ==="

# ── 1. Create resource group ──────────────────────────────────────────────
echo "Creating resource group..."
az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${LOCATION}"


# ── 2. Create Container Apps environment ─────────────────────────────────
echo "Creating Container Apps environment..."
az containerapp env create \
    --name "${CONTAINER_APP_ENV}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}"


# ── 3. Create the Container App ──────────────────────────────────────────
# A placeholder image is used for the initial create because:
#   - az containerapp create requires a runnable image immediately (it schedules
#     a revision on the spot; there is no "empty" state in the API)
#   - The real image (${IMAGE}) does not exist on Docker Hub yet at infra-setup
#     time — CI/CD builds and pushes it on the first commit to main
#   - CI/CD uses `az containerapp update --image` to replace this placeholder
#
# If your Docker Hub repo is private, add to the create command:
#   --registry-server index.docker.io
#   --registry-username "${DOCKERHUB_USERNAME}"
#   --registry-password "${DOCKERHUB_TOKEN}"

echo "Creating Container App..."
az containerapp create \
    --name "${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --environment "${CONTAINER_APP_ENV}" \
    --image "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest" \
    --target-port 8000 \
    --ingress external \
    --min-replicas 0 \
    --max-replicas 3 \
    --cpu 0.5 \
    --memory 1.0Gi \
    --env-vars \
        "OLLAMA_HOST=${OLLAMA_HOST}" \
        "INFERENCE_BACKEND=ollama" \
        "OLLAMA_MODEL=${OLLAMA_MODEL}" \
        "PROMPT_NAME=${PROMPT_NAME:-ner}" \
        "PROMPT_VERSION=${PROMPT_VERSION:-}" \
        "EXTRACTION_STRATEGY=${EXTRACTION_STRATEGY:-simple}" \
        "NER_AGENT_MAX_ATTEMPTS=${NER_AGENT_MAX_ATTEMPTS:-3}"


# ── 4. Get the app URL ───────────────────────────────────────────────────
APP_URL=$(az containerapp show \
    --name "${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query "properties.configuration.ingress.fqdn" \
    --output tsv)

echo ""
echo "=== Deployment complete ==="
echo "App URL:    https://${APP_URL}"
echo "Health:     https://${APP_URL}/health"
echo "NER API:    POST https://${APP_URL}/ner"
echo ""
echo "Test it:"
echo "  curl -X POST https://${APP_URL}/ner \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\": \"Петро працює у компанії SoftServe у Львові\"}'"


# ── 5. Create service principal for GitHub Actions CI/CD ─────────────────

echo ""
echo "Creating service principal for GitHub Actions CI/CD..."
SUBSCRIPTION_ID=$(az account show --query id --output tsv)
SP_JSON=$(az ad sp create-for-rbac \
    --name "ner-api-github-actions" \
    --role contributor \
    --scopes "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}")

AZURE_CREDENTIALS=$(python3 - <<EOF
import json, sys
sp = json.loads('''${SP_JSON}''')
print(json.dumps({
    "clientId":       sp["appId"],
    "clientSecret":   sp["password"],
    "tenantId":       sp["tenant"],
    "subscriptionId": "${SUBSCRIPTION_ID}",
}, indent=2))
EOF
)

echo ""
echo "Add these to GitHub Secrets (repo → Settings → Secrets → Actions):"
echo "  AZURE_CREDENTIALS  = (JSON below)"
echo "${AZURE_CREDENTIALS}"
echo ""
