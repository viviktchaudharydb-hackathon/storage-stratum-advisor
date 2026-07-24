#!/usr/bin/env bash
# Deploy Storage Advisor to Cloud Run (fully GCP-native, no API keys)
set -euo pipefail

PROJECT_ID="${1:?Usage: ./deploy.sh <PROJECT_ID> [REGION]}"
REGION="${2:-asia-south1}"   # Mumbai
SERVICE="infra-advisor"

gcloud config set project "$PROJECT_ID"

# One-time: enable required APIs
gcloud services enable run.googleapis.com aiplatform.googleapis.com \
    cloudbuild.googleapis.com artifactregistry.googleapis.com

# Build + deploy from source (Cloud Build handles the container)
# Both runtime SA and build SA are set to workload@ — the only SA our
# group has actAs on (default compute SA and infrastructure@ are locked down)
gcloud run deploy "$SERVICE" \
    --source . \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --allow-unauthenticated \
    --memory 1Gi \
    --service-account "workload@${PROJECT_ID}.iam.gserviceaccount.com" \
    --build-service-account "projects/${PROJECT_ID}/serviceAccounts/workload@${PROJECT_ID}.iam.gserviceaccount.com" \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},GEMINI_MODEL=gemini-2.5-flash"

# Grant the runtime SA permission to call Vertex AI (one-time).
# Non-fatal: our group may not have setIamPolicy; workload@ likely
# already has Vertex AI access pre-provisioned.
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:workload@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/aiplatform.user" --condition=None \
    || echo "IAM grant skipped (no permission) — workload SA likely already has Vertex AI access"

echo "Done. URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
