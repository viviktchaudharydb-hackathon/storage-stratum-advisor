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
gcloud run deploy "$SERVICE" \
    --source . \
    --region "$REGION" \
    --allow-unauthenticated \
    --memory 1Gi \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},GEMINI_MODEL=gemini-2.5-flash"

# Grant the service's runtime SA permission to call Vertex AI (one-time)
SA=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(spec.template.spec.serviceAccountName)')
SA=${SA:-"$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com"}
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:${SA}" \
    --role "roles/aiplatform.user" --condition=None

echo "Done. URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
