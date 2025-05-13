#!/bin/bash
set -e

# Load environment variables from .env
echo "Loading environment variables from .env..."
export $(grep -v '^#' /app/.env | xargs)

# Print Datadog related environment variables for debugging
echo "DD_TRACE_AGENTLESS: $DD_TRACE_AGENTLESS"
echo "DD_API_KEY: ${DD_API_KEY:0:5}****"
echo "DD_SITE: $DD_SITE"
echo "DD_LLMOBS_ML_APP: $DD_LLMOBS_ML_APP"

# Start the application with ddtrace-run
exec ddtrace-run uvicorn src.theo.api:app --host 0.0.0.0 --port 8000 