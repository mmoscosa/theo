#!/bin/bash

set -e

# Build and start Docker Compose services in detached mode
echo "🚀 Building and starting Theo services with Docker Compose..."
docker compose -f docker-compose.dev.yml up --build

echo "✅ Theo is up and running!"
echo ""
echo "To view logs:"
echo "  docker compose -f docker-compose.dev.yml logs -f"
echo ""
echo "To run tests inside the theo container:"
echo "  docker compose -f docker-compose.dev.yml exec theo python test_llm.py"
echo ""
echo "To stop all services:"
echo "  docker compose -f docker-compose.dev.yml down"
echo ""
echo "Visit http://localhost:8000 (or your mapped port) to access the FastAPI app." 