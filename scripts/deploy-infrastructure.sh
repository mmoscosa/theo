#!/bin/bash

# Deploy Theo infrastructure to AWS using CDK

set -e

echo "🚀 Deploying Theo infrastructure to AWS..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if CDK is installed
if ! command -v cdk &> /dev/null; then
    echo "❌ AWS CDK not installed. Please install it with 'npm install -g aws-cdk'"
    exit 1
fi

# Navigate to CDK directory
cd infra/aws-cdk

# Install Python dependencies
echo "📦 Installing CDK dependencies..."
pip install -r requirements.txt

# Bootstrap CDK (only needed once per account/region)
echo "🔧 Bootstrapping CDK..."
cdk bootstrap

# Deploy the stack
echo "🏗️ Deploying infrastructure..."
cdk deploy --require-approval never

echo "✅ Infrastructure deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create secrets in AWS Secrets Manager"
echo "2. Build and push Docker image to ECR"
echo "3. Update ECS service"
echo ""
echo "🔗 Useful commands:"
echo "  - View stack outputs: cdk ls"
echo "  - Check deployment: aws ecs describe-services --cluster TheoStack-TheoCluster --services TheoStack-TheoService" 