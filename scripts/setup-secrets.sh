#!/bin/bash

# Setup AWS Secrets Manager secrets for Theo

set -e

echo "🔐 Setting up AWS Secrets Manager secrets for Theo..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Function to create or update secret
create_secret() {
    local secret_name=$1
    local description=$2
    
    echo "📝 Setting up secret: $secret_name"
    echo "Please enter the value for $description:"
    read -s secret_value
    
    # Try to create the secret, if it exists, update it
    if aws secretsmanager describe-secret --secret-id "$secret_name" > /dev/null 2>&1; then
        echo "🔄 Updating existing secret: $secret_name"
        aws secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "$secret_value"
    else
        echo "🆕 Creating new secret: $secret_name"
        aws secretsmanager create-secret \
            --name "$secret_name" \
            --description "$description" \
            --secret-string "$secret_value"
    fi
    
    echo "✅ Secret $secret_name configured"
    echo ""
}

echo "This script will help you set up all the required secrets for Theo."
echo "You'll be prompted to enter each secret value securely."
echo ""

# Create all required secrets
create_secret "theo/openai-api-key" "OpenAI API Key"
create_secret "theo/anthropic-api-key" "Anthropic API Key"
create_secret "theo/slack-bot-token" "Slack Bot Token"
create_secret "theo/slack-signing-secret" "Slack Signing Secret"
create_secret "theo/datadog-api-key" "Datadog API Key"
create_secret "theo/github-token" "GitHub Token"
create_secret "theo/confluence-api-token" "Confluence API Token"
create_secret "theo/jira-api-token" "Jira API Token"
create_secret "theo/metabase-username" "Metabase Username"
create_secret "theo/metabase-password" "Metabase Password"

echo "🎉 All secrets have been configured in AWS Secrets Manager!"
echo ""
echo "📋 Next steps:"
echo "1. Deploy your infrastructure: ./scripts/deploy-infrastructure.sh"
echo "2. Build and push Docker image: ./scripts/deploy-docker.sh"
echo ""
echo "🔗 View secrets in AWS Console:"
echo "https://console.aws.amazon.com/secretsmanager/" 