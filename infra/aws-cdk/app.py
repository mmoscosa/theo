#!/usr/bin/env python3
import os
import aws_cdk as cdk
from theo_stack import TheoStack

app = cdk.App()

# Environment configuration
env_prod = cdk.Environment(
    account=os.getenv('CDK_DEFAULT_ACCOUNT'),
    region=os.getenv('CDK_DEFAULT_REGION', 'us-east-1')
)

# Create the Theo production stack
TheoStack(app, "TheoStack",
    env=env_prod,
    description="Theo AI Assistant Infrastructure"
)

app.synth() 