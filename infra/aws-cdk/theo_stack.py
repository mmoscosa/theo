from aws_cdk import (
    Duration,
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_secretsmanager as secretsmanager,
    aws_iam as iam,
    aws_ecr as ecr,
    CfnOutput,
)
from constructs import Construct
import os

class TheoStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create VPC with public and private subnets
        vpc = ec2.Vpc(
            self, "TheoVPC",
            max_azs=2,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ]
        )

        # Create ECR repository for Docker images
        ecr_repo = ecr.Repository(
            self, "TheoRepository",
            repository_name="theo",
            image_scan_on_push=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    max_image_count=10,
                    description="Keep only 10 images"
                )
            ]
        )

        # Create ECS cluster
        cluster = ecs.Cluster(
            self, "TheoCluster",
            vpc=vpc,
            container_insights=True
        )

        # Create task execution role
        task_execution_role = iam.Role(
            self, "TheoTaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
            ]
        )

        # Create task role with permissions for Bedrock, Secrets Manager, etc.
        task_role = iam.Role(
            self, "TheoTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            inline_policies={
                "TheoPolicy": iam.PolicyDocument(
                    statements=[
                        # Bedrock permissions
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "bedrock:InvokeModel",
                                "bedrock:ListFoundationModels",
                                "bedrock:GetFoundationModel",
                                "bedrock:InvokeModelWithResponseStream"
                            ],
                            resources=["*"]
                        ),
                        # Secrets Manager permissions
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "secretsmanager:GetSecretValue",
                                "secretsmanager:DescribeSecret"
                            ],
                            resources=[f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:theo/*"]
                        ),
                        # Minimal logging permissions (Datadog handles most logging)
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "logs:CreateLogStream",
                                "logs:PutLogEvents"
                            ],
                            resources=["*"]
                        )
                    ]
                )
            }
        )

        # We use Datadog for logging instead of CloudWatch
        # No CloudWatch log group needed

        # Create Fargate task definition - Start minimal, scale as needed
        task_definition = ecs.FargateTaskDefinition(
            self, "TheoTaskDef",
            memory_limit_mib=1024,  # Start with 1GB RAM
            cpu=512,                # Start with 0.5 vCPU
            execution_role=task_execution_role,
            task_role=task_role
        )

        # Add container to task definition
        container = task_definition.add_container(
            "theo",
            image=ecs.ContainerImage.from_ecr_repository(ecr_repo, "latest"),
            # Using Datadog for logging instead of CloudWatch
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="theo"
            ),
            environment={
                "PORT": "8000",
                "AWS_REGION": self.region,
                "DD_LLMOBS_ENABLED": "1",
                "DD_TRACE_AGENTLESS": "true"
            },
            secrets={
                # These will be created in Secrets Manager
                "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "OpenAISecret", "theo/openai-api-key")
                ),
                "ANTHROPIC_API_KEY": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "AnthropicSecret", "theo/anthropic-api-key")
                ),
                "SLACK_BOT_TOKEN": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "SlackSecret", "theo/slack-bot-token")
                ),
                "SLACK_SIGNING_SECRET": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "SlackSigningSecret", "theo/slack-signing-secret")
                ),
                "DD_API_KEY": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "DatadogSecret", "theo/datadog-api-key")
                ),
                "GITHUB_TOKEN": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "GitHubSecret", "theo/github-token")
                ),
                "CONFLUENCE_API_TOKEN": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "ConfluenceSecret", "theo/confluence-api-token")
                ),
                "JIRA_API_TOKEN": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "JiraSecret", "theo/jira-api-token")
                ),
                "METABASE_USERNAME": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "MetabaseUserSecret", "theo/metabase-username")
                ),
                "METABASE_PASSWORD": ecs.Secret.from_secrets_manager(
                    secretsmanager.Secret.from_secret_name_v2(self, "MetabasePassSecret", "theo/metabase-password")
                )
            }
        )

        # Add port mapping
        container.add_port_mappings(
            ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP)
        )

        # Create the ALB Fargate service - Start with 1 task, auto-scale up
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "TheoService",
            cluster=cluster,
            cpu=512,               # Start minimal
            memory_limit_mib=1024, # Start minimal  
            desired_count=1,       # Start with just 1 task
            task_definition=task_definition,
            public_load_balancer=True,
            listener_port=443,
            protocol=elbv2.ApplicationProtocol.HTTPS,
            redirect_http=True,
            domain_zone=None,  # We'll set up domain later
            health_check_grace_period=Duration.seconds(300)
        )

        # Configure health check
        fargate_service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            health_check_timeout=Duration.seconds(10),
            health_check_interval=Duration.seconds(30),
            unhealthy_threshold_count=3
        )

        # Configure aggressive auto scaling - Start at 1, scale up quickly when needed
        scaling = fargate_service.service.auto_scale_task_count(
            min_capacity=1,        # Start with just 1 task
            max_capacity=5         # Scale up to 5 tasks max
        )

        # Scale based on CPU utilization - More aggressive scaling
        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=60,  # Scale earlier
            scale_in_cooldown=Duration.minutes(3),   # Scale down faster
            scale_out_cooldown=Duration.minutes(1)   # Scale up very quickly
        )

        # Scale based on memory utilization - More aggressive scaling  
        scaling.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=70,  # Scale earlier
            scale_in_cooldown=Duration.minutes(3),   # Scale down faster
            scale_out_cooldown=Duration.minutes(1)   # Scale up very quickly
        )

        # Scale based on request count - Perfect for Slack traffic bursts
        scaling.scale_on_request_count(
            "RequestScaling",
            requests_per_target=10,  # Scale up when >10 requests per task
            scale_in_cooldown=Duration.minutes(3),
            scale_out_cooldown=Duration.minutes(1)
        )

        # Outputs
        CfnOutput(
            self, "LoadBalancerDNS",
            value=fargate_service.load_balancer.load_balancer_dns_name,
            description="Load Balancer DNS name"
        )

        CfnOutput(
            self, "ECRRepositoryURI",
            value=ecr_repo.repository_uri,
            description="ECR Repository URI"
        )

        CfnOutput(
            self, "ECSClusterName",
            value=cluster.cluster_name,
            description="ECS Cluster Name"
        )

        CfnOutput(
            self, "ECSServiceName",
            value=fargate_service.service.service_name,
            description="ECS Service Name"
        ) 