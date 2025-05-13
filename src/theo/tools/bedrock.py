import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Real AWS Bedrock vector search integration
class BedrockClient:
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.code_doc_kb = os.getenv("BEDROCK_CODE_DOCUMENTATION_KB_ID")
        self.db_schema_kb = os.getenv("BEDROCK_DB_SCHEMA_QUERIES_KB_ID")
        self.general_kb = os.getenv("BEDROCK_GENERAL_KB_ID")
        self.client = boto3.client("bedrock-agent-runtime", region_name=self.region)

    def search_code_documentation(self, query):
        if not self.code_doc_kb:
            return "[Error] CodeDocumentation KB ID not set."
        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.code_doc_kb,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": 3}
                }
            )
            results = response.get("retrievalResults", [])
            if results:
                # Return the top result's content (or aggregate as needed)
                top = results[0]
                content = top.get("content", {}).get("text", "")
                return content or "[No content in top result]"
            else:
                return "[No results found in CodeDocumentation KB for query: '{}']".format(query)
        except (BotoCoreError, ClientError) as e:
            return f"[AWS Bedrock error] {str(e)}"
        except Exception as e:
            return f"[BedrockClient error] {str(e)}"

    def search_db_schema(self, query):
        if not self.db_schema_kb:
            return "[Error] DbSchemaAndQueries KB ID not set."
        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.db_schema_kb,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": 3}
                }
            )
            results = response.get("retrievalResults", [])
            if results:
                top = results[0]
                content = top.get("content", {}).get("text", "")
                return content or "[No content in top result]"
            else:
                return f"[No results found in DbSchemaAndQueries KB for query: '{query}']"
        except (BotoCoreError, ClientError) as e:
            return f"[AWS Bedrock error] {str(e)}"
        except Exception as e:
            return f"[BedrockClient error] {str(e)}"

    def search_general_knowledge(self, query):
        if not self.general_kb:
            return "[Error] GeneralKnowledge KB ID not set."
        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.general_kb,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": 3}
                }
            )
            results = response.get("retrievalResults", [])
            if results:
                top = results[0]
                content = top.get("content", {}).get("text", "")
                return content or "[No content in top result]"
            else:
                return f"[No results found in GeneralKnowledge KB for query: '{query}']"
        except (BotoCoreError, ClientError) as e:
            return f"[AWS Bedrock error] {str(e)}"
        except Exception as e:
            return f"[BedrockClient error] {str(e)}"

    def write_code_documentation(self, doc):
        # TODO: Implement write to CodeDocumentation collection
        pass

    def write_db_schema(self, record):
        # TODO: Implement write to DbSchemaAndQueries collection
        pass

    def write_general_knowledge(self, entry):
        # TODO: Implement write to GeneralKnowledge collection
        pass 