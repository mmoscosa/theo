import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Real AWS Bedrock vector search integration
class BedrockClient:
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.code_doc_kb = os.getenv("BEDROCK_CORE_DOCUMENTATION_KB_ID")
        self.db_schema_kb = os.getenv("BEDROCK_DB_SCHEMA_QUERIES_KB_ID")
        self.general_kb = os.getenv("BEDROCK_GENERAL_KB_ID")
        self.client = boto3.client("bedrock-agent-runtime", region_name=self.region)

    def search_code_documentation(self, query):
        if not self.code_doc_kb:
            return {"content": "[Error] CodeDocumentation KB ID not set.", "sources": []}
        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.code_doc_kb,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": 3}
                }
            )
            results = response.get("retrievalResults", [])
            print(f"[DEBUG] Bedrock search_code_documentation: Found {len(results)} results")
            if results:
                # Aggregate content from all results
                content_parts = []
                sources = []
                
                for i, result in enumerate(results):
                    print(f"[DEBUG] Processing Bedrock result {i+1}: {result}")
                    result_content = result.get("content", {}).get("text", "")
                    if result_content:
                        content_parts.append(result_content)
                    
                    # Extract source information
                    location = result.get("location", {})
                    print(f"[DEBUG] Location data for result {i+1}: {location}")
                    if location.get("type") == "S3":
                        s3_location = location.get("s3Location", {})
                        source_uri = s3_location.get("uri", "")
                        print(f"[DEBUG] S3 URI for result {i+1}: {source_uri}")
                        
                        # Convert S3 URI to Confluence URL if possible
                        # Format: s3://bucket/path/to/page-id.txt or similar
                        confluence_url = self._convert_s3_to_confluence_url(source_uri)
                        print(f"[DEBUG] Converted to Confluence URL: {confluence_url}")
                        sources.append(confluence_url)
                    elif location.get("type") == "CONFLUENCE":
                        # Direct Confluence location
                        confluence_location = location.get("confluenceLocation", {})
                        source_url = confluence_location.get("url", "")
                        print(f"[DEBUG] Direct Confluence URL for result {i+1}: {source_url}")
                        if source_url:
                            sources.append(source_url)
                    else:
                        print(f"[DEBUG] Unknown location type: {location.get('type')}")
                
                print(f"[DEBUG] Final sources array: {sources}")
                combined_content = "\n\n".join(content_parts) if content_parts else "[No content in results]"
                return {"content": combined_content, "sources": sources}
            else:
                return {"content": "[No results found in CodeDocumentation KB for query: '{}']".format(query), "sources": []}
        except (BotoCoreError, ClientError) as e:
            return {"content": f"[AWS Bedrock error] {str(e)}", "sources": []}
        except Exception as e:
            return {"content": f"[BedrockClient error] {str(e)}", "sources": []}

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
    
    def _convert_s3_to_confluence_url(self, s3_uri):
        """
        Convert S3 URI to Confluence URL based on your knowledge base structure.
        This is a placeholder - you'll need to adjust based on how your KB stores source metadata.
        """
        print(f"[DEBUG] Converting S3 URI to Confluence URL: {s3_uri}")
        if not s3_uri:
            print(f"[DEBUG] S3 URI is empty or None")
            return None
            
        try:
            # Example: s3://my-kb-bucket/confluence/UP/page-123456789.txt
            # Extract page ID and construct Confluence URL
            import re
            
            # Try to extract a page ID pattern (adjust regex based on your actual S3 structure)
            page_id_match = re.search(r'page-(\d+)', s3_uri)
            print(f"[DEBUG] Page ID pattern match: {page_id_match}")
            if page_id_match:
                page_id = page_id_match.group(1)
                base_url = os.getenv("CONFLUENCE_BASE_URL")
                print(f"[DEBUG] Found page ID: {page_id}, base URL: {base_url}")
                if base_url:
                    confluence_url = f"{base_url}/pages/viewpage.action?pageId={page_id}"
                    print(f"[DEBUG] Generated Confluence URL: {confluence_url}")
                    return confluence_url
            
            # Fallback: try to extract any number that could be a page ID
            number_match = re.search(r'(\d{10,})', s3_uri)  # Look for long numbers (page IDs)
            print(f"[DEBUG] Number pattern match: {number_match}")
            if number_match:
                page_id = number_match.group(1)
                base_url = os.getenv("CONFLUENCE_BASE_URL")
                print(f"[DEBUG] Found number: {page_id}, base URL: {base_url}")
                if base_url:
                    confluence_url = f"{base_url}/pages/viewpage.action?pageId={page_id}"
                    print(f"[DEBUG] Generated Confluence URL from number: {confluence_url}")
                    return confluence_url
                    
            # If we can't extract page ID, return the S3 URI as a reference
            print(f"[DEBUG] Could not extract page ID, returning S3 URI as-is")
            return s3_uri
        except Exception as e:
            print(f"[DEBUG] Exception in URL conversion: {e}")
            return s3_uri
    
    def search_code_documentation_text_only(self, query):
        """Backward compatibility method that returns only content text."""
        result = self.search_code_documentation(query)
        if isinstance(result, dict):
            return result.get("content", "")
        return result 