"""
Metabase API integration for creating saved questions with SQL.
"""

import os
import requests
import urllib.parse
import re
from typing import Optional


class MetabaseClient:
    def __init__(self):
        self.base_url = os.getenv("METABASE_BASE_URL", "https://sunroom-rentals.metabaseapp.com")
        self.api_key = os.getenv("METABASE_API_KEY")
        self.database_id = int(os.getenv("METABASE_DATABASE_ID", "1"))  # Default to 1, adjust as needed
        
        if not self.api_key:
            print("[WARNING] METABASE_API_KEY not set, Metabase integration will be disabled")
    
    def search_recent_questions(self, query: str = "", limit: int = 3) -> str:
        """
        Search for recent Metabase questions and return them as formatted Slack links.
        Returns top 3 most recent questions by updated_at.
        """
        if not self.api_key:
            return "❌ Metabase API not configured"
            
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        try:
            # Build search parameters
            params = {
                "models": "card",  # Only search for questions/cards
                "limit": 50  # Get more results to sort properly
            }
            
            # Add query filter if provided
            if query.strip():
                params["q"] = query.strip()
            
            # Search via Metabase API
            response = requests.get(
                f"{self.base_url}/api/search",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                print(f"[DEBUG] Metabase search response type: {type(results)}")
                print(f"[DEBUG] Metabase search response: {results}")
                
                # Handle different response formats
                if isinstance(results, dict):
                    # If response is wrapped in a data object
                    if "data" in results:
                        questions_data = results["data"]
                    else:
                        # Response might be a dict with other structure
                        questions_data = results
                elif isinstance(results, list):
                    # Direct list response
                    questions_data = results
                else:
                    print(f"[ERROR] Unexpected response format: {type(results)}")
                    return "❌ Unexpected Metabase response format"
                
                # Filter for cards and sort by updated_at (most recent first)
                if isinstance(questions_data, list):
                    questions = [item for item in questions_data if item.get("model") == "card"]
                elif isinstance(questions_data, dict):
                    # If it's still a dict, try to extract questions array
                    questions = questions_data.get("questions", [])
                    if not questions:
                        questions = questions_data.get("results", [])
                    if not questions:
                        questions = [questions_data]  # Single item
                else:
                    return "❌ Unable to parse questions from response"
                
                questions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
                
                # Take top N results
                total_count = len(questions)
                top_questions = questions[:limit]
                has_more = total_count > limit
                
                if not top_questions:
                    return "❌ No questions found in Metabase"
                
                # Format as Slack links with updated_at and bullet points
                formatted_links = []
                for q in top_questions:
                    question_id = q.get("id")
                    title = q.get("name", f"Question {question_id}")
                    url = f"{self.base_url}/question/{question_id}"
                    updated_at = q.get("updated_at", "")
                    
                    # Clean title for Slack formatting
                    clean_title = title.replace("|", "│")  # Replace pipe chars that break Slack links
                    
                    # Format updated_at as date only if available
                    if updated_at:
                        # Parse and format the date (Metabase typically returns ISO format)
                        try:
                            from datetime import datetime
                            # Handle different date formats from Metabase
                            if "T" in updated_at:
                                # ISO format: 2024-03-15T10:30:00Z
                                dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                            else:
                                # Simple format: 2024-03-15 10:30:00
                                dt = datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S")
                            formatted_date = dt.strftime("%Y-%m-%d")  # Date only, no time
                        except Exception:
                            # Fallback to raw date if parsing fails - extract just the date part
                            formatted_date = updated_at[:10] if len(updated_at) >= 10 else updated_at
                        
                        formatted_links.append(f"- <{url}|{clean_title}> ({formatted_date})")
                    else:
                        formatted_links.append(f"- <{url}|{clean_title}>")
                
                results_text = "\n".join(formatted_links)
                
                # Add "see all" option if there are more results
                if has_more:
                    # Create Metabase search URL 
                    import urllib.parse
                    search_url = f"{self.base_url}/search?q={urllib.parse.quote(query)}"
                    if total_count >= 50:
                        results_text += f"\n\n- <{search_url}|See all results in Metabase>"
                    else:
                        results_text += f"\n\n- <{search_url}|See all {total_count} results in Metabase>"
                
                return results_text
                
            else:
                print(f"[ERROR] Metabase search failed: {response.status_code} - {response.text}")
                return f"❌ Metabase search failed (Status: {response.status_code})"
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Metabase search request failed: {e}")
            return "❌ Failed to search Metabase (connection error)"

    def create_saved_question(self, sql: str, title: str = None) -> Optional[str]:
        """
        Create a saved question in Metabase with the provided SQL.
        Returns the URL to the created question, or None if failed.
        """
        if not self.api_key:
            return None
            
        # Generate a descriptive title if none provided
        if not title:
            title = self._generate_title_from_sql(sql)
        
        # Use collection ID where user has write permissions (from error message: /collection/4/)
        collection_id = int(os.getenv("METABASE_COLLECTION_ID", "4"))
        
        # Create the payload for Metabase API
        payload = {
            "name": title,
            "dataset_query": {
                "type": "native",
                "native": {
                    "query": sql
                },
                "database": self.database_id
            },
            "display": "table",
            "visualization_settings": {},
            "collection_id": collection_id  # Specify collection where user has curate permissions
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        try:
            # Create the question via Metabase API
            response = requests.post(
                f"{self.base_url}/api/card",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                question_data = response.json()
                question_id = question_data.get("id")
                
                if question_id:
                    # Return direct link to the created question
                    question_url = f"{self.base_url}/question/{question_id}"
                    print(f"[DEBUG] Created Metabase question: {question_url}")
                    return question_url
                else:
                    print(f"[ERROR] Metabase API response missing question ID: {response.text}")
                    return None
            else:
                print(f"[ERROR] Metabase API failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Metabase API request failed: {e}")
            return None
    
    def _generate_title_from_sql(self, sql: str) -> str:
        """
        Generate a descriptive title from SQL query.
        """
        # Clean up the SQL for analysis
        sql_clean = re.sub(r'\s+', ' ', sql.strip())
        
        # Try to extract main table name
        table_match = re.search(r'FROM\s+(\w+)', sql_clean, re.IGNORECASE)
        table_name = table_match.group(1) if table_match else "data"
        
        # Check query type
        if re.search(r'^\s*SELECT\s+COUNT', sql_clean, re.IGNORECASE):
            return f"Count of {table_name}"
        elif re.search(r'^\s*SELECT.*AVG\(', sql_clean, re.IGNORECASE):
            return f"Average analysis for {table_name}"
        elif re.search(r'^\s*SELECT.*SUM\(', sql_clean, re.IGNORECASE):
            return f"Sum analysis for {table_name}"
        elif re.search(r'GROUP BY', sql_clean, re.IGNORECASE):
            return f"Grouped analysis of {table_name}"
        elif re.search(r'WHERE.*created.*today|current_date', sql_clean, re.IGNORECASE):
            return f"Today's {table_name}"
        else:
            return f"Query: {table_name} analysis"


def create_metabase_question(sql: str, title: str = None) -> Optional[str]:
    """
    Convenience function to create a Metabase question.
    Returns the URL to the created question, or None if failed.
    """
    client = MetabaseClient()
    return client.create_saved_question(sql, title)


def search_metabase_questions(query: str = "", limit: int = 3) -> str:
    """
    Convenience function to search for recent Metabase questions.
    Returns formatted Slack links to the top 3 most recent questions.
    """
    client = MetabaseClient()
    return client.search_recent_questions(query, limit) 