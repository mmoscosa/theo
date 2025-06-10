import os
import requests
from ddtrace.llmobs import LLMObs
import logging
import markdown_it
import re

def markdown_to_confluence_storage(md_text):
    md = markdown_it.MarkdownIt()
    html = md.render(md_text)
    return html

def update_confluence_page(page_id, new_content):
    base_url = os.getenv("CONFLUENCE_BASE_URL")
    user = os.getenv("CONFLUENCE_ADMIN_USER")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    url = f"{base_url}/rest/api/content/{page_id}?expand=body.storage,version,title"
    # Fetch current version and content
    resp = requests.get(url, auth=(user, api_token))
    resp.raise_for_status()
    current = resp.json()
    version = current["version"]["number"] + 1
    current_content = current["body"]["storage"]["value"]
    title = current["title"]
    logging.info(f"[Confluence] Fetched current content for page {page_id}")
    # Convert Markdown to Confluence storage format (HTML)
    new_content_html = markdown_to_confluence_storage(new_content)
    # Use LLM to merge new_content into current_content
    prompt = (
        "You are a technical writer. Here is the current Confluence documentation (in storage format):\n"
        f"---\n{current_content}\n---\n"
        "Here is new information to be included (in HTML):\n"
        f"---\n{new_content_html}\n---\n"
        "Update the documentation to include the new information, modifying only the relevant sections. Do not rewrite or duplicate existing content. Return the full updated document in Confluence storage format."
    )
    logging.info(f"[Confluence] Sending smart update prompt to LLM.")
    updated_content = LLMObs.annotate(
        input_data=[{"role": "user", "content": prompt}],
        output_data=[{"role": "assistant", "content": ""}],
        tags={"agent_role": "technical_writer", "action": "confluence_smart_update"}
    )
    if not updated_content or updated_content == "None":
        logging.warning(f"[Confluence] LLM did not return updated content, falling back to append.")
        updated_content = current_content + "<hr/>" + new_content_html
    data = {
        "version": {"number": version},
        "title": title,
        "type": "page",
        "body": {
            "storage": {
                "value": updated_content,
                "representation": "storage"
            }
        }
    }
    put_resp = requests.put(f"{base_url}/rest/api/content/{page_id}", json=data, auth=(user, api_token))
    put_resp.raise_for_status()
    logging.info(f"[Confluence] Page {page_id} updated successfully.")
    return put_resp.json()

def create_confluence_page(title, content, space_key=None, parent_id=None):
    base_url = os.getenv("CONFLUENCE_BASE_URL")
    if space_key is None:
        space_key = os.getenv("CONFLUENCE_SPACE_KEY_UP", "UP")
    user = os.getenv("CONFLUENCE_ADMIN_USER")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    url = f"{base_url}/rest/api/content/"
    # Convert Markdown to Confluence storage format (HTML)
    content_html = markdown_to_confluence_storage(content)
    MAX_TITLE_LENGTH = 255
    page_title = title[:MAX_TITLE_LENGTH]
    data = {
        "type": "page",
        "title": page_title,
        "space": {"key": space_key},
        "body": {
            "storage": {
                "value": content_html,
                "representation": "storage"
            }
        }
    }
    # Add parent page if specified
    if parent_id:
        data["ancestors"] = [{"id": str(parent_id)}]
    # Check for duplicate title before creating
    try:
        from theo.tools.confluence import get_all_confluence_pages, update_confluence_page
        all_pages = get_all_confluence_pages(space_key=space_key)
        for page in all_pages:
            if page["title"].strip().lower() == page_title.strip().lower():
                print(f"[DEBUG] Page with title '{page_title}' already exists (id={page['id']}), updating instead of creating.")
                return update_confluence_page(page["id"], content)
    except Exception as e:
        print(f"[ERROR] Error checking for duplicate title: {e}")
    print(f"[DEBUG] Creating Confluence page: url={url}, data={data}")
    resp = requests.post(url, json=data, auth=(user, api_token))
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Confluence API error: {resp.text}")
        raise
    print(f"[DEBUG] Confluence API response: {resp.text}")
    return resp.json()

def get_all_confluence_pages(space_key=None):
    base_url = os.getenv("CONFLUENCE_BASE_URL")
    if space_key is None:
        space_key = os.getenv("CONFLUENCE_SPACE_KEY_UP", "UP")
    user = os.getenv("CONFLUENCE_ADMIN_USER")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    url = f"{base_url}/rest/api/content"
    params = {
        "spaceKey": space_key,
        "limit": 100,  # adjust as needed
        "expand": "title"
    }
    pages = []
    start = 0
    while True:
        params["start"] = start
        resp = requests.get(url, params=params, auth=(user, api_token))
        resp.raise_for_status()
        data = resp.json()
        for result in data.get("results", []):
            pages.append({"id": result["id"], "title": result["title"]})
        if data.get("_links", {}).get("next"):
            start += len(data.get("results", []))
        else:
            break
    return pages

def get_confluence_page_by_id(page_id):
    """Fetch a Confluence page's full content by its ID."""
    base_url = os.getenv("CONFLUENCE_BASE_URL")
    user = os.getenv("CONFLUENCE_ADMIN_USER")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    url = f"{base_url}/rest/api/content/{page_id}?expand=body.storage,version,title"
    resp = requests.get(url, auth=(user, api_token))
    resp.raise_for_status()
    return resp.json()

def add_row_to_adr_index(adr_title, adr_url, status, authors, date):
    """
    Add a new row to the ADR index table in Confluence.
    """
    try:
        base_url = os.getenv("CONFLUENCE_BASE_URL")
        user = os.getenv("CONFLUENCE_ADMIN_USER")
        api_token = os.getenv("CONFLUENCE_API_TOKEN")
        space_key = os.getenv("CONFLUENCE_SPACE_KEY_UP", "UP")
        
        # Find the ADR index page
        adr_index_page_id = os.getenv("CONFLUENCE_ADR_INDEX_PAGE_ID")
        
        if not adr_index_page_id:
            # Try to find the ADR index page by title (try multiple possible titles)
            search_titles = [
                "Architecture Decision Record (ADR)",  # Singular with parentheses
                "Architecture Decision Records",        # Plural  
                "Architecture Decision Record",         # Singular
                "ADR"                                  # Short form
            ]
            
            search_url = f"{base_url}/rest/api/content"
            
            for title in search_titles:
                params = {
                    "spaceKey": space_key,
                    "title": title,
                    "expand": "body.storage,version"
                }
                response = requests.get(search_url, auth=(user, api_token), params=params)
                
                if response.status_code == 200:
                    search_results = response.json()
                    if search_results.get("results"):
                        adr_index_page_id = search_results["results"][0]["id"]
                        print(f"[DEBUG] Found ADR index page '{title}' with ID: {adr_index_page_id}")
                        break
                        
            if not adr_index_page_id:
                print("[WARNING] ADR index page not found with any of the expected titles")
                print(f"[DEBUG] Searched for titles: {search_titles}")
                return {"error": "ADR index page not found"}

        if not adr_index_page_id:
            print("[WARNING] ADR index page ID not available")
            return {"error": "ADR index page ID not available"}
        
        # Get current page content
        page_url = f"{base_url}/rest/api/content/{adr_index_page_id}?expand=body.storage,version"
        response = requests.get(page_url, auth=(user, api_token))
        
        if response.status_code != 200:
            print(f"[ERROR] Failed to get ADR index page: {response.text}")
            return {"error": "Failed to get ADR index page"}
        
        page_data = response.json()
        current_content = page_data["body"]["storage"]["value"]
        current_version = page_data["version"]["number"]
        page_title = page_data["title"]
        
        # Create new table row in Confluence storage format
        new_row = f"""<tr>
<td><p><a href="{adr_url}">{adr_title}</a></p></td>
<td><p>{status}</p></td>
<td><p>{date}</p></td>
<td><p>{authors}</p></td>
<td><p>🧠 Human-written</p></td>
</tr>"""
        
        # Find the ADR Index table specifically and insert after its header row
        import re
        
        # Step 1: Find the ADR Index section specifically
        adr_index_section_pattern = r'(<h2[^>]*>.*?📄.*?ADR.*?Index.*?</h2>.*?<table[^>]*>.*?</table>)'
        section_match = re.search(adr_index_section_pattern, current_content, re.DOTALL | re.IGNORECASE)
        
        if section_match:
            # Step 2: Within the ADR Index section, find the header row
            adr_index_section = section_match.group(1)
            section_start = section_match.start()
            
            # Look for header row within this specific section
            header_patterns = [
                r'(<tr[^>]*>.*?Title.*?Status.*?Date.*?Authors.*?Source.*?</tr>)',
                r'(<tr[^>]*>.*?Title.*?Status.*?</tr>)'
            ]
            
            header_end_relative = None
            for pattern in header_patterns:
                header_match = re.search(pattern, adr_index_section, re.DOTALL | re.IGNORECASE)
                if header_match:
                    header_end_relative = header_match.end()
                    print(f"[DEBUG] Found ADR Index header row in section")
                    break
            
            if header_end_relative:
                # Calculate absolute position in the full content
                header_end_absolute = section_start + header_end_relative
                
                # Insert the new row right after the header row
                updated_content = (
                    current_content[:header_end_absolute] + 
                    "\n" + new_row + 
                    current_content[header_end_absolute:]
                )
                print(f"[DEBUG] Inserted ADR '{adr_title}' after ADR Index header row")
            else:
                print(f"[WARNING] Found ADR Index section but no header row, appending to section end")
                section_end = section_match.end() - 8  # Before </table>
                updated_content = (
                    current_content[:section_end] + 
                    new_row + "\n" +
                    current_content[section_end:]
                )
        else:
            # Fallback: Look for any header row if we can't find the specific section
            fallback_pattern = r'(<tr[^>]*>.*?Title.*?Status.*?</tr>)'
            fallback_match = re.search(fallback_pattern, current_content, re.DOTALL | re.IGNORECASE)
            
            if fallback_match:
                header_end = fallback_match.end()
                updated_content = (
                    current_content[:header_end] + 
                    "\n" + new_row + 
                    current_content[header_end:]
                )
                print(f"[DEBUG] Used fallback header detection")
            else:
                print(f"[WARNING] Could not find any header row, appending to end")
                updated_content = current_content + new_row
        
        # Update the page
        update_data = {
            "version": {"number": current_version + 1},
            "title": page_title,
            "type": "page",
            "body": {
                "storage": {
                    "value": updated_content,
                    "representation": "storage"
                }
            }
        }
        
        update_response = requests.put(
            f"{base_url}/rest/api/content/{adr_index_page_id}",
            json=update_data,
            auth=(user, api_token)
        )
        
        if update_response.status_code == 200:
            print(f"[DEBUG] Successfully added ADR '{adr_title}' to index")
            return {"success": True, "page_id": adr_index_page_id}
        else:
            print(f"[ERROR] Failed to update ADR index: {update_response.text}")
            return {"error": "Failed to update ADR index"}
            
    except Exception as e:
        print(f"[ERROR] Exception in add_row_to_adr_index: {e}")
        return {"error": str(e)} 