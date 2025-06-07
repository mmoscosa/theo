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

# Note: ADR Index functionality has been removed as we've transitioned from ADR pages to Code Commit pages
# Code commits are now stored directly in the Code Commits folder without requiring index tables 