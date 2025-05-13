import os
import requests
import logging
import markdown_it
from markdown_it.token import Token

# Stub for Jira integration
def create_jira_ticket(summary, description, issue_type="Task"):
    jira_base_url = os.getenv("JIRA_BASE_URL")
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_project_key = os.getenv("JIRA_PROJECT_KEY")
    jira_user = os.getenv("CONFLUENCE_ADMIN_USER")  # JIRA user email
    if not all([jira_base_url, jira_api_token, jira_project_key, jira_user]):
        return {"error": "Missing JIRA configuration in environment variables."}

    # Extract Slack link if present in description
    slack_link = None
    for para in (description or "").split("\n\n"):
        if para.strip().startswith("Slack conversation: "):
            slack_link = para.strip().replace("Slack conversation: ", "").strip()
            break
    logging.info(f"Extracted Slack link for JIRA web link: {slack_link}")

    # Remove the Slack link from the description for the main ticket
    clean_description = "\n\n".join([
        para for para in (description or "No description provided.").split("\n\n")
        if not para.strip().startswith("Slack conversation: ")
    ])

    # Enhanced: Use markdown-it-py to parse Markdown and convert to ADF
    def markdown_to_adf(md_text):
        md = markdown_it.MarkdownIt()
        tokens = md.parse(md_text)
        adf_blocks = []
        stack = []
        def flush_stack():
            nonlocal stack
            if stack:
                adf_blocks.extend(stack)
                stack = []
        def text_marks(token):
            marks = []
            if token.markup == '**' or token.markup == '__':
                marks.append({"type": "strong"})
            if token.markup == '*' or token.markup == '_':
                marks.append({"type": "em"})
            if token.markup == '`':
                marks.append({"type": "code"})
            return marks
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.type == 'heading_open':
                level = int(t.tag[1])
                i += 1
                content = tokens[i].content if i < len(tokens) else ''
                adf_blocks.append({
                    "type": "heading",
                    "attrs": {"level": level},
                    "content": [{"type": "text", "text": content}]
                })
            elif t.type == 'paragraph_open':
                i += 1
                para_content = []
                while i < len(tokens) and tokens[i].type != 'paragraph_close':
                    if tokens[i].type == 'inline':
                        for child in tokens[i].children or []:
                            if child.type == 'text':
                                para_content.append({"type": "text", "text": child.content})
                            elif child.type == 'strong_open':
                                # Bold
                                j = 1
                                if i + j < len(tokens[i].children):
                                    para_content.append({
                                        "type": "text",
                                        "text": tokens[i].children[i + j].content,
                                        "marks": [{"type": "strong"}]
                                    })
                            elif child.type == 'em_open':
                                # Italic
                                j = 1
                                if i + j < len(tokens[i].children):
                                    para_content.append({
                                        "type": "text",
                                        "text": tokens[i].children[i + j].content,
                                        "marks": [{"type": "em"}]
                                    })
                            elif child.type == 'code_inline':
                                para_content.append({
                                    "type": "text",
                                    "text": child.content,
                                    "marks": [{"type": "code"}]
                                })
                    i += 1
                adf_blocks.append({"type": "paragraph", "content": para_content})
            elif t.type == 'bullet_list_open':
                items = []
                i += 1
                while i < len(tokens) and tokens[i].type != 'bullet_list_close':
                    if tokens[i].type == 'list_item_open':
                        i += 1
                        item_content = []
                        while i < len(tokens) and tokens[i].type != 'list_item_close':
                            if tokens[i].type == 'inline':
                                for child in tokens[i].children or []:
                                    if child.type == 'text':
                                        item_content.append({"type": "text", "text": child.content})
                                    elif child.type == 'strong_open':
                                        j = 1
                                        if i + j < len(tokens[i].children):
                                            item_content.append({
                                                "type": "text",
                                                "text": tokens[i].children[i + j].content,
                                                "marks": [{"type": "strong"}]
                                            })
                                    elif child.type == 'em_open':
                                        j = 1
                                        if i + j < len(tokens[i].children):
                                            item_content.append({
                                                "type": "text",
                                                "text": tokens[i].children[i + j].content,
                                                "marks": [{"type": "em"}]
                                            })
                                    elif child.type == 'code_inline':
                                        item_content.append({
                                            "type": "text",
                                            "text": child.content,
                                            "marks": [{"type": "code"}]
                                        })
                            i += 1
                        items.append({
                            "type": "listItem",
                            "content": [{"type": "paragraph", "content": item_content}]
                        })
                    else:
                        i += 1
                adf_blocks.append({"type": "bulletList", "content": items})
            elif t.type == 'ordered_list_open':
                items = []
                i += 1
                while i < len(tokens) and tokens[i].type != 'ordered_list_close':
                    if tokens[i].type == 'list_item_open':
                        i += 1
                        item_content = []
                        while i < len(tokens) and tokens[i].type != 'list_item_close':
                            if tokens[i].type == 'inline':
                                for child in tokens[i].children or []:
                                    if child.type == 'text':
                                        item_content.append({"type": "text", "text": child.content})
                                    elif child.type == 'strong_open':
                                        j = 1
                                        if i + j < len(tokens[i].children):
                                            item_content.append({
                                                "type": "text",
                                                "text": tokens[i].children[i + j].content,
                                                "marks": [{"type": "strong"}]
                                            })
                                    elif child.type == 'em_open':
                                        j = 1
                                        if i + j < len(tokens[i].children):
                                            item_content.append({
                                                "type": "text",
                                                "text": tokens[i].children[i + j].content,
                                                "marks": [{"type": "em"}]
                                            })
                                    elif child.type == 'code_inline':
                                        item_content.append({
                                            "type": "text",
                                            "text": child.content,
                                            "marks": [{"type": "code"}]
                                        })
                            i += 1
                        items.append({
                            "type": "listItem",
                            "content": [{"type": "paragraph", "content": item_content}]
                        })
                    else:
                        i += 1
                adf_blocks.append({"type": "orderedList", "content": items})
            elif t.type == 'blockquote_open':
                i += 1
                quote_content = []
                while i < len(tokens) and tokens[i].type != 'blockquote_close':
                    if tokens[i].type == 'inline':
                        for child in tokens[i].children or []:
                            if child.type == 'text':
                                quote_content.append({"type": "text", "text": child.content})
                    i += 1
                adf_blocks.append({"type": "blockquote", "content": [{"type": "paragraph", "content": quote_content}]})
            else:
                i += 1
        return adf_blocks

    # Use the markdown-to-adf parser for ADF content
    adf_blocks = markdown_to_adf(clean_description or "No description provided.")

    # If a Slack link is present, append a clickable ADF link paragraph
    if slack_link:
        adf_blocks.append({
            "type": "paragraph",
            "content": [
                {"type": "text", "text": "Slack conversation: "},
                {
                    "type": "text",
                    "text": "View in Slack",
                    "marks": [
                        {
                            "type": "link",
                            "attrs": {
                                "href": slack_link,
                                "title": "Slack Thread"
                            }
                        }
                    ]
                }
            ]
        })

    adf_description = {
        "type": "doc",
        "version": 1,
        "content": adf_blocks
    }

    url = f"{jira_base_url}/rest/api/3/issue"
    auth = (jira_user, jira_api_token)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "fields": {
            "project": {"key": jira_project_key},
            "summary": summary,
            "description": adf_description,
            "issuetype": {"name": issue_type}
        }
    }
    response = requests.post(url, auth=auth, headers=headers, json=data)
    if response.status_code == 201:
        issue = response.json()
        ticket_key = issue.get("key")
        ticket_url = f"{jira_base_url}/browse/{ticket_key}"
        # Add Slack link as a web link (remote link) if present
        if slack_link:
            logging.info(f"Adding JIRA web link: {slack_link} to ticket {ticket_key}")
            add_jira_web_link(ticket_key, slack_link, title="Slack Conversation")
        return {"key": ticket_key, "url": ticket_url}
    else:
        try:
            error = response.json()
        except Exception:
            error = response.text
        return {"error": error, "status_code": response.status_code}


def add_jira_web_link(issue_key, url, title="Slack Conversation"):
    jira_base_url = os.getenv("JIRA_BASE_URL")
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_user = os.getenv("CONFLUENCE_ADMIN_USER")
    if not all([jira_base_url, jira_api_token, jira_user]):
        return {"error": "Missing JIRA configuration in environment variables."}
    api_url = f"{jira_base_url}/rest/api/3/issue/{issue_key}/remotelink"
    auth = (jira_user, jira_api_token)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "object": {
            "url": url,
            "title": title
        }
    }
    logging.info(f"add_jira_web_link called with issue_key={issue_key}, url={url}, title={title}")
    response = requests.post(api_url, auth=auth, headers=headers, json=data)
    logging.info(f"JIRA web link response: {response.status_code} {response.text}")
    if response.status_code in (200, 201):
        return {"success": True}
    else:
        try:
            error = response.json()
        except Exception:
            error = response.text
        return {"error": error, "status_code": response.status_code} 