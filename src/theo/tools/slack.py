import os
import requests

# Stub for Slack integration
def send_slack_message(channel, text, thread_ts=None):
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_token:
        print("[Slack] SLACK_BOT_TOKEN not set.")
        return
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/json"
    }
    data = {
        "channel": channel,
        "text": text
    }
    if thread_ts:
        data["thread_ts"] = thread_ts
    response = requests.post(url, headers=headers, json=data)
    if not response.ok or not response.json().get("ok"):
        print(f"[Slack] Failed to send message: {response.text}")
    else:
        print("[Slack] Message sent successfully.")

def add_slack_reaction(channel, timestamp, emoji):
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_token:
        print("[Slack] SLACK_BOT_TOKEN not set for reactions.")
        return
    url = "https://slack.com/api/reactions.add"
    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/json"
    }
    data = {
        "channel": channel,
        "timestamp": timestamp,
        "name": emoji
    }
    response = requests.post(url, headers=headers, json=data)
    if not response.ok or not response.json().get("ok"):
        print(f"[Slack] Failed to add reaction: {response.text}")
    else:
        print(f"[Slack] Reaction :{emoji}: added successfully.")

def remove_slack_reaction(channel, timestamp, emoji):
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_token:
        print("[Slack] SLACK_BOT_TOKEN not set for reactions.")
        return
    url = "https://slack.com/api/reactions.remove"
    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/json"
    }
    data = {
        "channel": channel,
        "timestamp": timestamp,
        "name": emoji
    }
    response = requests.post(url, headers=headers, json=data)
    if not response.ok or not response.json().get("ok"):
        print(f"[Slack] Failed to remove reaction: {response.text}")
    else:
        print(f"[Slack] Reaction :{emoji}: removed successfully.")

def format_slack_response(
    category_emoji, category_title,
    clarification=None,
    quick_fix=None,
    main_content=None,
    agent_role=None,
    agent_emoji=None
):
    message = f"{category_emoji} *{category_title}*\n"
    if main_content:
        message += f"\n{main_content}\n"
    if clarification:
        message += f"\n❓ *To clarify:*\n{clarification}\n"
    elif quick_fix:
        message += f"\n⚡ *Quick fix:*\n{quick_fix}\n"
    if agent_role and agent_emoji:
        message += f"\n_Taken by {agent_emoji} {agent_role}_"
    return message

def fetch_thread_history(channel, thread_ts):
    """Fetch all messages in a Slack thread using conversations.replies API."""
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_token:
        print("[Slack] SLACK_BOT_TOKEN not set for thread history.")
        return []
    url = "https://slack.com/api/conversations.replies"
    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    params = {
        "channel": channel,
        "ts": thread_ts
    }
    response = requests.get(url, headers=headers, params=params)
    if not response.ok or not response.json().get("ok"):
        print(f"[Slack] Failed to fetch thread history: {response.text}")
        return []
    return response.json().get("messages", []) 