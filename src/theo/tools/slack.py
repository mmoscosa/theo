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
        return False
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
    response_data = response.json() if response.ok else {}
    
    if not response.ok or not response_data.get("ok"):
        error = response_data.get("error", "unknown_error")
        if error != "already_reacted":  # Don't log already_reacted as an error
            print(f"[Slack] Failed to add reaction: {response.text}")
        return False
    else:
        print(f"[Slack] Reaction :{emoji}: added successfully.")
        return True

def remove_slack_reaction(channel, timestamp, emoji):
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_token:
        print("[Slack] SLACK_BOT_TOKEN not set for reactions.")
        return False
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
    response_data = response.json() if response.ok else {}
    
    if not response.ok or not response_data.get("ok"):
        error = response_data.get("error", "unknown_error")
        if error not in ["no_reaction", "not_authed"]:  # Don't log these as errors
            print(f"[Slack] Failed to remove reaction: {response.text}")
        return False
    else:
        print(f"[Slack] Reaction :{emoji}: removed successfully.")
        return True

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

def get_last_user_message_ts(thread_history, bot_user_id=None):
    """Return the timestamp of the last user (non-bot) message in the thread."""
    if not thread_history:
        return None
    if bot_user_id is None:
        bot_user_id = os.getenv("SLACK_BOT_USER_ID")
    # Filter out bot messages
    user_msgs = [msg for msg in thread_history if not msg.get("bot_id") and msg.get("user") != bot_user_id]
    if not user_msgs:
        return None
    # Return the ts of the last user message
    return user_msgs[-1].get("ts")

def get_slack_user_info(user_id):
    """Get user information from Slack API to get display name."""
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_token:
        print("[Slack] SLACK_BOT_TOKEN not set for user info.")
        return None
    url = "https://slack.com/api/users.info"
    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    params = {
        "user": user_id
    }
    response = requests.get(url, headers=headers, params=params)
    if not response.ok or not response.json().get("ok"):
        print(f"[Slack] Failed to get user info for {user_id}: {response.text}")
        return None
    user_data = response.json().get("user", {})
    profile = user_data.get("profile", {})
    
    # Try to get display name, real name, or fallback to username
    display_name = (
        profile.get("display_name") or 
        profile.get("real_name") or 
        user_data.get("name") or
        user_id
    )
    return display_name 