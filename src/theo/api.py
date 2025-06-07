from fastapi import FastAPI, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import os
from threading import Lock
import requests

# Initialize tracer
from ddtrace import tracer

from theo.crew import Theo
from theo.tools.datadog import DatadogClient
from theo.tools.slack import send_slack_message, add_slack_reaction, remove_slack_reaction, fetch_thread_history, get_last_user_message_ts

app = FastAPI(title="Theo Crew Orchestrator")
datadog = DatadogClient()

print("DD_TRACE_AGENTLESS:", os.environ.get("DD_TRACE_AGENTLESS"))
print("DD_API_KEY:", os.environ.get("DD_API_KEY"))
print("DD_SITE:", os.environ.get("DD_SITE"))
print("DD_LLMOBS_ML_APP:", os.environ.get("DD_LLMOBS_ML_APP"))

SLACK_BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID")

concluded_threads = set()
concluded_threads_lock = Lock()
processed_messages = set()
processed_messages_lock = Lock()
processed_reactions = set()

def extract_conversation_id(event_type, payload):
    if event_type == "slack":
        return payload.get("thread_ts") or payload.get("ts") or payload.get("event_ts")
    elif event_type == "github":
        return payload.get("thread_id") or payload.get("issue", {}).get("id") or payload.get("pull_request", {}).get("id")
    return None

def route_event_to_crew(event_type: str, payload: dict, parent_span=None, thread_history=None):
    if event_type == "slack" and "event" in payload:
        event = payload["event"]
        # Ignore messages sent by the bot itself FIRST
        if event.get("user") == SLACK_BOT_USER_ID or event.get("bot_id") or event.get("subtype") == "bot_message":
            print("[DEBUG] Ignoring message from bot itself.")
            return {"result": "Ignoring bot's own message.", "task": None, "conversation_id": extract_conversation_id(event_type, payload)}
        # Only respond to app_mention events or messages in a thread
        if event.get("type") != "app_mention":
            if event.get("type") == "message" and event.get("thread_ts"):
                pass  # allow
            else:
                print(f"[DEBUG] Ignoring Slack event type: {event.get('type')}")
                return {"result": f"Ignoring Slack event type: {event.get('type')}", "task": None, "conversation_id": extract_conversation_id(event_type, payload)}
    if parent_span is not None:
        span = tracer.start_span("crew.route_event", child_of=parent_span)
    else:
        span = tracer.start_span("crew.route_event")
    with span:
        user_message = ""
        if event_type == "slack" and "event" in payload and "text" in payload["event"]:
            user_message = payload["event"]["text"]
            mention_str = f"<@{SLACK_BOT_USER_ID}>"
            thread_ts = payload["event"].get("thread_ts")
            # Only require mention if not in a thread
            if not thread_ts and mention_str not in user_message and "@theo" not in user_message.lower():
                print("[DEBUG] Bot not mentioned, ignoring message.")
                return {"result": "Bot not mentioned, ignoring message.", "task": None, "conversation_id": extract_conversation_id(event_type, payload)}
            user_message = user_message.replace(mention_str, "").replace("@theo", "").strip()
        else:
            user_message = payload.get("text", "")
        conversation_id = extract_conversation_id(event_type, payload)
        theo = Theo()
        print(f"[DEBUG] Asking Supervisor agent to route message: {user_message}")
        supervisor_result = theo.supervisor_routing(question=user_message, conversation_id=conversation_id, thread_history=thread_history)
        print(f"[DEBUG] Supervisor agent LLM output: {supervisor_result}")
        valid_tasks = {"support_request", "documentation_update", "bi_report", "ticket_creation", "platform_health", "supervisor_health", "clarification_needed"}
        # Handle supervisor health/heartbeat direct response
        if isinstance(supervisor_result, tuple) and supervisor_result[0] in ("supervisor_health", "platform_health"):
            return {"result": supervisor_result[1], "task": supervisor_result[0], "conversation_id": conversation_id}
        # If supervisor_result is a dict, unpack for bi_report or documentation_update
        if isinstance(supervisor_result, dict):
            task_name = supervisor_result.get("task_name")
            print(f"[DEBUG] Task selected by Supervisor (dict): {task_name}")
            task_fn = getattr(theo, task_name, None)
            if task_fn and task_name in valid_tasks and task_name != "clarification_needed":
                # Pass channel and thread_ts for ticket_creation
                if task_name == "ticket_creation":
                    channel = payload.get("channel") or payload["event"].get("channel")
                    thread_ts = payload["event"].get("thread_ts") or payload["event"].get("ts") or payload.get("thread_ts") or payload.get("ts")
                    result = task_fn(
                        question=user_message,
                        conversation_id=conversation_id,
                        # Always pass full thread_history for ticket_creation
                        thread_history=thread_history,
                        channel=channel,
                        thread_ts=thread_ts
                    )
                else:
                    result = task_fn(
                        question=user_message,
                        conversation_id=conversation_id,
                        context_summary=supervisor_result.get("context_summary"),
                        thread_history=supervisor_result.get("thread_history")
                    )
                print(f"[DEBUG] Result from {task_name}: {result}")
                return {"result": result, "task": task_name, "conversation_id": conversation_id}
            else:
                return {"result": "Supervisor could not determine the correct agent. Please clarify your request.", "task": task_name, "conversation_id": conversation_id}
        else:
            task_name = str(supervisor_result).strip().lower()
            print(f"[DEBUG] Task selected by Supervisor: {task_name}")
            task_fn = getattr(theo, task_name, None)
            if task_fn and task_name in valid_tasks and task_name != "clarification_needed":
                # Pass channel and thread_ts for ticket_creation
                if task_name == "ticket_creation":
                    channel = payload.get("channel") or payload["event"].get("channel")
                    thread_ts = payload["event"].get("thread_ts") or payload["event"].get("ts") or payload.get("thread_ts") or payload.get("ts")
                    result = task_fn(
                        question=user_message,
                        conversation_id=conversation_id,
                        thread_history=thread_history,
                        channel=channel,
                        thread_ts=thread_ts
                    )
                else:
                    result = task_fn(question=user_message, conversation_id=conversation_id, thread_history=thread_history)
                print(f"[DEBUG] Result from {task_name}: {result}")
                return {"result": result, "task": task_name, "conversation_id": conversation_id}
            elif task_name == "clarification_needed":
                return {"result": "Supervisor could not determine the correct agent. Please clarify your request.", "task": task_name, "conversation_id": conversation_id}
            else:
                return {"result": f"Supervisor returned unknown task: {task_name}", "task": task_name, "conversation_id": conversation_id}

# Stub: Hand off event to Supervisor agent
async def handle_event_with_supervisor(event_type: str, payload: dict, parent_span=None, thread_history=None):
    if parent_span is not None:
        span = tracer.start_span("supervisor.handle_event", child_of=parent_span)
    else:
        span = tracer.start_span("supervisor.handle_event")
    with span:
        datadog.log_audit_event({"event_type": event_type, "payload": payload})
        response = route_event_to_crew(event_type, payload, parent_span=span, thread_history=thread_history)
        return response

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

@app.post("/slack/events")
async def slack_events(request: Request):
    payload = await request.json()
    event = payload.get("event", {})
    channel = payload.get("channel") or event.get("channel")
    parent_ts = event.get("thread_ts") or event.get("ts") or payload.get("thread_ts") or payload.get("ts")

    # Early exit: If thread is concluded, ignore all events unless bot is tagged
    with concluded_threads_lock:
        if parent_ts in concluded_threads:
            user_message = event.get("text", "")
            mention_str = f"<@{os.getenv('SLACK_BOT_USER_ID')}>"
            if mention_str not in user_message and "@theo" not in user_message.lower():
                return JSONResponse({"message": "Thread concluded, ignoring message unless bot is tagged."}, status_code=status.HTTP_200_OK)

    def thread_already_documented(thread_history):
        for msg in thread_history:
            # Check if the message is from the bot and contains a documentation update marker
            if (
                (msg.get("bot_id") or msg.get("user") == SLACK_BOT_USER_ID)
                and "Documentation Update" in (msg.get("text", "") or "")
            ):
                return True
        return False

    # Handle reaction_added events for conversation conclusion
    if event.get("type") == "reaction_added":
        reaction = event.get("reaction")
        item = event.get("item", {})
        thread_ts = item.get("ts")
        channel = item.get("channel")
        user = event.get("user")
        bot_user_id = os.getenv("SLACK_BOT_USER_ID")
        # Deduplicate by (channel, thread_ts, reaction)
        reaction_key = (channel, thread_ts, reaction)
        if reaction_key in processed_reactions:
            print(f"[INFO] Reaction {reaction} for thread {thread_ts} in channel {channel} already processed.")
            return JSONResponse({"message": "Duplicate reaction event, already processed."}, status_code=status.HTTP_200_OK)
        processed_reactions.add(reaction_key)
        if reaction == "white_check_mark" and thread_ts and channel and user != bot_user_id:
            with concluded_threads_lock:
                if thread_ts in concluded_threads:
                    print(f"[INFO] Thank-you message already sent for thread {thread_ts}, skipping duplicate.")
                    return JSONResponse({"message": "Thank-you already sent"}, status_code=status.HTTP_200_OK)
                concluded_threads.add(thread_ts)
            thread_history = fetch_thread_history(channel, thread_ts)
            theo = Theo()
            # Generate a supervisor LLM summary for TKB
            context_summary = theo.summarize_thread(thread_history)
            if thread_already_documented(thread_history):
                print(f"[INFO] Documentation already exists for thread {thread_ts}, skipping update.")
                # Always update TKB (General Knowledge) with a summary
                theo.documentation_update(
                    question="Update the documentation based on this thread.",
                    conversation_id=thread_ts,
                    thread_history=thread_history,
                    context_summary=context_summary,
                    close_conversation=True
                )
                send_slack_message(channel, "Thank you for concluding the conversation! :green_heart:\nI have updated our knowledge base with what I learned from this conversation.", thread_ts=thread_ts)
                return JSONResponse({"message": "Documentation already exists, but my knowledge base has been updated."}, status_code=status.HTTP_200_OK)
            # Only trigger if thread has technical content
            if any(
                any(keyword in (msg.get('text', '') or '').lower() for keyword in ["error", "workflow", "api", "scraper", "bug", "fix", "update", "document", "sql", "data"]) 
                for msg in thread_history
            ):
                theo.documentation_update(
                    question="Update the documentation based on this thread.",
                    conversation_id=thread_ts,
                    thread_history=thread_history,
                    context_summary=context_summary,
                    close_conversation=True
                )
                print(f"[INFO] Triggered documentation update for thread {thread_ts} in channel {channel} via :white_check_mark: reaction.")
            else:
                print(f"[INFO] Thread {thread_ts} did not contain technical content, skipping doc update.")
            # Thank and prompt for rating
            send_slack_message(channel, "Thank you for concluding the conversation! :green_heart:\nI have updated our knowledge base with what I learned from this conversation.", thread_ts=thread_ts)
            return JSONResponse({"message": "Slack conversation concluded"}, status_code=status.HTTP_200_OK)
        elif reaction == "x" and thread_ts and channel:
            with concluded_threads_lock:
                concluded_threads.add(thread_ts)
            admin_id = os.getenv("ADMIN_SLACK_USER_ID")
            apology = f"We're sorry this didn't help. <@{admin_id}> will assist you shortly!"
            send_slack_message(channel, apology, thread_ts=thread_ts)
            return JSONResponse({"message": "Slack conversation concluded"}, status_code=status.HTTP_200_OK)

    # Pre-check: Should we process this event?
    should_process = True
    if event.get("user") == SLACK_BOT_USER_ID or event.get("bot_id") or event.get("subtype") == "bot_message":
        should_process = False
    elif event.get("type") != "app_mention":
        if not (event.get("type") == "message" and event.get("thread_ts")):
            should_process = False

    if not should_process:
        return JSONResponse({"message": "Slack event ignored"}, status_code=status.HTTP_200_OK)

    # Only add reactions if we are processing
    if channel and parent_ts:
        add_slack_reaction(channel, parent_ts, "robot_face")
        add_slack_reaction(channel, parent_ts, "hourglass_flowing_sand")
        # Add :loading-circle: to last user message in thread, but not if it's the root message
        thread_history_for_reaction = fetch_thread_history(channel, parent_ts)
        last_user_ts = get_last_user_message_ts(thread_history_for_reaction, SLACK_BOT_USER_ID)
        if last_user_ts and last_user_ts != parent_ts:
            add_slack_reaction(channel, last_user_ts, "loading-circle")
    try:
        with tracer.start_span("conversation.workflow", service="theo", resource="slack_event") as parent_span:
            parent_span.set_tag("conversation.id", parent_ts)
            parent_span.set_tag("event.type", "slack")
            msg_ts = event.get("ts")
            with processed_messages_lock:
                if msg_ts and msg_ts in processed_messages:
                    return JSONResponse({"message": "Duplicate event, already processed."}, status_code=status.HTTP_200_OK)
                if msg_ts:
                    processed_messages.add(msg_ts)
            # Fetch thread history for context
            thread_history = []
            if channel and parent_ts:
                thread_history = fetch_thread_history(channel, parent_ts)
            supervisor_response = await handle_event_with_supervisor("slack", payload, parent_span=parent_span, thread_history=thread_history)
            answer = supervisor_response.get("result") or "No answer generated."
            print(f"[DEBUG] Slack message to be sent: {answer}")
            if channel and answer and not answer.startswith("Ignoring bot's own message") and not answer.startswith("Bot not mentioned") and not answer.startswith("Ignoring Slack event type"):
                send_slack_message(channel, answer, thread_ts=parent_ts)
    except Exception as e:
        if channel and parent_ts:
            add_slack_reaction(channel, parent_ts, "internet-problems")
        admin_id = os.getenv("ADMIN_SLACK_USER_ID")
        if admin_id and channel and parent_ts:
            send_slack_message(channel, f":internet-problems: Error occurred, <@{admin_id}>", thread_ts=parent_ts)
        raise
    finally:
        # Always remove hourglass after processing, unless this was a reaction_added event
        if event.get("type") != "reaction_added" and channel and parent_ts:
            remove_slack_reaction(channel, parent_ts, "hourglass_flowing_sand")
            # Remove :loading-circle: from last user message, but not if it's the root message
            thread_history_for_reaction = fetch_thread_history(channel, parent_ts)
            last_user_ts = get_last_user_message_ts(thread_history_for_reaction, SLACK_BOT_USER_ID)
            if last_user_ts and last_user_ts != parent_ts:
                remove_slack_reaction(channel, last_user_ts, "loading-circle")
    return JSONResponse({"message": "Slack event received", "supervisor_response": supervisor_response}, status_code=status.HTTP_200_OK)

@app.post("/github/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    # Only handle push events to main or test branch
    if payload.get("ref") in ["refs/heads/main", "refs/heads/PD-7618"] and "commits" in payload:
        def process_github_push(payload):
            theo = Theo()
            parent_adr_page_id = os.getenv("CONFLUENCE_ADR_PARENT_PAGE_ID", "2117566465")
            commits = payload["commits"]
            push_info = {
                "pusher": payload.get("pusher", {}),
                "after": payload.get("after"),
                "before": payload.get("before"),
                "ref": payload.get("ref"),
                "repository": payload.get("repository", {}).get("full_name", "")
            }
            # Create a single ADR for the push
            adr_result = theo.create_adr_from_github_push(commits, push_info, parent_adr_page_id=parent_adr_page_id)
            # Collect all .model.ts changes across all commits
            all_model_changes = []
            for commit in commits:
                commit_hash = commit.get("id", "")
                commit_body = commit.get("message", "")
                diff_url = commit.get("url", "") + ".diff" if commit.get("url") else None
                for file_path in commit.get("added", []) + commit.get("modified", []) + commit.get("removed", []):
                    if file_path.endswith(".model.ts"):
                        all_model_changes.append({
                            "path": file_path,
                            "commit_hash": commit_hash,
                            "commit_message": commit_body,
                            "diff_url": diff_url
                        })
            tbikb_result = None
            if all_model_changes:
                # Only update TBIKB if at least one .model.ts file has a schema change
                schema_change_detected = False
                for model_change in all_model_changes:
                    diff_url = model_change.get("diff_url")
                    if diff_url:
                        try:
                            resp = requests.get(diff_url)
                            if resp.ok:
                                diff_text = resp.text
                                if theo._is_schema_change(diff_text):
                                    schema_change_detected = True
                                    break
                        except Exception as e:
                            print(f"[ERROR] Could not fetch diff for {diff_url}: {e}")
                if schema_change_detected:
                    push_date = payload.get("head_commit", {}).get("timestamp") or None
                    tbikb_result = theo.update_tbikb_for_model_changes(all_model_changes, push_info, push_date=push_date)
            # Optionally log or store results
        background_tasks.add_task(process_github_push, payload)
        return JSONResponse({"message": "Processing GitHub push in background"}, status_code=status.HTTP_200_OK)
    # Fallback: default supervisor handling for other events
    conversation_id = payload.get("thread_id") or payload.get("issue", {}).get("id") or payload.get("pull_request", {}).get("id")
    with tracer.trace("conversation.workflow", service="theo", resource="github_event") as parent_span:
        parent_span.set_tag("conversation.id", conversation_id)
        parent_span.set_tag("event.type", "github")
        supervisor_response = await handle_event_with_supervisor("github", payload, parent_span=parent_span)
    return JSONResponse({"message": "GitHub webhook received", "supervisor_response": supervisor_response}, status_code=status.HTTP_200_OK)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.theo.api:app", host="0.0.0.0", port=port, reload=True) 