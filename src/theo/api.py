from fastapi import FastAPI, Request, status, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import os
from threading import Lock
import requests
import asyncio
import hmac
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
import uuid
from urllib.parse import unquote

# Initialize tracer
from ddtrace import tracer

from theo.crew import Theo
from theo.tools.datadog import DatadogClient
from theo.tools.slack import send_slack_message, add_slack_reaction, remove_slack_reaction, fetch_thread_history, get_last_user_message_ts
from theo.threading import (
    conversation_manager,
    reaction_manager,
    ConversationStatus,
    with_retry,
    timeout_wrapper
)

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

# Start the conversation manager cleanup task on startup
@app.on_event("startup")
async def startup_event():
    await conversation_manager.start_cleanup_task()
    print("[INFO] Started conversation cleanup task")

@app.on_event("shutdown")
async def shutdown_event():
    await conversation_manager.stop_cleanup_task()
    print("[INFO] Stopped conversation cleanup task")

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
        valid_tasks = {"support_request", "documentation_update", "adr_creation", "documentation_and_adr", "bi_report", "ticket_creation", "platform_health", "supervisor_health", "clarification_needed"}
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

# Async wrapper for supervisor handling
async def handle_event_with_supervisor(event_type: str, payload: dict, parent_span=None, thread_history=None):
    """Async wrapper that runs the synchronous CrewAI processing in a thread pool"""
    if parent_span is not None:
        span = tracer.start_span("supervisor.handle_event", child_of=parent_span)
    else:
        span = tracer.start_span("supervisor.handle_event")
    
    with span:
        datadog.log_audit_event({"event_type": event_type, "payload": payload})
        
        # Run the synchronous crew routing in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            route_event_to_crew, 
            event_type, 
            payload, 
            span, 
            thread_history
        )
        return response

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

@app.post("/test/bedrock")
async def test_bedrock(request: Request):
    """Test endpoint for debugging Bedrock functionality"""
    payload = await request.json()
    question = payload.get("question", "test question")
    
    # Import and test Bedrock directly
    from theo.tools.bedrock import BedrockClient
    bedrock = BedrockClient()
    
    print(f"[TEST] Testing Bedrock with question: {question}")
    print(f"[TEST] Bedrock KB IDs: code_doc={bedrock.code_doc_kb}, db_schema={bedrock.db_schema_kb}, general={bedrock.general_kb}")
    
    try:
        result = bedrock.search_code_documentation(question)
        print(f"[TEST] Bedrock result: {result}")
        return {"result": result, "question": question}
    except Exception as e:
        print(f"[TEST] Bedrock error: {e}")
        import traceback
        print(f"[TEST] Full traceback: {traceback.format_exc()}")
        return {"error": str(e), "question": question}

@app.get("/conversations/stats")
async def conversation_stats():
    """Get conversation processing statistics"""
    stats = await conversation_manager.get_conversation_stats()
    active = await conversation_manager.get_active_conversations()
    
    return {
        "stats": stats,
        "active_conversations": len(active),
        "active_conversation_ids": list(active.keys())
    }

@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    """Fast endpoint that quickly acknowledges Slack events and processes them in background"""
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

    # Handle reaction_added events for conversation conclusion (keep synchronous for quick response)
    if event.get("type") == "reaction_added":
        return await handle_reaction_event(event, channel, parent_ts, payload)

    # Pre-check: Should we process this event?
    should_process = True
    if event.get("user") == SLACK_BOT_USER_ID or event.get("bot_id") or event.get("subtype") == "bot_message":
        should_process = False
    elif event.get("type") != "app_mention":
        if not (event.get("type") == "message" and event.get("thread_ts")):
            should_process = False

    if not should_process:
        return JSONResponse({"message": "Slack event ignored"}, status_code=status.HTTP_200_OK)

    # Check for duplicate messages
    msg_ts = event.get("ts")
    with processed_messages_lock:
        if msg_ts and msg_ts in processed_messages:
            return JSONResponse({"message": "Duplicate event, already processed."}, status_code=status.HTTP_200_OK)
        if msg_ts:
            processed_messages.add(msg_ts)

    # Quick response to Slack
    conversation_id = parent_ts or msg_ts
    
    # Immediately add "robot_face" reaction (agent has taken the conversation)
    if channel and parent_ts:
        await reaction_manager.add_reaction_safe(channel, parent_ts, "robot_face")
    
    # Start background processing
    background_tasks.add_task(process_slack_conversation, payload, conversation_id, channel, parent_ts)
    
    return JSONResponse({"message": "Processing started", "conversation_id": conversation_id}, status_code=status.HTTP_200_OK)

async def handle_reaction_event(event, channel, parent_ts, payload):
    """Handle reaction events synchronously for quick response"""
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
        
        def thread_already_documented(thread_history):
            for msg in thread_history:
                if (
                    (msg.get("bot_id") or msg.get("user") == SLACK_BOT_USER_ID)
                    and "Documentation Update" in (msg.get("text", "") or "")
                ):
                    return True
            return False
        
        thread_history = fetch_thread_history(channel, thread_ts)
        theo = Theo()
        context_summary = theo.summarize_thread(thread_history)
        
        if thread_already_documented(thread_history):
            print(f"[INFO] Documentation already exists for thread {thread_ts}, skipping update.")
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
        
        send_slack_message(channel, "Thank you for concluding the conversation! :green_heart:\nI have updated our knowledge base with what I learned from this conversation.", thread_ts=thread_ts)
        return JSONResponse({"message": "Slack conversation concluded"}, status_code=status.HTTP_200_OK)
        
    elif reaction == "x" and thread_ts and channel:
        with concluded_threads_lock:
            concluded_threads.add(thread_ts)
        admin_id = os.getenv("ADMIN_SLACK_USER_ID")
        apology = f"We're sorry this didn't help. <@{admin_id}> will assist you shortly!"
        send_slack_message(channel, apology, thread_ts=thread_ts)
        return JSONResponse({"message": "Slack conversation concluded"}, status_code=status.HTTP_200_OK)
    
    return JSONResponse({"message": "Reaction processed"}, status_code=status.HTTP_200_OK)

async def process_slack_conversation(payload: dict, conversation_id: str, channel: str, parent_ts: str):
    """Background task to process Slack conversations with full async support"""
    print(f"[INFO] Starting background processing for conversation {conversation_id}")
    
    # Create conversation state
    conversation = await conversation_manager.create_conversation(conversation_id, channel, parent_ts)
    
    # Track last user message timestamp for cleanup (outside try block)
    last_user_ts = None
    current_progress_stage = None
    
    try:
        # Get last user message timestamp for loading reaction (async)
        loop = asyncio.get_event_loop()
        thread_history_for_reaction = await loop.run_in_executor(None, fetch_thread_history, channel, parent_ts)
        last_user_ts = get_last_user_message_ts(thread_history_for_reaction, SLACK_BOT_USER_ID)
        
        # Acquire processing slot (max 10 concurrent)
        async with conversation_manager.acquire_slot():
            await conversation_manager.update_conversation_status(conversation_id, ConversationStatus.PROCESSING)
            await conversation_manager.update_conversation_progress(conversation_id, "processing")
            current_progress_stage = "processing"
            
            # Add ⏳ (start processing, separate from 🤖)
            await reaction_manager.add_reaction_safe(channel, parent_ts, "hourglass_flowing_sand")
            
            # Add loading reaction to last user message
            if last_user_ts and last_user_ts != parent_ts:
                await reaction_manager.add_reaction_safe(channel, last_user_ts, "loading-circle")
            
            # Process with retry and timeout
            async def process_with_supervisor():
                nonlocal current_progress_stage
                
                with tracer.start_span("conversation.workflow", service="theo", resource="slack_event_async") as parent_span:
                    parent_span.set_tag("conversation.id", conversation_id)
                    parent_span.set_tag("event.type", "slack")
                    
                    # Stage 2: ⏳ → 🧠 (thinking)
                    await conversation_manager.update_conversation_progress(conversation_id, "thinking")
                    await reaction_manager.update_progress_reaction(
                        channel, parent_ts, "processing", "thinking", 
                        conversation_manager.progress_reactions
                    )
                    current_progress_stage = "thinking"
                    
                    # Stage 2: 🧠 → 🔍 (researching/fetching context)
                    await conversation_manager.update_conversation_progress(conversation_id, "researching")
                    await reaction_manager.update_progress_reaction(
                        channel, parent_ts, current_progress_stage, "researching", 
                        conversation_manager.progress_reactions
                    )
                    current_progress_stage = "researching"
                    
                    # Fetch thread history asynchronously to avoid blocking
                    loop = asyncio.get_event_loop()
                    thread_history = await loop.run_in_executor(None, fetch_thread_history, channel, parent_ts)
                    
                    # Stage 3: 🔍 → 📝 (writing/processing with supervisor)
                    await conversation_manager.update_conversation_progress(conversation_id, "writing")
                    await reaction_manager.update_progress_reaction(
                        channel, parent_ts, current_progress_stage, "writing", 
                        conversation_manager.progress_reactions
                    )
                    current_progress_stage = "writing"
                    
                    supervisor_response = await handle_event_with_supervisor(
                        "slack", payload, parent_span=parent_span, thread_history=thread_history
                    )
                    
                    answer = supervisor_response.get("result") or "No answer generated."
                    print(f"[DEBUG] Slack message to be sent: {answer}")
                    
                    if channel and answer and not answer.startswith("Ignoring bot's own message") and not answer.startswith("Bot not mentioned") and not answer.startswith("Ignoring Slack event type"):
                        # Send Slack message asynchronously
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, send_slack_message, channel, answer, parent_ts)
                    
                    return supervisor_response
            
            # Run with retry and timeout (reduced timeout to 45 seconds)
            supervisor_response = await with_retry(
                lambda: timeout_wrapper(process_with_supervisor(), 45),
                max_retries=2,
                conversation_id=conversation_id
            )
            
            # Mark as completed successfully
            await conversation_manager.complete_conversation(conversation_id)
            
            print(f"[INFO] Successfully completed conversation {conversation_id}")
            
    except asyncio.TimeoutError:
        # Handle timeout specifically
        await conversation_manager.fail_conversation(conversation_id, "Conversation timed out after 45 seconds")
        await reaction_manager.add_reaction_safe(channel, parent_ts, "internet-problems")
        
        admin_id = os.getenv("ADMIN_SLACK_USER_ID")
        error_msg = f"Sorry, this conversation timed out after 45 seconds. <@{admin_id}> will assist you shortly!"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_slack_message, channel, error_msg, parent_ts)
        
        print(f"[ERROR] Conversation {conversation_id} timed out")
        
    except Exception as e:
        # Handle general errors
        await conversation_manager.fail_conversation(conversation_id, str(e))
        await reaction_manager.add_reaction_safe(channel, parent_ts, "internet-problems")
        
        admin_id = os.getenv("ADMIN_SLACK_USER_ID")
        error_msg = f"Sorry, an error occurred while processing your request. <@{admin_id}> will assist you shortly!"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_slack_message, channel, error_msg, parent_ts)
        
        print(f"[ERROR] Error in conversation {conversation_id}: {e}")
        
    finally:
        # Always clean up reactions - remove all progress indicators
        if current_progress_stage:
            await reaction_manager.remove_all_progress_reactions(
                channel, parent_ts, conversation_manager.progress_reactions
            )
        
        # Remove loading reaction from last user message
        if last_user_ts and last_user_ts != parent_ts:
            await reaction_manager.remove_reaction_safe(channel, last_user_ts, "loading-circle")
        
        print(f"[INFO] Finished processing conversation {conversation_id}")

@app.post("/github/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    # Only handle push events to main, prod, or test branch
    if payload.get("ref") in ["refs/heads/main", "refs/heads/prod"] and "commits" in payload:
        def process_github_push(payload):
            theo = Theo()
            parent_code_commits_folder_id = os.getenv("CONFLUENCE_CODE_COMMITS_FOLDER_ID", "2141782054")
            commits = payload["commits"]
            push_info = {
                "pusher": payload.get("pusher", {}),
                "after": payload.get("after"),
                "before": payload.get("before"),
                "ref": payload.get("ref"),
                "repository": payload.get("repository", {}).get("full_name", "")
            }
            # Create a single Code Commit page for the push
            code_commit_result = theo.create_code_commit_from_github_push(commits, push_info, parent_folder_id=parent_code_commits_folder_id)
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

@app.get("/create-metabase-question")
async def create_metabase_question_endpoint(sql: str, title: str = None):
    """
    Create a Metabase question on-demand and redirect to it.
    This endpoint is called when users click '▶️ Run in Metabase' links.
    """
    try:
        from theo.tools.metabase import create_metabase_question
        
        # Decode the SQL if it's URL-encoded
        decoded_sql = unquote(sql)
        
        # Create the Metabase question
        question_url = create_metabase_question(decoded_sql, title)
        
        if question_url:
            # Redirect to the created question
            return RedirectResponse(url=question_url, status_code=302)
        else:
            # Fallback to Metabase new question page if creation fails
            metabase_base_url = os.getenv("METABASE_BASE_URL", "https://sunroom-rentals.metabaseapp.com")
            return RedirectResponse(url=f"{metabase_base_url}/question/new", status_code=302)
            
    except Exception as e:
        print(f"[ERROR] Failed to create Metabase question: {e}")
        # Fallback to Metabase new question page
        metabase_base_url = os.getenv("METABASE_BASE_URL", "https://sunroom-rentals.metabaseapp.com")
        return RedirectResponse(url=f"{metabase_base_url}/question/new", status_code=302)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.theo.api:app", host="0.0.0.0", port=port, reload=True) 