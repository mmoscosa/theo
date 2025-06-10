import os
print("[DEBUG] LLM CONFIG:")
print(f"  AGENT_SUPERVISOR_MODEL: {os.getenv('AGENT_SUPERVISOR_MODEL')}")
print(f"  AGENT_SUPPORT_ENGINEER_MODEL: {os.getenv('AGENT_SUPPORT_ENGINEER_MODEL')}")
print(f"  AGENT_TECHNICAL_WRITER_MODEL: {os.getenv('AGENT_TECHNICAL_WRITER_MODEL')}")
print(f"  AGENT_BI_ENGINEER_MODEL: {os.getenv('AGENT_BI_ENGINEER_MODEL')}")
print(f"  AGENT_PRODUCT_MANAGER_MODEL: {os.getenv('AGENT_PRODUCT_MANAGER_MODEL')}")
# Mask API keys for safety
openai_key = os.getenv('OPENAI_API_KEY')
print(f"  OPENAI_API_KEY: {'set' if openai_key else 'NOT SET'}{' (' + openai_key[:5] + '...' if openai_key else ''}")
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
print(f"  ANTHROPIC_API_KEY: {'set' if anthropic_key else 'NOT SET'}{' (' + anthropic_key[:5] + '...' if anthropic_key else ''}")
google_key = os.getenv('GOOGLE_API_KEY')
print(f"  GOOGLE_API_KEY: {'set' if google_key else 'NOT SET'}{' (' + google_key[:5] + '...' if google_key else ''}")
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
print(f"  AWS_ACCESS_KEY_ID: {'set' if aws_key else 'NOT SET'}{' (' + aws_key[:5] + '...' if aws_key else ''}")
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
print(f"  AWS_SECRET_ACCESS_KEY: {'set' if aws_secret else 'NOT SET'}{' (' + aws_secret[:5] + '...' if aws_secret else ''}")

print("[DEBUG] crew.py loaded")
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from theo.tools.bedrock import BedrockClient
from theo.tools.datadog import DatadogClient
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm
import os
from theo.tools.slack import format_slack_response
import re
import litellm
from theo.tools.jira import create_jira_ticket
from theo.tools.slack import send_slack_message
import logging
from theo.tools.confluence import update_confluence_page, create_confluence_page, get_all_confluence_pages, add_row_to_adr_index
import html
import requests
from datetime import datetime
from theo.config import load_yaml_config
import pytz
import sqlalchemy  # or your DB driver of choice
from theo.tools.metabase import create_metabase_question

# Global set to track which threads have updated TBIKB
updated_tbikb_threads = set()

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Theo():
    """Theo crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    
    # Override the default tasks config path to avoid looking for tasks.yaml
    tasks_config = None

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: We don't use YAML tasks - all tasks are handled via direct method calls
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def supervisor(self) -> Agent:
        return Agent(
            config=self.agents_config['supervisor'],
            verbose=True
        )

    @agent
    def support_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['support_engineer'],
            verbose=True
        )

    @agent
    def technical_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_writer'],
            verbose=True
        )

    @agent
    def bi_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['bi_engineer'],
            verbose=True
        )

    @agent
    def product_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['product_manager'],
            verbose=True
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supervisor_model = os.getenv("AGENT_SUPERVISOR_MODEL", "gemini/gemini-2.0-flash-exp")
        self.support_engineer_model = os.getenv("AGENT_SUPPORT_ENGINEER_MODEL", "anthropic/claude-3-5-sonnet-20240620")
        self.technical_writer_model = os.getenv("AGENT_TECHNICAL_WRITER_MODEL", "gpt-4o")
        self.bi_engineer_model = os.getenv("AGENT_BI_ENGINEER_MODEL", "anthropic/claude-3-5-sonnet-20240620")
        self.product_manager_model = os.getenv("AGENT_PRODUCT_MANAGER_MODEL", "gemini/gemini-2.0-flash-exp")
        # Load agent and prompt configs from YAML
        self.agents_config = load_yaml_config('agents.yaml')
        self.supervisor_prompts = load_yaml_config('supervisor_prompts.yaml')
        self.support_prompts = load_yaml_config('support_prompts.yaml')
        self.technical_writer_prompts = load_yaml_config('technical_writer_prompts.yaml')
        self.bi_engineer_prompts = load_yaml_config('bi_engineer_prompts.yaml')
        self.product_manager_prompts = load_yaml_config('product_manager_prompts.yaml')
        
        # Since we don't use YAML-defined tasks, create empty tasks list
        self.tasks = []

    def summarize_thread(self, thread_history, question=None):
        """Summarize a Slack thread using the supervisor's LLM model, and include relevant general knowledge from Bedrock only if useful."""
        model = os.getenv("AGENT_SUPERVISOR_MODEL", "unknown")
        bedrock = BedrockClient()
        def is_substantive(text):
            # Filter out greetings, sign-offs, and very short/irrelevant messages
            if not text or len(text.strip()) < 2:
                return False
            text_lower = text.strip().lower()
            greetings = ["hi", "hello", "hey", "thanks", "thank you", "ok", "cool", "bye", "goodbye", ":thumbsup:", ":wave:"]
            return not any(text_lower == g for g in greetings)
        def extract_message_text(msg):
            if msg.get("text"):
                return msg["text"]
            # Try to extract from blocks if text is empty
            if "blocks" in msg:
                for block in msg["blocks"]:
                    if block.get("type") == "rich_text":
                        for element in block.get("elements", []):
                            if element.get("type") == "rich_text_section":
                                return "".join(
                                    subel.get("text", "")
                                    for subel in element.get("elements", [])
                                    if "text" in subel
                                )
            return ""
        # Build thread_timeline: [timestamp] User: message
        timeline = []
        for msg in thread_history or []:
            text = extract_message_text(msg)
            if not is_substantive(text):
                continue
            ts = msg.get("ts")
            try:
                # Slack timestamps are floats as strings (epoch seconds)
                dt = datetime.fromtimestamp(float(ts)) if ts else None
                ts_str = dt.strftime("%Y-%m-%d %H:%M") if dt else ""
            except Exception:
                ts_str = ts or ""
            
            # Get display name with better logic
            display_name = (
                msg.get("user_profile", {}).get("display_name") or
                msg.get("user_profile", {}).get("real_name") or 
                msg.get("username") or
                msg.get("user") or
                ("Bot" if msg.get("bot_id") else "User")
            )
            
            # Handle Slack user ID format
            if display_name and display_name.startswith("U") and len(display_name) == 11:
                better_name = (
                    msg.get("user_profile", {}).get("display_name") or
                    msg.get("user_profile", {}).get("real_name") or
                    msg.get("username")
                )
                if better_name:
                    display_name = better_name
                else:
                    display_name = f"User_{display_name[-4:]}"
            
            timeline.append(f"[{ts_str}] {display_name}: {text.strip()}")
        thread_timeline = "\n".join(timeline)
        general_kb_query = question or thread_timeline
        general_knowledge = bedrock.search_general_knowledge(general_kb_query)
        print(f"[DEBUG] GeneralKnowledge KB result: {general_knowledge}")
        # Only include general knowledge if it is non-empty, not an error, and not a 'no results' message
        if (
            not general_knowledge
            or "no results found" in general_knowledge.lower()
            or "error" in general_knowledge.lower()
            or "not set" in general_knowledge.lower()
            or general_knowledge.strip() == ""
        ):
            general_knowledge_section = ""
        else:
            general_knowledge_section = self.supervisor_prompts["general_knowledge_section"].replace("{general_knowledge}", general_knowledge)
        @llm(model_name=model, name="supervisor_thread_summary", model_provider="openai")
        def _llm_call():
            prompt = self.supervisor_prompts["summary_prompt"] \
                .replace("{thread_timeline}", thread_timeline) \
                .replace("{general_knowledge_section}", general_knowledge_section)
            summary = ""
            try:
                summary = LLMObs.annotate(
                    input_data=[{"role": "user", "content": prompt}],
                    output_data=[{"role": "assistant", "content": ""}],
                    tags={"agent_role": "supervisor", "model_provider": "openai"}
                )
            except Exception:
                summary = None
            # Defensive: always return a real string, never None or 'None'
            if not summary or summary == "None":
                summary = thread_timeline[:1000]  # Use the actual messages as fallback
            return summary if isinstance(summary, str) else str(summary)
        return _llm_call()

    def supervisor_routing(self, question=None, conversation_id=None, thread_history=None, **kwargs):
        print(f"[DEBUG] supervisor_routing called with question={question}, conversation_id={conversation_id}, kwargs={kwargs}")
        print(f"[DEBUG] supervisor_routing thread_history: {thread_history}")
        datadog = DatadogClient()
        # Summarize thread history for context using LLM and include general knowledge
        context_summary = ""
        if thread_history:
            context_summary = self.summarize_thread(thread_history, question=question)
        # Defensive: always ensure context_summary is a real string
        if not context_summary or context_summary == "None":
            context_summary = ""
        print(f"[DEBUG] supervisor_routing context_summary: {context_summary}")
        # Build a role-specific prompt for the downstream agent
        result = self.supervisor_answer(question or kwargs.get('question', ''), conversation_id=conversation_id, context_summary=context_summary)
        print(f"[DEBUG] supervisor_answer raw return: {result}")
        answer, downstream_prompt = result
        print(f"[DEBUG] supervisor_routing raw answer: {answer}")
        answer_clean = answer.strip().lower() if isinstance(answer, str) else answer
        print(f"[DEBUG] supervisor_routing answer_clean: {answer_clean}")
        datadog.log_audit_event({
            "event_type": "supervisor_routing",
            "question": question,
            "answer": answer_clean,
            "conversation_id": conversation_id
        })
        # If supervisor_health, return the tuple directly
        if answer_clean in ("supervisor_health", "platform_health"):
            return result
        # For bi_report, documentation_update, adr_creation, and documentation_and_adr, pass both summary and raw thread
        if answer_clean in ["bi_report", "documentation_update", "adr_creation", "documentation_and_adr"]:
            print(f"[DEBUG] Passing to downstream agent: answer={answer_clean}, context_summary={context_summary}, thread_history={thread_history}")
            return {
                "task_name": answer_clean,
                "downstream_prompt": downstream_prompt,
                "context_summary": context_summary,
                "thread_history": thread_history
            }
        # Handle clarification_needed
        elif answer_clean == "clarification_needed":
            return f"I need more information to help you properly. Could you please clarify what you're looking for or provide more context about your request?"
        
        # For all other agents, pass only context_summary
        else:
            print(f"[DEBUG] Passing to downstream agent: answer={answer_clean}, context_summary only")
            return {
                "task_name": answer_clean,
                "downstream_prompt": downstream_prompt,
                "context_summary": context_summary
            } if answer_clean in ["support_request", "ticket_creation"] else answer_clean

    def _contains_sql(self, thread_history):
        sql_pattern = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\b", re.IGNORECASE)
        code_block_pattern = re.compile(r"```(.*?)```", re.DOTALL)
        inline_code_pattern = re.compile(r"`([^`]+)`")
        for msg in thread_history or []:
            text = msg.get("text", "") or ""
            # Check all code blocks (regardless of language)
            for match in code_block_pattern.finditer(text):
                if sql_pattern.search(match.group(1)):
                    return True
            # Check all inline code segments
            for match in inline_code_pattern.finditer(text):
                if sql_pattern.search(match.group(1)):
                    return True
            # Also check all lines for SQL keywords
            for line in text.splitlines():
                if sql_pattern.search(line):
                    return True
        return False

    def _extract_successful_sqls(self, thread_history):
        sql_blocks = []
        sql_pattern = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\b", re.IGNORECASE)
        code_block_pattern = re.compile(r"```(.*?)```", re.DOTALL)
        inline_code_pattern = re.compile(r"`([^`]+)`")
        for msg in thread_history or []:
            text = msg.get("text", "") or ""
            # Extract all code blocks and check for SQL
            for match in code_block_pattern.finditer(text):
                code = match.group(1).strip()
                if sql_pattern.search(code):
                    sql_blocks.append(code)
            # Extract all inline code segments and check for SQL
            for match in inline_code_pattern.finditer(text):
                code = match.group(1).strip()
                if sql_pattern.search(code):
                    sql_blocks.append(code)
            # Also extract lines that look like SQL
            for line in text.splitlines():
                if sql_pattern.match(line.strip()):
                    sql_blocks.append(line.strip())
        return sql_blocks

    def documentation_update(self, question=None, conversation_id=None, context_summary=None, thread_history=None, close_conversation=False, **kwargs):
        """
        If close_conversation is True, this is a :white_check_mark: event and should update TKB (General Knowledge) and TBIKB (if SQL present).
        If not, this is a documentation prompt and should update UP (main doc) and TBIKB (if SQL present).
        """
        print(f"[DEBUG] documentation_update called with question={question}, conversation_id={conversation_id}, context_summary={context_summary}, thread_history={thread_history}, close_conversation={close_conversation}, kwargs={kwargs}")
        datadog = DatadogClient()
        agent_role = "Technical Writer"
        model = os.getenv("AGENT_TECHNICAL_WRITER_MODEL", "unknown")
        model_provider = self.get_model_provider("AGENT_TECHNICAL_WRITER_MODEL", "AGENT_TECHNICAL_WRITER_PROVIDER")
        timeline = []
        if thread_history:
            for msg in thread_history:
                # Get display name with better logic
                display_name = (
                    msg.get("user_profile", {}).get("display_name") or
                    msg.get("user_profile", {}).get("real_name") or 
                    msg.get("username") or
                    msg.get("user") or
                    ("bot" if msg.get("bot_id") else "unknown")
                )
                
                # Handle Slack user ID format
                if display_name and display_name.startswith("U") and len(display_name) == 11:
                    better_name = (
                        msg.get("user_profile", {}).get("display_name") or
                        msg.get("user_profile", {}).get("real_name") or
                        msg.get("username")
                    )
                    if better_name:
                        display_name = better_name
                    else:
                        display_name = f"User_{display_name[-4:]}"
                
                text = msg.get("text", "").strip()
                ts = msg.get("ts", "")
                if text:
                    # Filter out agent signatures to prevent LLM from copying them
                    if "_Taken by" not in text:
                        timeline.append(f"[{ts}] {display_name}: {text}")
        timeline_str = "\n".join(timeline) if timeline else "No timeline available."
        doc_prompt = self.technical_writer_prompts["documentation_update"] \
            .replace("{timeline}", timeline_str) \
            .replace("{context_summary}", context_summary or "")
        import litellm
        from ddtrace.llmobs import LLMObs
        # Generate documentation content
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a technical writer."},
                    {"role": "user", "content": doc_prompt}
                ]
            )
            doc_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
            if not doc_content or doc_content == "None":
                doc_content = "Documentation could not be generated."
            # Ensure ~EOF~ is present at the end
            doc_content = doc_content.rstrip()
            if not doc_content.endswith("~EOF~"):
                doc_content += "\n~EOF~"
            else:
                # Remove any extra content after ~EOF~
                doc_content = doc_content[:doc_content.find("~EOF~") + 5]
            print(f"[DEBUG] Documentation content completed and ends with EOF marker")
            LLMObs.annotate(
                input_data=[{"role": "system", "content": "You are a technical writer."}, {"role": "user", "content": doc_prompt}],
                output_data=[{"role": "assistant", "content": doc_content}],
                tags={"agent_role": agent_role, "action": "generate_doc_content", "model_provider": model_provider}
            )
        except Exception as e:
            print(f"[ERROR] Technical Writer LLM call failed: {e}")
            doc_content = "Documentation could not be generated.\n~EOF~"
        # Generate a concise, descriptive page title
        title_prompt = self.technical_writer_prompts["title_generation"] \
            .replace("{doc_content}", doc_content or "")
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a technical writer."},
                    {"role": "user", "content": title_prompt}
                ]
            )
            page_title = response['choices'][0]['message']['content'] if response and 'choices' in response else None
            if page_title:
                page_title = re.sub(r'^[\'\"]+|[\'\"]+$', '', page_title.strip())
                # Clean up common LLM response patterns for title generation
                page_title = re.sub(r'^Here are.*?options.*?title.*?:', '', page_title, flags=re.IGNORECASE | re.DOTALL).strip()
                page_title = re.sub(r'^\d+\.\s*[\*\*]*', '', page_title).strip()  # Remove numbering like "1. **"
                page_title = re.sub(r'^\*\*([^*]+)\*\*.*', r'\1', page_title).strip()  # Extract just bold title
                page_title = re.sub(r'^([^.\n]+)\..*', r'\1', page_title, flags=re.DOTALL).strip()  # Take first sentence
                # If still too long or contains explanation text, take first meaningful part
                if len(page_title) > 80 or any(word in page_title.lower() for word in ['here are', 'options', 'why:', 'reason:']):
                    lines = page_title.split('\n')
                    for line in lines:
                        clean_line = re.sub(r'^\d+\.\s*[\*\*]*(.+?)[\*\*]*$', r'\1', line.strip())
                        if clean_line and len(clean_line) < 80 and not any(word in clean_line.lower() for word in ['here are', 'options', 'why:', 'reason:']):
                            page_title = clean_line
                            break
            if not page_title or page_title.strip().lower() == "none":
                page_title = question or 'New Documentation'
            LLMObs.annotate(
                input_data=[{"role": "system", "content": "You are a technical writer."}, {"role": "user", "content": title_prompt}],
                output_data=[{"role": "assistant", "content": page_title}],
                tags={"agent_role": agent_role, "action": "generate_doc_title", "model_provider": model_provider}
            )
        except Exception as e:
            print(f"[ERROR] Technical Writer Title LLM call failed: {e}")
            page_title = question or 'New Documentation'
        # SQL detection
        has_sql = self._contains_sql(thread_history)
        sql_blocks = self._extract_successful_sqls(thread_history) if has_sql else []
        # Defensive: Ensure conversation_id is set
        if conversation_id is None and thread_history:
            for msg in thread_history:
                if msg.get("thread_ts"):
                    conversation_id = msg["thread_ts"]
                    print(f"[DEBUG] Fallback: set conversation_id from thread_history: {conversation_id}")
                    break
        print(f"[DEBUG] has_sql: {has_sql}, sql_blocks: {sql_blocks}, conversation_id: {conversation_id}")
        # Main doc update (UP) if prompted
        if not close_conversation:
            from theo.tools.confluence import create_confluence_page
            print(f"[DEBUG] Creating/Updating main doc in UP space under documentation folder.")
            docs_folder_id = os.getenv("CONFLUENCE_DOCS_FOLDER_ID", "2141782055")
            confluence_response = create_confluence_page(
                page_title, 
                doc_content, 
                space_key=os.getenv("CONFLUENCE_SPACE_KEY_UP", "UP"),
                parent_id=docs_folder_id
            )
            # Build Confluence URL
            page_id = confluence_response.get("id")
            base_url = os.getenv("CONFLUENCE_BASE_URL")
            confluence_url = f"{base_url}/pages/viewpage.action?pageId={page_id}" if page_id and base_url else ""
            # Generate summary with <title:url> style
            summary = f"Documentation for <{confluence_url}|{page_title}> has been created/updated."
            slack_message = (
                ":books: Documentation Update\n\n"
                f"{summary}\n\n"
                "Taken by :memo: Technical Writer"
            )
        else:
            slack_message = None
        # BI KB update (TBIKB) if SQL present and not already updated
        if has_sql and sql_blocks and conversation_id is not None:
            global updated_tbikb_threads
            if conversation_id not in updated_tbikb_threads:
                print(f"[DEBUG] Updating TBIKB for conversation_id: {conversation_id}")
                from theo.tools.confluence import create_confluence_page
                unique_sqls = list(set(sql_blocks))
                sql_descriptions = []
                for sql in unique_sqls:
                    # Use LLM to generate a description
                    llm_desc_prompt = (
                        "You are a BI engineer. Given the following SQL query, write a one-sentence description of what it does, suitable for future reference.\n\nSQL:\n"
                        f"{sql}\n\nDescription:"
                    )
                    try:
                        response = litellm.completion(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a BI engineer."},
                                {"role": "user", "content": llm_desc_prompt}
                            ]
                        )
                        description = response['choices'][0]['message']['content'].strip() if response and 'choices' in response else "No description generated."
                    except Exception as e:
                        print(f"[ERROR] LLM call for SQL description failed: {e}")
                        description = "No description generated."
                    # Escape SQL for XML
                    sql_escaped = html.escape(sql)
                    sql_block = (
                        f'<p><strong>Description:</strong> {html.escape(description)}</p>'
                        f'<ac:structured-macro ac:name="code">'
                        f'<ac:parameter ac:name="language">sql</ac:parameter>'
                        f'<ac:plain-text-body><![CDATA[{sql}\n]]></ac:plain-text-body>'
                        f'</ac:structured-macro>'
                    )
                    sql_descriptions.append(sql_block)
                sql_doc = "\n\n".join(sql_descriptions)
                sql_title = f"SQLs for: {page_title}"
                print(f"[DEBUG] Creating/Updating BI KB in TBIKB space.")
                create_confluence_page(sql_title, sql_doc, space_key=os.getenv("CONFLUENCE_SPACE_KEY_TBIKB", "TBIKB"))
                updated_tbikb_threads.add(conversation_id)
            else:
                print(f"[DEBUG] Skipping TBIKB update for conversation_id: {conversation_id} (already updated)")
        # General Knowledge update (TKB) if closing conversation
        if close_conversation:
            # Defensive: If context_summary is None, 'None', or empty, build a fallback summary from thread_history
            if not context_summary or str(context_summary).strip().lower() == "none":
                if thread_history:
                    fallback_msgs = []
                    for msg in thread_history[-5:]:
                        user = msg.get("user") or msg.get("username") or ("bot" if msg.get("bot_id") else "unknown")
                        text = msg.get("text", "").strip()
                        if text:
                            fallback_msgs.append(f"{user}: {text}")
                    context_summary = "\n".join(fallback_msgs) if fallback_msgs else "No summary available."
                else:
                    context_summary = "No summary available."
            from theo.tools.confluence import create_confluence_page
            summary_title = f"Summary for: {page_title}"
            summary_content = f"Summary of conversation for supervisor context:\n\n{context_summary}\n\nKey points:\n{doc_content}"
            # Ensure ~EOF~ at the end of summary_content
            summary_content = summary_content.rstrip()
            if not summary_content.endswith("~EOF~"):
                summary_content += "\n~EOF~"
            else:
                summary_content = summary_content[:summary_content.find("~EOF~") + 5]
            print(f"[DEBUG] Creating/Updating General Knowledge in TKB space with EOF marker.")
            create_confluence_page(summary_title, summary_content, space_key=os.getenv("CONFLUENCE_SPACE_KEY_TKB", "TKB"))
        # Return the correct Slack message
        if not close_conversation:
            return slack_message
        else:
            return f"Documentation updated. Main doc: False, BI KB: {has_sql}, General KB: True"

    def bi_report(self, question=None, conversation_id=None, context_summary=None, thread_history=None, **kwargs):
        print(f"[DEBUG] bi_report called with question={question}, conversation_id={conversation_id}, context_summary={context_summary}, thread_history={thread_history}, kwargs={kwargs}")
        datadog = DatadogClient()
        answer = self.bi_engineer_answer(question or kwargs.get('question', ''), conversation_id=conversation_id, context_summary=context_summary, thread_history=thread_history)
        print(f"[DEBUG] bi_report answer: {answer}")
        datadog.log_audit_event({
            "event_type": "bi_report",
            "question": question,
            "answer": answer,
            "conversation_id": conversation_id
        })
        return answer

    def ticket_creation(self, question=None, conversation_id=None, **kwargs):
        print(f"[DEBUG] ticket_creation called with question={question}, conversation_id={conversation_id}, kwargs={kwargs}")
        datadog = DatadogClient()
        # Pass channel and thread_ts to product_manager_answer
        answer = self.product_manager_answer(
            question or kwargs.get('question', ''),
            conversation_id=conversation_id,
            thread_history=kwargs.get('thread_history'),
            channel=kwargs.get('channel'),
            thread_ts=kwargs.get('thread_ts')
        )
        print(f"[DEBUG] ticket_creation answer: {answer}")
        datadog.log_audit_event({
            "event_type": "ticket_creation",
            "question": question,
            "answer": answer,
            "conversation_id": conversation_id
        })
        return answer

    def adr_creation(self, question=None, conversation_id=None, context_summary=None, thread_history=None, **kwargs):
        print(f"[DEBUG] adr_creation called with question={question}, conversation_id={conversation_id}, context_summary={context_summary}, thread_history={thread_history}, kwargs={kwargs}")
        datadog = DatadogClient()
        
        # Create ADR from conversation
        adr_result = self.create_adr_from_conversation(
            question=question,
            conversation_id=conversation_id,
            context_summary=context_summary,
            thread_history=thread_history,
            **kwargs
        )
        
        # Format the response message
        page_title = adr_result.get("adr_title", "New ADR")
        page_url = adr_result.get("adr_page_url", "")
        authors_info = adr_result.get("authors_info", "Unknown")
        
        # Check if ADR creation failed
        if "error" in adr_result:
            error_msg = adr_result.get("error", "Unknown error")
            slack_message = (
                f":warning: *ADR Creation Failed*\n\n"
                f"Failed to create Architecture Decision Record: {page_title}\n\n"
                f"*Error:* {error_msg}\n\n"
                f"Please check Confluence configuration or contact admin.\n\n"
                f"_Attempted by :memo: Technical Writer_"
            )
        else:
            slack_message = (
                f":memo: *ADR Created*\n\n"
                f"Architecture Decision Record <{page_url}|{page_title}> has been created and added to the ADR Index.\n\n"
                f"*Status:* Proposed\n"
                f"*Contributors:* {authors_info}\n\n"
                f"_Taken by :memo: Technical Writer_"
            )
        
        print(f"[DEBUG] adr_creation result: {slack_message}")
        datadog.log_audit_event({
            "event_type": "adr_creation",
            "question": question,
            "adr_title": page_title,
            "adr_url": page_url,
            "conversation_id": conversation_id
        })
        return slack_message

    def documentation_and_adr(self, question=None, conversation_id=None, context_summary=None, thread_history=None, **kwargs):
        """
        Create both documentation AND an ADR based on a conversation/request.
        """
        print(f"[DEBUG] documentation_and_adr called with question={question}, conversation_id={conversation_id}, context_summary={context_summary}, thread_history={thread_history}, kwargs={kwargs}")
        datadog = DatadogClient()
        
        # Create documentation first
        doc_result = self.documentation_update(
            question=question,
            conversation_id=conversation_id,
            context_summary=context_summary,
            thread_history=thread_history,
            close_conversation=False,  # This is a documentation prompt, not a conversation close
            **kwargs
        )
        
        # Create ADR second
        adr_result = self.create_adr_from_conversation(
            question=question,
            conversation_id=conversation_id,
            context_summary=context_summary,
            thread_history=thread_history,
            **kwargs
        )
        
        # Format combined response message
        adr_title = adr_result.get("adr_title", "New ADR")
        adr_url = adr_result.get("adr_page_url", "")
        authors_info = adr_result.get("authors_info", "Unknown")
        
        # Extract documentation URL from doc_result (it's in the format of slack message)
        doc_url_match = re.search(r'<([^|>]+)\|([^>]+)>', doc_result) if isinstance(doc_result, str) else None
        doc_url = doc_url_match.group(1) if doc_url_match else ""
        doc_title = doc_url_match.group(2) if doc_url_match else "Documentation"
        
        slack_message = (
            f":books: *Documentation & ADR Created*\n\n"
            f"Documentation: <{doc_url}|{doc_title}>\n"
            f"ADR: <{adr_url}|{adr_title}> (added to ADR Index)\n\n"
            f"*ADR Status:* Proposed\n"
            f"*Contributors:* {authors_info}\n\n"
            f"_Taken by :memo: Technical Writer_"
        )
        
        print(f"[DEBUG] documentation_and_adr result: {slack_message}")
        datadog.log_audit_event({
            "event_type": "documentation_and_adr",
            "question": question,
            "doc_url": doc_url,
            "adr_title": adr_title,
            "adr_url": adr_url,
            "conversation_id": conversation_id
        })
        return slack_message

    def support_request(self, question=None, conversation_id=None, context_summary=None, thread_history=None, **kwargs):
        print(f"[DEBUG] support_request called with question={question}, conversation_id={conversation_id}, kwargs={kwargs}")
        # Always ensure context_summary is a string and thread_history is a list
        context_summary = (context_summary or kwargs.get('context_summary') or "")
        thread_history = (thread_history or kwargs.get('thread_history') or [])
        datadog = DatadogClient()
        answer = self.support_engineer_answer(
            question or kwargs.get('question', ''),
            conversation_id=conversation_id,
            context_summary=context_summary,
            thread_history=thread_history
        )
        print(f"[DEBUG] support_request answer: {answer}")
        datadog.log_audit_event({
            "event_type": "support_request",
            "question": question,
            "answer": answer,
            "conversation_id": conversation_id
        })
        return answer

    @crew
    def crew(self) -> Crew:
        """Creates the Theo crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks or [], # Empty list since we handle tasks via direct method calls
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

    def support_engineer_answer(self, question, conversation_id=None, context_summary=None, thread_history=None):
        import litellm
        from datetime import datetime
        import pytz
        bedrock = BedrockClient()
        print(f"[DEBUG] Bedrock KB IDs: code_doc={bedrock.code_doc_kb}, db_schema={bedrock.db_schema_kb}, general={bedrock.general_kb}")
        agent_role = self.agents_config['support_engineer']['role']
        agent_emoji = "👨‍💻"
        category_emoji = "📄"
        category_title = "Support Request"
        model = self.support_engineer_model
        model_provider = self.get_model_provider("AGENT_SUPPORT_ENGINEER_MODEL", "AGENT_SUPPORT_ENGINEER_PROVIDER")
        context_summary_clean = "" if context_summary in (None, "None") else context_summary
        
        # Get current date/time in Central Time first
        tz = pytz.timezone("America/Chicago")
        now = datetime.now(tz)
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M %Z")
        
        # Enhance search query with time context detection
        enhanced_question = self._enhance_question_with_time_context(question, current_date)
        print(f"[DEBUG] Original question: {question}")
        print(f"[DEBUG] Enhanced question: {enhanced_question}")
        
        try:
            docs_result = bedrock.search_code_documentation(enhanced_question)
            print(f"[DEBUG] Bedrock docs result type: {type(docs_result)}")
            print(f"[DEBUG] Bedrock docs result: {docs_result}")
            
            # Fallback: if enhanced search doesn't find recent info, try basic search
            if isinstance(docs_result, dict):
                sources = docs_result.get("sources", [])
                # Check if we have any sources from the current month
                current_month = current_date[:7]  # "2025-06"
                has_recent_sources = any(current_month in str(source) for source in sources if source)
                
                if not has_recent_sources:
                    print(f"[DEBUG] Enhanced search didn't find recent sources, trying basic search...")
                    fallback_result = bedrock.search_code_documentation(question)
                    print(f"[DEBUG] Fallback search result: {fallback_result}")
                    
                    # Use fallback if it has more recent sources
                    if isinstance(fallback_result, dict):
                        fallback_sources = fallback_result.get("sources", [])
                        fallback_has_recent = any(current_month in str(source) for source in fallback_sources if source)
                        if fallback_has_recent:
                            print(f"[DEBUG] Using fallback search results as they contain more recent information")
                            docs_result = fallback_result
        except Exception as e:
            print(f"[ERROR] Bedrock search_code_documentation failed: {e}")
            docs_result = f"[ERROR] Bedrock search failed: {str(e)}"
        
        # Handle both old string format and new dict format for backward compatibility
        if isinstance(docs_result, dict):
            docs = docs_result.get("content", "")
            sources = docs_result.get("sources", [])
            print(f"[DEBUG] Extracted sources from dict: {sources}")
            print(f"[DEBUG] Sources type: {type(sources)}, length: {len(sources) if sources else 0}")
            
            # Extract and sort dates from sources to help LLM understand chronology
            source_dates = []
            for source in sources:
                if source and "Stand-Up+Notes+" in source:
                    # Extract date from URL like "Daily+Stand-Up+Notes+6+3+25"
                    import re
                    date_match = re.search(r'Stand-Up\+Notes\+(\d+)\+(\d+)\+(\d+)', source)
                    if date_match:
                        month, day, year = date_match.groups()
                        # Convert to full year format for comparison
                        full_year = f"20{year}" if len(year) == 2 else year
                        date_str = f"{full_year}-{month.zfill(2)}-{day.zfill(2)}"
                        source_dates.append(date_str)
            
            if source_dates:
                source_dates.sort(reverse=True)  # Most recent first
                most_recent_date = source_dates[0]
                print(f"[DEBUG] Source dates found: {source_dates}, most recent: {most_recent_date}")
                
                # Add prominent chronological context to docs content
                docs = f"🗓️ **CHRONOLOGICAL CONTEXT - READ THIS FIRST:**\n" \
                       f"Sources available: {', '.join(source_dates)}\n" \
                       f"MOST RECENT DATE: {most_recent_date}\n" \
                       f"USE INFORMATION FROM {most_recent_date} FIRST!\n\n{docs}"
        else:
            docs = docs_result
            sources = []
            print(f"[DEBUG] Docs result is string format, no sources available")
        # Use YAML prompt for support
        llm_prompt = self.support_prompts["main_prompt"] \
            .replace("{question}", question or "") \
            .replace("{documentation}", docs or "") \
            .replace("{context_summary}", context_summary_clean or "") \
            .replace("{current_date}", current_date) \
            .replace("{current_time}", current_time)
        print(f"[DEBUG] LLM prompt sent to support engineer:\n{llm_prompt}")
        from ddtrace.llmobs import LLMObs
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": agent_role},
                    {"role": "user", "content": llm_prompt}
                ]
            )
            main_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
            if not main_content or not isinstance(main_content, str) or main_content.strip().lower() == "none":
                main_content = self.support_prompts["no_answer_found"]
            
            # Validate: Support agent should not create tickets or documentation
            if any(phrase in main_content.lower() for phrase in ['create ticket', 'jira ticket', 'new ticket', 'i will create', 'creating documentation']):
                print(f"[ERROR] Support agent tried to create tickets/docs: {main_content[:100]}...")
                main_content = "I can help with technical support questions. For ticket creation or documentation, please ask explicitly for those services."
            LLMObs.annotate(
                input_data=[{"role": "system", "content": agent_role}, {"role": "user", "content": llm_prompt}],
                output_data=[{"role": "assistant", "content": main_content}],
                tags={"agent_role": agent_role, "model_provider": model_provider}
            )
            print(f"[DEBUG] LLM raw output: {main_content}")
            main_content = convert_markdown_to_slack(main_content)
            
            # Always format JSON examples as code blocks for Slack
            import re
            def format_json_code_blocks(text):
                # Find JSON blocks (simple heuristic: lines that look like JSON)
                json_pattern = re.compile(r'(\{\s*\n(?:[^\n]*\n)+?\})', re.MULTILINE)
                def replacer(match):
                    code = match.group(1)
                    # Remove any language hint from code block if present
                    code = re.sub(r'^```[a-zA-Z]+\n', '```\n', code, flags=re.MULTILINE)
                    # Only wrap if not already in a code block
                    if not code.strip().startswith('```'):
                        return f'```\n{code.strip()}\n```'
                    return code
                return json_pattern.sub(replacer, text)
            main_content = format_json_code_blocks(main_content)
            
            # Suppress sources if answer is general knowledge or not found in docs
            general_knowledge_phrases = [
                "no direct mention in provided documentation",
                "general knowledge",
                "not found in documentation",
                "not found in docs",
                "no documentation found",
                "no relevant documentation",
                "not available in documentation"
            ]
            if any(phrase in main_content.lower() for phrase in general_knowledge_phrases):
                # Remove any existing Sources line
                main_content = re.sub(r"Sources:.*", "", main_content)
            elif sources and any(source for source in sources if source):
                valid_sources = [source for source in sources if source and source != "[No content in results]"]
                print(f"[DEBUG] Valid sources found: {valid_sources}")
                if valid_sources:
                    source_links = []
                    for i, source in enumerate(valid_sources, 1):
                        print(f"[DEBUG] Processing source {i}: {source}")
                        if source and source.startswith("http"):
                            source_links.append(f"<{source}|[{i}]>")
                            print(f"[DEBUG] Added HTTP link: <{source}|[{i}]>")
                        else:
                            source_links.append(f"[{i}]")
                            print(f"[DEBUG] Added non-HTTP reference: [{i}] for source: {source}")
                    if source_links:
                        import re
                        sources_pattern = r'Sources:\s*(\[[0-9,\s\[\]]+\]|\[[0-9]+\](?:\s*\[[0-9]+\])*)'
                        if re.search(sources_pattern, main_content):
                            main_content = re.sub(sources_pattern, f"Sources: {' '.join(source_links)}", main_content)
                            print(f"[DEBUG] Replaced existing sources line with: Sources: {' '.join(source_links)}")
                        else:
                            main_content += f"\n\nSources: {' '.join(source_links)}"
                            print(f"[DEBUG] Added new sources line: Sources: {' '.join(source_links)}")
            else:
                print(f"[DEBUG] No valid sources to display")
                import re
                sources_pattern = r'Sources:\s*(\[[0-9,\s\[\]]+\]|\[[0-9]+\](?:\s*\[[0-9]+\])*)'
                if re.search(sources_pattern, main_content):
                    main_content = re.sub(sources_pattern, "Sources: None found", main_content)
                    print(f"[DEBUG] Replaced placeholder sources with 'Sources: None found'")
                elif "Sources: [1]" in main_content:
                    main_content = main_content.replace("Sources: [1]", "Sources: None found")
                    print(f"[DEBUG] Replaced 'Sources: [1]' with 'Sources: None found'")
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            main_content = self.support_prompts["no_answer_found"]
        return format_slack_response(
            category_emoji=category_emoji,
            category_title=category_title,
            main_content=main_content if isinstance(main_content, str) else str(main_content),
            agent_role=agent_role,
            agent_emoji=agent_emoji
        )

    def _extract_readable_title_from_url(self, url):
        """
        Extract a human-readable title from a Confluence URL.
        
        Examples:
        - 'Daily+Stand-Up+Notes+5+27+25' -> 'Stand-up 5/27/25'
        - 'System+Refactor+Initiative' -> 'System Refactor Initiative'
        """
        import re
        from urllib.parse import unquote
        
        try:
            # Extract the page title from URL (after last slash, before any query params)
            if '/pages/' in url:
                # Extract page title part: /pages/123456789/Page+Title+Here
                page_part = url.split('/pages/')[-1]
                if '/' in page_part:
                    title_part = page_part.split('/', 1)[1]  # Get part after page ID
                else:
                    title_part = page_part
                
                # Remove query parameters
                if '?' in title_part:
                    title_part = title_part.split('?')[0]
                
                # URL decode and clean up
                title = unquote(title_part.replace('+', ' '))
                
                # Special handling for standup notes
                standup_match = re.search(r'Daily Stand-Up Notes (\d+) (\d+) (\d+)', title, re.IGNORECASE)
                if standup_match:
                    month, day, year = standup_match.groups()
                    return f"Stand-up {month}/{day}/{year}"
                
                # General cleanup - limit length and clean up
                title = re.sub(r'\s+', ' ', title).strip()
                if len(title) > 35:
                    title = title[:32] + "..."
                
                return title if title else "Document"
        except Exception:
            pass
        
        # Fallback to domain name or generic title
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            if 'atlassian.net' in domain:
                return "Confluence Doc"
            return domain
        except Exception:
            return "Document"

    def _enhance_question_with_time_context(self, question, current_date):
        """
        Enhance the search question with time context for better Bedrock results.
        
        For current time references (this week, today, recently), add current date context.
        For specific time references (week of May 27th), keep the specific timeframe.
        """
        if not question:
            return question
            
        question_lower = question.lower()
        
        # Current time references - add current date context
        current_time_keywords = [
            "this week", "today", "recently", "current", "now", "currently", 
            "latest", "new", "recent", "this sprint", "this month"
        ]
        
        if any(keyword in question_lower for keyword in current_time_keywords):
            # Add current date context but make search more inclusive
            enhanced = f"{question} standup notes recent updates current date {current_date}"
            return enhanced
        
        # Specific time references - extract and use the specific timeframe
        import re
        
        # Look for specific dates, weeks, months
        date_patterns = [
            r"week of (\w+ \d+)",  # "week of May 27"
            r"(\w+ \d+)",          # "May 27"
            r"last (\w+)",         # "last week", "last month"
            r"(\w+day)",           # "Monday", "Tuesday", etc.
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                # Include the specific timeframe in the search
                timeframe = matches[0]
                enhanced = f"{question} {timeframe} standup notes"
                return enhanced
        
        # Default: return original question
        return question

    def _is_heartbeat_question(self, question):
        import litellm
        if not question:
            return False
        model = os.getenv("AGENT_SUPERVISOR_MODEL", "gpt-4o")
        prompt = (
            "A user sent the following message:\n"
            f"\"{question.strip()}\"\n"
            "Is this message asking if the platform, system, or bot is online, working, alive, available, or requesting a status/heartbeat check? "
            "Respond with only 'yes' or 'no'."
        )
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an intent classifier."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response['choices'][0]['message']['content'].strip().lower()
            result = answer.startswith("yes")
            print(f"[DEBUG] LLM heartbeat detection for '{question}': {result} (raw: {answer})")
            return result
        except Exception as e:
            print(f"[ERROR] LLM heartbeat detection failed: {e}")
            return False

    def supervisor_answer(self, question, conversation_id=None, context_summary=None):
        from theo.tools.slack import format_slack_response
        agent_role = self.agents_config['supervisor']['role']
        agent_emoji = ":shield:"
        category_emoji = ":satellite_antenna:"
        category_title = "Platform Health"
        model = self.supervisor_model
        model_provider = self.get_model_provider("AGENT_SUPERVISOR_MODEL", "AGENT_SUPERVISOR_PROVIDER")
        context_summary = context_summary or ""
        valid_tasks = {"support_request", "documentation_update", "adr_creation", "documentation_and_adr", "bi_report", "ticket_creation", "platform_health", "supervisor_health", "clarification_needed"}

        print(f"[DEBUG] supervisor_answer called with question='{question}'")
        
        # Heartbeat/health check interception
        print(f"[DEBUG] Checking if '{question}' is a heartbeat question...")
        is_heartbeat = self._is_heartbeat_question(question)
        print(f"[DEBUG] Heartbeat check result: {is_heartbeat}")
        
        if is_heartbeat:
            heartbeat_prompt = self.supervisor_prompts["heartbeat_prompt"]
            prompt = heartbeat_prompt.replace("{user_message}", question or "")
            import litellm
            from ddtrace.llmobs import LLMObs
            try:
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                main_content = response['choices'][0]['message']['content'] if response and 'choices' in response else self.supervisor_prompts["heartbeat_response"]
                LLMObs.annotate(
                    input_data=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                    output_data=[{"role": "assistant", "content": main_content}],
                    tags={"agent_role": agent_role, "model_provider": model_provider, "action": "heartbeat_response"}
                )
            except Exception as e:
                main_content = self.supervisor_prompts["heartbeat_response"]
            slack_message = format_slack_response(
                category_emoji=category_emoji,
                category_title=category_title,
                main_content=main_content,
                agent_role=agent_role,
                agent_emoji=agent_emoji
            )
            # Always return both keys for compatibility
            return ("platform_health", slack_message)
        prompt = self.supervisor_prompts["routing_prompt"] \
            .replace('{user_message}', question or '') \
            .replace('{context_summary}', context_summary or '')
        print(f"[DEBUG] Supervisor LLM full prompt:\n{prompt}")
        import litellm
        from ddtrace.llmobs import LLMObs
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an agent router. You must respond with exactly one valid routing option."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response['choices'][0]['message']['content'] if response and 'choices' in response else None
            if answer:
                answer = answer.strip().lower()
                # Additional validation: ensure single action
                if ' and ' in answer or ',' in answer or '\n' in answer:
                    print(f"[ERROR] LLM tried to return multiple actions: {answer}")
                    answer = "clarification_needed"
            else:
                answer = "clarification_needed"
            LLMObs.annotate(
                input_data=[{"role": "system", "content": "You are an agent router. You must respond with exactly one valid routing option."}, {"role": "user", "content": prompt}],
                output_data=[{"role": "assistant", "content": answer}],
                tags={"agent_role": agent_role, "model_provider": model_provider}
            )
            print(f"[DEBUG] Supervisor LLM raw output: {answer}")
            if answer not in valid_tasks:
                print(f"[ERROR] LLM returned invalid task: {answer}")
                answer = "clarification_needed"
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            answer = "clarification_needed"
        if answer == "clarification_needed" and context_summary:
            lowered_context = context_summary.lower()
            lowered_question = (question or '').lower()
            
            # Only override if the CURRENT QUESTION explicitly mentions ticket-related keywords
            explicit_ticket_keywords = ["create ticket", "new ticket", "make ticket", "ticket for", "jira ticket"]
            current_ticket_request = any(kw in lowered_question for kw in explicit_ticket_keywords)
            
            # Only override for support if current question is clearly a support question
            support_question_patterns = ["how to", "how do", "what is", "why", "where", "when", "how can"]
            current_support_question = any(pattern in lowered_question for pattern in support_question_patterns)
            
            if current_ticket_request:
                print(f"[DEBUG] Heuristic fallback: current question explicitly requests ticket creation, overriding to ticket_creation")
                answer = "ticket_creation"
            elif current_support_question:
                print(f"[DEBUG] Heuristic fallback: current question is a support question, overriding to support_request")
                answer = "support_request"
            else:
                print(f"[DEBUG] Keeping clarification_needed - no clear override pattern detected")
                # Keep as clarification_needed if no clear pattern
        downstream_prompt = f"Conversation context (summarized thread):\n{context_summary}\n\nUser request: {question}"
        if answer == "platform_health":
            # Always use the heartbeat response for platform health
            main_content = self.supervisor_prompts["heartbeat_response"]
            slack_message = format_slack_response(
                category_emoji=category_emoji,
                category_title=category_title,
                main_content=main_content,
                agent_role=agent_role,
                agent_emoji=agent_emoji
            )
            return ("platform_health", slack_message)
        return answer, downstream_prompt

    def technical_writer_answer(self, question, conversation_id=None, context_summary=None, thread_history=None):
        print(f"[DEBUG] technical_writer_answer called with question={question}, conversation_id={conversation_id}, context_summary={context_summary}, thread_history={thread_history}")
        agent_role = self.agents_config['technical_writer']['role']
        agent_emoji = "📝"
        category_emoji = "📚"
        category_title = "Documentation Update"
        model = self.technical_writer_model
        @llm(model_name=model, name="technical_writer_answer", model_provider="anthropic")
        def _llm_call():
            timeline = []
            if thread_history:
                for msg in thread_history:
                    # Get display name with better logic
                    display_name = (
                        msg.get("user_profile", {}).get("display_name") or
                        msg.get("user_profile", {}).get("real_name") or 
                        msg.get("username") or
                        msg.get("user") or
                        ("bot" if msg.get("bot_id") else "unknown")
                    )
                    
                    # Handle Slack user ID format
                    if display_name and display_name.startswith("U") and len(display_name) == 11:
                        better_name = (
                            msg.get("user_profile", {}).get("display_name") or
                            msg.get("user_profile", {}).get("real_name") or
                            msg.get("username")
                        )
                        if better_name:
                            display_name = better_name
                        else:
                            display_name = f"User_{display_name[-4:]}"
                    
                    text = msg.get("text", "").strip()
                    ts = msg.get("ts", "")
                    if text:
                        # Filter out agent signatures to prevent LLM from copying them
                        if "_Taken by" not in text:
                            timeline.append(f"[{ts}] {display_name}: {text}")
            timeline_str = "\n".join(timeline) if timeline else "No timeline available."
            # Use YAML task description for documentation update
            base_prompt = self.supervisor_prompts["routing_prompt"]
            llm_prompt = (
                f"{base_prompt}\n\n"
                f"User question or documentation request: {question}\n"
                f"Summary of conversation context: {context_summary}\n"
                f"Conversation timeline:\n{timeline_str}\n"
                f"Generate a clear, user-friendly documentation update or summary for the user."
            )
            print(f"[DEBUG] technical_writer_answer LLM prompt: {llm_prompt}")
            import litellm
            from ddtrace.llmobs import LLMObs
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a technical writer."},
                    {"role": "user", "content": llm_prompt}
                ]
            )
            main_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
            if not main_content or not isinstance(main_content, str) or main_content.strip().lower() == "none":
                main_content = "I apologize, but I couldn't generate a proper response. Please try rephrasing your request."
            LLMObs.annotate(
                input_data=[{"role": "system", "content": "You are a technical writer."}, {"role": "user", "content": llm_prompt}],
                output_data=[{"role": "assistant", "content": main_content}],
                tags={"agent_role": agent_role, "conversation_id": conversation_id or "unknown", "model_provider": "anthropic"}
            )
            print(f"[DEBUG] technical_writer_answer LLM main_content: {main_content}")
            slack_message = format_slack_response(
                category_emoji=category_emoji,
                category_title=category_title,
                main_content=main_content if isinstance(main_content, str) else str(main_content),
                agent_role=agent_role,
                agent_emoji=agent_emoji
            )
            print(f"[DEBUG] technical_writer_answer Slack message: {slack_message}")
            return slack_message
        return _llm_call()

    def bi_engineer_answer(self, question, conversation_id=None, context_summary=None, thread_history=None):
        print(f"[DEBUG] bi_engineer_answer called with question={question}, conversation_id={conversation_id}, context_summary={context_summary}, thread_history={thread_history}")
        agent_role = self.agents_config['bi_engineer']['role']
        agent_emoji = "📊"
        category_emoji = "📈"
        category_title = "BI Report"
        model = self.bi_engineer_model
        model_provider = self.get_model_provider("AGENT_BI_ENGINEER_MODEL")

        def _llm_call():
            # Build the LLM prompt using the same pattern as other agents
            prompt = self.bi_engineer_prompts.get("main_prompt", "You are a BI Engineer. Analyze the data request and provide SQL queries and insights.")
            
            # Replace placeholders with actual values
            if "{question}" in prompt:
                prompt = prompt.replace("{question}", question or "")
            if "{context_summary}" in prompt:
                prompt = prompt.replace("{context_summary}", context_summary or "")
            
            # If thread_history is needed, build timeline
            timeline = []
            if thread_history:
                for msg in thread_history:
                    # Get display name with better logic
                    display_name = (
                        msg.get("user_profile", {}).get("display_name") or
                        msg.get("user_profile", {}).get("real_name") or 
                        msg.get("username") or
                        msg.get("user") or
                        ("bot" if msg.get("bot_id") else "unknown")
                    )
                    
                    # Handle Slack user ID format
                    if display_name and display_name.startswith("U") and len(display_name) == 11:
                        better_name = (
                            msg.get("user_profile", {}).get("display_name") or
                            msg.get("user_profile", {}).get("real_name") or
                            msg.get("username")
                        )
                        if better_name:
                            display_name = better_name
                        else:
                            display_name = f"User_{display_name[-4:]}"
                    
                    text = msg.get("text", "").strip()
                    ts = msg.get("ts", "")
                    if text and "_Taken by" not in text:  # Filter out agent signatures
                        timeline.append(f"[{ts}] {display_name}: {text}")
            timeline_str = "\n".join(timeline) if timeline else "No timeline available."
            
            if "{timeline}" in prompt:
                prompt = prompt.replace("{timeline}", timeline_str)
            
            print(f"[DEBUG] bi_engineer_answer LLM prompt: {prompt}")
            
            import litellm
            from ddtrace.llmobs import LLMObs
            
            try:
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": agent_role},
                        {"role": "user", "content": prompt}
                    ]
                )
                main_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
                
                if not main_content or not isinstance(main_content, str) or main_content.strip().lower() == "none":
                    main_content = "I apologize, but I couldn't generate a proper response. Please try rephrasing your request."
                
                LLMObs.annotate(
                    input_data=[{"role": "system", "content": agent_role}, {"role": "user", "content": prompt}],
                    output_data=[{"role": "assistant", "content": main_content}],
                    tags={"agent_role": "BI Engineer", "conversation_id": conversation_id or "unknown", "model_provider": model_provider}
                )
                
            except Exception as e:
                print(f"[ERROR] BI Engineer LLM call failed: {e}")
                main_content = "I apologize, but I couldn't generate a proper response due to a technical issue. Please try again."
            
            # Only search for existing Metabase questions if user specifically asks for them
            if self._is_asking_for_existing_metabase_questions(question):
                try:
                    from theo.tools.metabase import search_metabase_questions
                    from theo.tools.slack import format_slack_response
                    
                    # Extract key terms from the question for search
                    search_terms = self._generate_search_query(question)
                    print(f"[DEBUG] Searching for existing questions with terms: '{search_terms}'")
                    existing_questions = search_metabase_questions(search_terms, limit=3)
                    
                    if existing_questions and existing_questions != "❌ Metabase API not configured":
                        # Return ONLY the search results, formatted properly
                        metabase_response = f"Here are the *top 3 most recent* Metabase questions for '{search_terms}':\n\n{existing_questions}"
                        
                        slack_message = format_slack_response(
                            category_emoji="🔍",
                            category_title="Existing Metabase Questions",
                            main_content=metabase_response,
                            agent_role="BI Engineer", 
                            agent_emoji=agent_emoji
                        )
                        return slack_message
                    else:
                        # No existing questions found
                        no_results_response = f"No existing Metabase questions found for '{search_terms}'. You can ask me to create a new analysis instead!"
                        
                        slack_message = format_slack_response(
                            category_emoji="🔍",
                            category_title="Existing Metabase Questions",
                            main_content=no_results_response,
                            agent_role="BI Engineer", 
                            agent_emoji=agent_emoji
                        )
                        return slack_message
                        
                except Exception as e:
                    print(f"[DEBUG] Failed to search existing questions: {e}")
                    # Return error message instead of SQL
                    error_response = "Sorry, I couldn't search for existing Metabase questions due to a technical issue. Please try again or ask me to create a new analysis."
                    
                    slack_message = format_slack_response(
                        category_emoji="❌",
                        category_title="Search Error",
                        main_content=error_response,
                        agent_role="BI Engineer", 
                        agent_emoji=agent_emoji
                    )
                    return slack_message
            
            # For regular SQL requests, don't show existing questions
            
            # Format the final message with category header and apply Metabase link processing
            from theo.tools.slack import format_slack_response
            
            slack_message = format_slack_response(
                category_emoji=category_emoji,
                category_title=category_title,
                main_content=main_content if isinstance(main_content, str) else str(main_content),
                agent_role="BI Engineer", 
                agent_emoji=agent_emoji
            )
            
            # Apply Metabase link processing
            def add_metabase_links_to_sql_blocks(text):
                import re
                import urllib.parse
                
                # Step 1: Remove any 'sql' language hints from code blocks 
                text = re.sub(r'```sql\n', '```\n', text)
                
                # Step 2: Find and process each code block individually
                sql_keywords = re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\b', re.IGNORECASE)
                
                # Updated regex to handle code blocks with or without language specifiers
                code_block_pattern = re.compile(r'```(\w+)?\n?(.*?)\n?```', re.DOTALL)
                
                def generate_title_from_sql(sql: str) -> str:
                    """Generate a descriptive title from SQL query."""
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
                
                def replacer(match):
                    language = match.group(1) if match.group(1) else ""  # Optional language specifier
                    code = match.group(2).strip()
                    
                    # Check if this code block contains SQL
                    if sql_keywords.search(code):
                        # Generate title from SQL
                        title = generate_title_from_sql(code)
                        
                        # For very long SQL queries (>500 chars), just provide a generic Metabase link
                        if len(code) > 500:
                            metabase_base_url = os.getenv("METABASE_BASE_URL", "https://sunroom-rentals.metabaseapp.com")
                            metabase_link = f"{metabase_base_url}/question/new"
                            return f'```\n{code}\n```\n<{metabase_link}|▶️ Create in Metabase>'
                        else:
                            # For shorter queries, use the dynamic link
                            encoded_sql = urllib.parse.quote(code)
                            encoded_title = urllib.parse.quote(title) if title else ""
                            
                            # Use environment variable for the base URL, default to localhost for dev
                            base_url = os.getenv("THEO_BASE_URL", "http://localhost:8000")
                            metabase_link = f"{base_url}/create-metabase-question?sql={encoded_sql}&title={encoded_title}"
                            return f'```\n{code}\n```\n<{metabase_link}|▶️ Run in Metabase>'
                    else:
                        # Not SQL, return unchanged
                        return match.group(0)
                
                return code_block_pattern.sub(replacer, text)
            
            # Apply Metabase link processing to the slack message
            slack_message = add_metabase_links_to_sql_blocks(slack_message)
            
            # Convert markdown formatting to Slack formatting
            slack_message = convert_markdown_to_slack(slack_message)
            
            print(f"[DEBUG] bi_engineer_answer Slack message: {slack_message}")
            return slack_message
        return _llm_call()

    def product_manager_answer(self, question, conversation_id=None, context_summary=None, thread_history=None):
        print(f"[DEBUG] product_manager_answer called with question={question}, conversation_id={conversation_id}, context_summary={context_summary}, thread_history={thread_history}")
        agent_role = self.agents_config['product_manager']['role']
        agent_emoji = "🏢"
        category_emoji = "🎫"
        category_title = "Ticket Creation"
        model = self.product_manager_model
        model_provider = self.get_model_provider("AGENT_PRODUCT_MANAGER_MODEL", "AGENT_PRODUCT_MANAGER_PROVIDER")
        from ddtrace.llmobs import LLMObs
        import litellm
        def _llm_call():
            timeline = []
            if thread_history:
                for msg in thread_history:
                    # Get display name with better logic
                    display_name = (
                        msg.get("user_profile", {}).get("display_name") or
                        msg.get("user_profile", {}).get("real_name") or 
                        msg.get("username") or
                        msg.get("user") or
                        ("bot" if msg.get("bot_id") else "unknown")
                    )
                    
                    # Handle Slack user ID format
                    if display_name and display_name.startswith("U") and len(display_name) == 11:
                        better_name = (
                            msg.get("user_profile", {}).get("display_name") or
                            msg.get("user_profile", {}).get("real_name") or
                            msg.get("username")
                        )
                        if better_name:
                            display_name = better_name
                        else:
                            display_name = f"User_{display_name[-4:]}"
                    
                    text = msg.get("text", "").strip()
                    ts = msg.get("ts", "")
                    if text:
                        # Filter out agent signatures to prevent LLM from copying them
                        if "_Taken by" not in text:
                            timeline.append(f"[{ts}] {display_name}: {text}")
            timeline_str = "\n".join(timeline) if timeline else "No timeline available."
            llm_prompt = self.product_manager_prompts["main_prompt"] \
                .replace("{question}", question or "") \
                .replace("{context_summary}", context_summary or "") \
                .replace("{timeline}", timeline_str)
            print(f"[DEBUG] product_manager_answer LLM prompt: {llm_prompt}")
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": agent_role},
                    {"role": "user", "content": llm_prompt}
                ]
            )
            main_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
            if not main_content or not isinstance(main_content, str) or main_content.strip().lower() == "none":
                main_content = "I apologize, but I couldn't generate a proper response. Please try rephrasing your request."
            
            # Validate: Product Manager should not create ADRs or documentation outside of ticket creation
            if any(phrase in main_content.lower() for phrase in ['adr created', 'documentation updated', 'confluence page']):
                print(f"[ERROR] Product Manager tried to create docs/ADRs: {main_content[:100]}...")
                main_content = "I can help with ticket creation. For documentation or ADR creation, please ask explicitly for those services."
            
            LLMObs.annotate(
                input_data=[{"role": "system", "content": agent_role}, {"role": "user", "content": llm_prompt}],
                output_data=[{"role": "assistant", "content": main_content}],
                tags={"agent_role": agent_role, "conversation_id": conversation_id or "unknown", "model_provider": model_provider}
            )
            print(f"[DEBUG] product_manager_answer LLM main_content: {main_content}")
            slack_message = format_slack_response(
                category_emoji=category_emoji,
                category_title=category_title,
                main_content=main_content if isinstance(main_content, str) else str(main_content),
                agent_role=agent_role,
                agent_emoji=agent_emoji
            )
            print(f"[DEBUG] product_manager_answer Slack message: {slack_message}")
            return slack_message
        return _llm_call()

    def get_model_provider(self, model_env_var: str, provider_env_var: str = None) -> str:
        """Extract the model provider from a model name"""
        # If a specific provider env var is provided, use that
        if provider_env_var:
            provider = os.getenv(provider_env_var, "")
            if provider:
                return provider
        
        # Otherwise, extract from the model name
        model = os.getenv(model_env_var, "")
        if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3") or model.startswith("chatgpt"):
            return "openai"
        elif model.startswith("claude"):
            return "anthropic"
        elif model.startswith("gemini"):
            return "google"
        elif model.startswith("meta-llama"):
            return "bedrock"
        else:
            return "unknown"
    
    def _extract_search_terms(self, question: str) -> str:
        """Extract key terms from a question for Metabase search."""
        import re
        
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "show", "get", "find", "give", "tell", "how", "what", "when", "where", "why", "who",
            "many", "much", "some", "all", "any", "me", "us", "we", "i", "you", "can", "could",
            "would", "should", "will", "do", "does", "did", "is", "are", "was", "were", "been",
            "have", "has", "had", "this", "that", "these", "those"
        }
        
        # Extract words and common database terms
        words = re.findall(r'\b\w+\b', question.lower())
        meaningful_words = [w for w in words if len(w) > 2 and w not in stop_words]
        
        # Prioritize database/business terms
        priority_terms = []
        business_keywords = {
            "activation", "partner", "utility", "billing", "user", "property", "revenue",
            "count", "total", "sum", "report", "data", "analytics", "weekly", "monthly",
            "daily", "created", "updated", "status", "type", "manager", "platform"
        }
        
        for word in meaningful_words:
            if word in business_keywords:
                priority_terms.append(word)
        
        # Use priority terms first, then other meaningful words
        search_terms = priority_terms + [w for w in meaningful_words if w not in priority_terms]
        
        # Return top 3-4 most relevant terms
        return " ".join(search_terms[:4])

    def _generate_search_query(self, question: str) -> str:
        """Generate an optimized Metabase search query using LLM based on user's question."""
        import litellm
        
        try:
            search_prompt = f"""You are helping find existing Metabase questions related to a user's data request. Generate a concise search query to find the most relevant existing questions.

User's question: "{question}"

Business context: This is for a utility/property management platform with partners, activations, billing, revenue tracking, and mapping systems.

Key data entities: partners, activations, utility, revenue, billing, properties, mappings, completion rates, payouts, conversions, leads, tours, leases

Instructions:
1. Extract the CORE business concept (e.g., "activation", "revenue", "partner")
2. Add relevant modifiers (e.g., "weekly", "completion", "rate")  
3. Use 2-4 words maximum
4. Use terms that would appear in dashboard/report titles
5. Focus on business metrics, not technical SQL terms

Examples:
"show me weekly activation counts" → "weekly activation"
"partner revenue this month" → "partner revenue monthly"
"utility completion rates going down" → "utility completion rate"
"how many partners have churned" → "partner churn"
"mapping accuracy by provider" → "mapping accuracy"
"revenue per partner by state" → "revenue partner state"
"lead conversion funnel analysis" → "lead conversion"

Generate only the search query (2-4 words):"""

            response = litellm.completion(
                model=self.bi_engineer_model,
                messages=[{
                    "role": "user", 
                    "content": search_prompt
                }],
                max_tokens=15,
                temperature=0.1
            )
            
            generated_query = response['choices'][0]['message']['content'].strip()
            
            # Clean and validate the response
            import re
            cleaned_query = re.sub(r'["\']', '', generated_query)
            cleaned_query = re.sub(r'\b(search|query|find|for|the|a|an)\b', '', cleaned_query, flags=re.IGNORECASE)
            cleaned_query = re.sub(r'[^\w\s]', ' ', cleaned_query)  # Remove special chars
            cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
            
            # Ensure we have 2-4 words
            words = cleaned_query.split()
            if len(words) > 4:
                cleaned_query = " ".join(words[:4])
            elif len(words) < 1:
                cleaned_query = self._extract_search_terms_fallback(question)
            
            print(f"[DEBUG] Generated search query: '{cleaned_query}' for question: '{question[:100]}...'")
            return cleaned_query
            
        except Exception as e:
            print(f"[ERROR] Failed to generate search query: {e}")
            # Fallback to simple keyword extraction
            return self._extract_search_terms_fallback(question)
    
    def _extract_search_terms_fallback(self, question: str) -> str:
        """Fallback method for search term extraction if LLM fails."""
        import re
        
        # Extract key business terms
        business_terms = {
            "activation", "partner", "utility", "billing", "user", "property", "revenue",
            "count", "total", "sum", "report", "analytics", "weekly", "monthly", "daily",
            "completion", "mapping", "churn", "payout", "conversion", "rate"
        }
        
        words = re.findall(r'\b\w+\b', question.lower())
        found_terms = [w for w in words if w in business_terms]
        
        return " ".join(found_terms[:3]) if found_terms else "utility partner"

    def _is_asking_for_existing_metabase_questions(self, question: str) -> bool:
        """Detect if the user is specifically asking for existing Metabase questions using LLM."""
        import litellm
        
        try:
            intent_prompt = f"""You are an intent classifier for a BI system. Users can either:

1. **Request existing Metabase questions/dashboards** - they want to see what already exists
2. **Request new SQL analysis** - they want you to create new queries

Analyze this user request and determine their intent:

User request: "{question}"

Examples of requests for EXISTING Metabase questions:
- "show me the metabase for revenue analysis"
- "give me the metabase to see unmappings week over week"
- "what metabase questions exist for partner data?"
- "find existing dashboards about utility completion rates"
- "what reports do we have for activation trends?"

Examples of requests for NEW SQL analysis:
- "analyze partner revenue trends"
- "show me SQL for weekly activation counts"
- "create a report on utility completion rates"
- "generate analysis of unmapping patterns"
- "I need data on partner performance"

Response with only: "EXISTING" or "NEW" """

            response = litellm.completion(
                model=self.bi_engineer_model,
                messages=[{
                    "role": "user", 
                    "content": intent_prompt
                }],
                max_tokens=10,
                temperature=0.1
            )
            
            intent_result = response['choices'][0]['message']['content'].strip().upper()
            is_existing = intent_result == "EXISTING"
            
            print(f"[DEBUG] LLM Intent detection for '{question}': {intent_result} -> existing={is_existing}")
            return is_existing
            
        except Exception as e:
            print(f"[ERROR] LLM intent detection failed: {e}")
            # Fallback to simple keyword detection
            question_lower = question.lower()
            fallback_keywords = ["metabase", "existing", "what questions", "what reports", "show me the", "give me the"]
            result = any(keyword in question_lower for keyword in fallback_keywords)
            print(f"[DEBUG] Fallback keyword detection for '{question}': {result}")
            return result

    def _build_agent_messages(self, agent_role, question, conversation_id=None, context_summary=None, thread_history=None, config_key=None):
        # Implementation of _build_agent_messages method
        pass

    def create_code_commit_from_github_push(self, commits, push_info, parent_folder_id=None):
        """
        Create a Confluence page documenting GitHub commits from a push event.
        """
        try:
            from theo.tools.confluence import create_confluence_page
            import html
            
            print(f"[DEBUG] Creating code commit documentation for {len(commits)} commits")
            
            # Generate title from commit message(s)
            if len(commits) == 1:
                # Single commit: use the commit message as title
                title = commits[0].get("message", "").strip()
                if not title:
                    repo_name = push_info.get("repository", "")
                    after_hash = push_info.get("after", "")[:8]
                    title = f"{repo_name} - {after_hash}"
            else:
                # Multiple commits: use descriptive title with count
                repo_name = push_info.get("repository", "")
                title = f"{len(commits)} commits to {repo_name}"
            
            # Use the technical writer prompt to generate documentation
            agent_role = "Technical Writer"
            model = self.technical_writer_model
            
            # Build commit summaries
            commit_summaries = []
            for i, commit in enumerate(commits, 1):
                commit_hash = commit.get("id", "")[:8]
                commit_message = commit.get("message", "")
                commit_author = commit.get("author", {}).get("name", "Unknown")
                commit_timestamp = commit.get("timestamp", "")
                commit_url = commit.get("url", "")
                diff_url = f"{commit_url}.diff" if commit_url else ""
                
                files_changed = len(commit.get("added", [])) + len(commit.get("modified", [])) + len(commit.get("removed", []))
                
                commit_summary = f"""
**Commit {i}: {commit_hash}**
- Message: {commit_message}
- Author: {commit_author}
- Timestamp: {commit_timestamp}
- Files changed: {files_changed}
- URL: {commit_url}
- Diff: {diff_url}
"""
                commit_summaries.append(commit_summary)
            
            commit_summaries_str = "\n".join(commit_summaries)
            
            # Generate documentation using technical writer prompt
            doc_prompt = self.technical_writer_prompts["code_commit_multiple"] \
                .replace("{push_info}", str(push_info)) \
                .replace("{commit_summaries}", commit_summaries_str)
            
            import litellm
            from ddtrace.llmobs import LLMObs
            
            try:
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a technical writer."},
                        {"role": "user", "content": doc_prompt}
                    ]
                )
                doc_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
                
                if not doc_content:
                    doc_content = f"Code commit documentation for {title}"
                
                LLMObs.annotate(
                    input_data=[{"role": "system", "content": "You are a technical writer."}, {"role": "user", "content": doc_prompt}],
                    output_data=[{"role": "assistant", "content": doc_content}],
                    tags={"agent_role": agent_role, "action": "code_commit_documentation"}
                )
                
            except Exception as e:
                print(f"[ERROR] Technical Writer LLM call failed for code commit: {e}")
                doc_content = f"Code commit documentation for {title}\n\n{commit_summaries_str}"
            
            # Create Confluence page
            space_key = os.getenv("CONFLUENCE_SPACE_KEY_UP", "UP")
            confluence_response = create_confluence_page(
                title, 
                doc_content, 
                space_key=space_key,
                parent_id=parent_folder_id
            )
            
            print(f"[DEBUG] Created code commit page: {confluence_response}")
            return confluence_response
            
        except Exception as e:
            print(f"[ERROR] Failed to create code commit documentation: {e}")
            return {"error": str(e)}

    def update_tbikb_for_model_changes(self, model_changes, push_info, push_date=None):
        """
        Update TBIKB (Technical Business Intelligence Knowledge Base) for model changes.
        """
        try:
            from theo.tools.confluence import create_confluence_page
            import html
            
            print(f"[DEBUG] Updating TBIKB for {len(model_changes)} model changes")
            
            # Generate title
            repo_name = push_info.get("repository", "")
            title = f"DB Schema Changes: {repo_name} - {push_date or 'Recent'}"
            
            # Build model changes summary
            summary_str = ""
            for change in model_changes:
                file_path = change.get("path", "")
                commit_hash = change.get("commit_hash", "")[:8]
                commit_message = change.get("commit_message", "")
                diff_url = change.get("diff_url", "")
                
                summary_str += f"""
**File: {file_path}**
- Commit: {commit_hash}
- Message: {commit_message}
- Diff: {diff_url}

"""
            
            # Generate documentation using technical writer prompt
            doc_prompt = self.technical_writer_prompts["db_schema_changes"] \
                .replace("{commit_info}", str(push_info)) \
                .replace("{summary_str}", summary_str)
            
            import litellm
            from ddtrace.llmobs import LLMObs
            
            try:
                response = litellm.completion(
                    model=self.technical_writer_model,
                    messages=[
                        {"role": "system", "content": "You are a technical writer."},
                        {"role": "user", "content": doc_prompt}
                    ]
                )
                doc_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
                
                if not doc_content:
                    doc_content = f"Database schema changes for {title}\n\n{summary_str}"
                
                LLMObs.annotate(
                    input_data=[{"role": "system", "content": "You are a technical writer."}, {"role": "user", "content": doc_prompt}],
                    output_data=[{"role": "assistant", "content": doc_content}],
                    tags={"agent_role": "Technical Writer", "action": "db_schema_documentation"}
                )
                
            except Exception as e:
                print(f"[ERROR] Technical Writer LLM call failed for DB schema: {e}")
                doc_content = f"Database schema changes for {title}\n\n{summary_str}"
            
            # Create TBIKB page
            space_key = os.getenv("CONFLUENCE_SPACE_KEY_TBIKB", "TBIKB")
            confluence_response = create_confluence_page(
                title, 
                doc_content, 
                space_key=space_key
            )
            
            print(f"[DEBUG] Created TBIKB page: {confluence_response}")
            return confluence_response
            
        except Exception as e:
            print(f"[ERROR] Failed to update TBIKB: {e}")
            return {"error": str(e)}

    def _is_schema_change(self, diff_text):
        """
        Analyze diff text to determine if it contains schema changes.
        """
        schema_indicators = [
            "@Column", "@ManyToOne", "@OneToMany", "@Entity", "@Table",
            "export interface", "export class", "export enum",
            "extends BaseEntity", "extends Entity",
            "database", "schema", "migration", "model"
        ]
        
        return any(indicator in diff_text for indicator in schema_indicators)

    def create_adr_from_conversation(self, question=None, conversation_id=None, context_summary=None, thread_history=None, **kwargs):
        """
        Create an Architecture Decision Record (ADR) from a conversation.
        """
        try:
            from theo.tools.confluence import create_confluence_page, add_row_to_adr_index
            import html
            from datetime import datetime
            
            print(f"[DEBUG] Creating ADR from conversation: {conversation_id}")
            
            # Extract timeline from thread history
            timeline = []
            authors = set()
            if thread_history:
                for msg in thread_history:
                    # Try to get display name, fallback to user ID
                    display_name = (
                        msg.get("user_profile", {}).get("display_name") or
                        msg.get("user_profile", {}).get("real_name") or 
                        msg.get("username") or
                        msg.get("user") or
                        ("bot" if msg.get("bot_id") else "unknown")
                    )
                    
                    # For user ID format (starts with U), try to get a better name
                    if display_name and display_name.startswith("U") and len(display_name) == 11:
                        # This is a Slack user ID, try to get better name from other fields
                        better_name = (
                            msg.get("user_profile", {}).get("display_name") or
                            msg.get("user_profile", {}).get("real_name") or
                            msg.get("username")
                        )
                        if better_name:
                            display_name = better_name
                        else:
                            # Try to get user info from Slack API
                            try:
                                from theo.tools.slack import get_slack_user_info
                                better_name = get_slack_user_info(display_name)
                                if better_name and better_name != display_name:
                                    display_name = better_name
                                else:
                                    display_name = f"User_{display_name[-4:]}"
                            except Exception as e:
                                print(f"[DEBUG] Could not get Slack user info for {display_name}: {e}")
                                display_name = f"User_{display_name[-4:]}"
                    
                    if display_name != "bot" and "bot" not in display_name.lower():
                        authors.add(display_name)
                    text = msg.get("text", "").strip()
                    ts = msg.get("ts", "")
                    if text and "_Taken by" not in text:
                        timeline.append(f"[{ts}] {display_name}: {text}")
            
            timeline_str = "\n".join(timeline) if timeline else "No timeline available."
            authors_info = ", ".join(authors) if authors else "Unknown"
            
            # Generate ADR content using LLM
            adr_prompt = f"""You are a technical writer creating an Architecture Decision Record (ADR).

CRITICAL INSTRUCTION: Generate ONLY the ADR content itself. Do NOT include any conversational responses, introductions, or explanations like "Here's an ADR..." or "Okay, here's an ADR based on...". Start directly with the ADR title.

Based on this conversation, create a comprehensive ADR following this structure:

# ADR: [Decision Title]

## Status
Proposed

## Context
[What is the issue that we're seeing that is motivating this decision or change?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult to do because of this change?]

## Implementation Notes
[Any specific implementation details, timelines, or considerations]

Conversation context:
{context_summary or ""}

Conversation timeline:
{timeline_str}

User question/request: {question or ""}

Generate ONLY the ADR content - start directly with the title heading. Do NOT include any conversational text or explanations."""

            model = self.technical_writer_model
            import litellm
            from ddtrace.llmobs import LLMObs
            
            try:
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a technical writer specializing in Architecture Decision Records."},
                        {"role": "user", "content": adr_prompt}
                    ]
                )
                adr_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
                
                if not adr_content:
                    adr_content = f"# ADR: {question or 'New Decision'}\n\n## Status\nProposed\n\n## Context\n{context_summary or 'Context not available'}"
                
                LLMObs.annotate(
                    input_data=[{"role": "system", "content": "You are a technical writer specializing in Architecture Decision Records."}, {"role": "user", "content": adr_prompt}],
                    output_data=[{"role": "assistant", "content": adr_content}],
                    tags={"agent_role": "Technical Writer", "action": "adr_creation"}
                )
                
            except Exception as e:
                print(f"[ERROR] Technical Writer LLM call failed for ADR: {e}")
                adr_content = f"# ADR: {question or 'New Decision'}\n\n## Status\nProposed\n\n## Context\n{context_summary or 'Context not available'}"
            
            # Extract title from ADR content
            import re
            title_match = re.search(r'# ADR[-\s]*\d*:?\s*(.+)', adr_content)
            if title_match:
                adr_title = title_match.group(1).strip()
            else:
                adr_title = question or 'New ADR'
            
            # Create ADR page in UP space under ADR parent folder
            adr_parent_page_id = os.getenv("CONFLUENCE_ADR_PARENT_PAGE_ID")
            space_key = os.getenv("CONFLUENCE_SPACE_KEY_UP", "UP")
            
            try:
                confluence_response = create_confluence_page(
                    f"ADR: {adr_title}", 
                    adr_content, 
                    space_key=space_key,
                    parent_id=adr_parent_page_id
                )
                
                # Check if Confluence page creation was successful
                if "error" in confluence_response or not confluence_response.get("id"):
                    error_msg = confluence_response.get("error", "Unknown Confluence error")
                    print(f"[ERROR] ADR creation failed: {error_msg}")
                    return {
                        "adr_title": adr_title,
                        "adr_page_url": "",
                        "adr_content": f"❌ **ADR Creation Failed**\n\nError: {error_msg}\n\nPlease check Confluence configuration or contact admin.",
                        "authors_info": authors_info,
                        "error": error_msg
                    }
                    
            except Exception as confluence_error:
                error_msg = str(confluence_error)
                print(f"[ERROR] ADR Confluence creation exception: {error_msg}")
                return {
                    "adr_title": adr_title,
                    "adr_page_url": "",
                    "adr_content": f"❌ **ADR Creation Failed**\n\nConfluence Error: {error_msg}\n\nThe ADR parent page (CONFLUENCE_ADR_PARENT_PAGE_ID={adr_parent_page_id}) may not exist or you may not have permissions. Please contact admin.",
                    "authors_info": authors_info,
                    "error": error_msg
                }
            
            # Build ADR page URL
            page_id = confluence_response.get("id")
            base_url = os.getenv("CONFLUENCE_BASE_URL")
            adr_page_url = f"{base_url}/pages/viewpage.action?pageId={page_id}" if page_id and base_url else ""
            
            # Add to ADR index
            try:
                add_row_to_adr_index(
                    adr_title=adr_title,
                    adr_url=adr_page_url,
                    status="Proposed",
                    authors=authors_info,
                    date=datetime.now().strftime("%Y-%m-%d")
                )
            except Exception as e:
                print(f"[ERROR] Failed to add ADR to index: {e}")
            
            print(f"[DEBUG] Created ADR: {adr_title} at {adr_page_url}")
            
            return {
                "adr_title": adr_title,
                "adr_page_url": adr_page_url,
                "adr_content": adr_content,
                "authors_info": authors_info,
                "confluence_response": confluence_response
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to create ADR: {e}")
            return {
                "adr_title": question or "New ADR",
                "adr_page_url": "",
                "adr_content": f"Error creating ADR: {str(e)}",
                "authors_info": "Unknown",
                "error": str(e)
            }

def convert_markdown_to_slack(text):
    """
    Convert common markdown formatting to Slack formatting.
    """
    import re
    
    # Convert headers (## Header -> *Header*)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    
    # Convert bold (**text** -> *text*)
    text = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', text)
    
    # Convert bullet points (* item -> • item)
    text = re.sub(r'^\s*\*\s+(.+)$', r'• \1', text, flags=re.MULTILINE)
    
    # Convert numbered lists (1. item -> 1. item - keep as is, but ensure proper spacing)
    text = re.sub(r'^\s*(\d+)\.\s+(.+)$', r'\1. \2', text, flags=re.MULTILINE)
    
    # Convert code inline (`code` -> `code` - keep as is)
    # Already handled properly
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def format_description(description_dict):
    """
    Convert a dictionary of description sections into a Markdown string for JIRA.
    """
    if not isinstance(description_dict, dict):
        return str(description_dict)
    lines = []
    for section, content in description_dict.items():
        lines.append(f"### {section}\n{content}\n")
    return "\n".join(lines)

def validate_agent_response(agent_type, response, expected_action):
    """
    Final safety check to ensure agents only perform their assigned tasks.
    """
    if not response or not isinstance(response, str):
        return True  # Let other validation handle this
    
    response_lower = response.lower()
    
    # Define what each agent should NOT do
    forbidden_actions = {
        "support_engineer": {
            "patterns": ["created ticket", "jira ticket", "confluence page", "documentation created"],
            "message": "Support Engineer attempted to create tickets or documentation"
        },
        "product_manager": {
            "patterns": ["adr created", "documentation updated", "confluence page"],
            "message": "Product Manager attempted to create ADRs or documentation"
        },
        "technical_writer": {
            "patterns": ["jira ticket", "created ticket"] if expected_action not in ["documentation_and_adr"] else [],
            "message": "Technical Writer attempted to create tickets outside of expected scope"
        }
    }
    
    if agent_type in forbidden_actions:
        patterns = forbidden_actions[agent_type]["patterns"]
        if any(pattern in response_lower for pattern in patterns):
            print(f"[ERROR] {forbidden_actions[agent_type]['message']}: {response[:100]}...")
            return False
    
    return True
