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
            user = msg.get("user") or msg.get("username") or ("Bot" if msg.get("bot_id") else "User")
            timeline.append(f"[{ts_str}] {user}: {text.strip()}")
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
        # For bi_report and documentation_update, pass both summary and raw thread
        if answer_clean == "bi_report" or answer_clean == "documentation_update":
            print(f"[DEBUG] Passing to downstream agent: answer={answer_clean}, context_summary={context_summary}, thread_history={thread_history}")
            return {
                "task_name": answer_clean,
                "downstream_prompt": downstream_prompt,
                "context_summary": context_summary,
                "thread_history": thread_history
            }
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
                user = msg.get("user") or msg.get("username") or ("bot" if msg.get("bot_id") else "unknown")
                text = msg.get("text", "").strip()
                ts = msg.get("ts", "")
                if text:
                    timeline.append(f"[{ts}] {user}: {text}")
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
            LLMObs.annotate(
                input_data=[{"role": "system", "content": "You are a technical writer."}, {"role": "user", "content": doc_prompt}],
                output_data=[{"role": "assistant", "content": doc_content}],
                tags={"agent_role": agent_role, "action": "generate_doc_content", "model_provider": model_provider}
            )
        except Exception as e:
            print(f"[ERROR] Technical Writer LLM call failed: {e}")
            doc_content = "Documentation could not be generated."
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
            print(f"[DEBUG] Creating/Updating main doc in UP space.")
            confluence_response = create_confluence_page(page_title, doc_content, space_key=os.getenv("CONFLUENCE_SPACE_KEY_UP", "UP"))
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
            print(f"[DEBUG] Creating/Updating General Knowledge in TKB space.")
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
        agent_role = self.agents_config['support_engineer']['role']
        agent_emoji = "👨‍💻"
        category_emoji = "📄"
        category_title = "Support Request"
        model = self.support_engineer_model
        model_provider = get_model_provider("AGENT_SUPPORT_ENGINEER_MODEL", "AGENT_SUPPORT_ENGINEER_PROVIDER")
        context_summary_clean = "" if context_summary in (None, "None") else context_summary
        docs = bedrock.search_code_documentation(question)
        print(f"[DEBUG] Bedrock docs: {docs}")
        # Get current date/time in Central Time
        tz = pytz.timezone("America/Chicago")
        now = datetime.now(tz)
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M %Z")
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
            LLMObs.annotate(
                input_data=[{"role": "system", "content": agent_role}, {"role": "user", "content": llm_prompt}],
                output_data=[{"role": "assistant", "content": main_content}],
                tags={"agent_role": agent_role, "model_provider": model_provider}
            )
            print(f"[DEBUG] LLM raw output: {main_content}")
            main_content = markdown_to_slack(main_content)
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
            return answer.startswith("yes")
        except Exception:
            return False

    def supervisor_answer(self, question, conversation_id=None, context_summary=None):
        from theo.tools.slack import format_slack_response
        agent_role = self.agents_config['supervisor']['role']
        agent_emoji = ":shield:"
        category_emoji = ":satellite_antenna:"
        category_title = "Platform Health"
        model = self.supervisor_model
        model_provider = get_model_provider("AGENT_SUPERVISOR_MODEL", "AGENT_SUPERVISOR_PROVIDER")
        context_summary = context_summary or ""
        valid_tasks = {"support_request", "documentation_update", "bi_report", "ticket_creation", "platform_health", "supervisor_health", "clarification_needed"}

        # Heartbeat/health check interception
        if self._is_heartbeat_question(question):
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
        # Format routing_actions for prompt injection
        routing_actions_dict = self.supervisor_prompts.get("routing_actions", {})
        routing_actions_str = "\n".join([f"- {k}: {v}" for k, v in routing_actions_dict.items()])
        prompt = self.supervisor_prompts["routing_prompt"] \
            .replace('{user_message}', question or '') \
            .replace('{context_summary}', context_summary or '') \
            .replace('{routing_actions}', routing_actions_str)
        print(f"[DEBUG] Supervisor LLM full prompt:\n{prompt}")
        import litellm
        from ddtrace.llmobs import LLMObs
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an agent router."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response['choices'][0]['message']['content'] if response and 'choices' in response else None
            if answer:
                answer = answer.strip().lower()
            else:
                answer = "clarification_needed"
            LLMObs.annotate(
                input_data=[{"role": "system", "content": "You are an agent router."}, {"role": "user", "content": prompt}],
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
            ticket_keywords = ["ticket", "jira", "sprint", "board", "issue", "story", "bug", "hotfix", "task"]
            if any(kw in lowered_context for kw in ticket_keywords) or any(kw in (question or '').lower() for kw in ticket_keywords):
                print("[DEBUG] Heuristic fallback: context indicates ticket/board/sprint, overriding to ticket_creation")
                answer = "ticket_creation"
            elif ("support request" in lowered_context or "troubleshoot" in lowered_context or "error" in lowered_context or "help" in lowered_context):
                print("[DEBUG] Heuristic fallback: context indicates support, overriding to support_request")
                answer = "support_request"
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
                    user = msg.get("user") or msg.get("username") or ("bot" if msg.get("bot_id") else "unknown")
                    text = msg.get("text", "").strip()
                    ts = msg.get("ts", "")
                    if text:
                        timeline.append(f"[{ts}] {user}: {text}")
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
            main_content = LLMObs.annotate(
                input_data=[{"role": "user", "content": llm_prompt}],
                output_data=[{"role": "assistant", "content": ""}],
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
        model_provider = get_model_provider("AGENT_BI_ENGINEER_MODEL", "AGENT_BI_ENGINEER_PROVIDER")
        bedrock = BedrockClient()
        db_docs = bedrock.search_db_schema(question)
        from ddtrace.llmobs import LLMObs
        import litellm
        import re
        def extract_table_names(text):
            table_pattern = re.compile(r'"([a-zA-Z0-9_]+)"|\b([a-zA-Z][a-zA-Z0-9_]+)\b')
            matches = table_pattern.findall(text)
            names = [m[0] or m[1] for m in matches if m[0] or m[1]]
            keywords = {"select", "from", "where", "join", "on", "group", "by", "order", "limit", "as", "and", "or", "not", "in", "is", "null", "count", "sum", "avg", "min", "max", "left", "right", "inner", "outer", "having", "distinct", "case", "when", "then", "else", "end"}
            return [n for n in set(names) if n.lower() not in keywords]
        def _llm_call():
            timeline = []
            if thread_history:
                for msg in thread_history:
                    user = msg.get("user") or msg.get("username") or ("bot" if msg.get("bot_id") else "unknown")
                    text = msg.get("text", "").strip()
                    ts = msg.get("ts", "")
                    if text:
                        timeline.append(f"[{ts}] {user}: {text}")
            timeline_str = "\n".join(timeline) if timeline else "No timeline available."
            table_names = extract_table_names(question)
            print(f"[DEBUG] Extracted table names: {table_names}")
            schemas = []
            for table in table_names:
                schema = bedrock.search_db_schema(f"{table} schema")
                if schema and "no results" not in schema.lower() and "error" not in schema.lower():
                    schemas.append((table, schema))
            base_prompt = self.bi_engineer_prompts["main_prompt"]
            if schemas:
                schema_text = "\n\n".join([f"Schema for {t}:\n{s}" for t, s in schemas])
                focused_prompt = base_prompt \
                    .replace("{question}", question or "") \
                    .replace("{context_summary}", context_summary or "") \
                    .replace("{db_docs}", schema_text) \
                    .replace("{timeline}", timeline_str)
                focused_prompt += "\n" + self.bi_engineer_prompts["sql_generation"]
                focused_prompt += "\n\n" + self.bi_engineer_prompts["business_analysis_template"]
                print(f"[DEBUG] Focused LLM prompt: {focused_prompt}")
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "system", "content": agent_role}, {"role": "user", "content": focused_prompt}]
                )
                main_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
                if not main_content or not isinstance(main_content, str) or main_content.strip().lower() == "none":
                    main_content = self.bi_engineer_prompts["no_sql_generated"]
                LLMObs.annotate(
                    input_data=[{"role": "system", "content": agent_role}, {"role": "user", "content": focused_prompt}],
                    output_data=[{"role": "assistant", "content": main_content}],
                    tags={"agent_role": agent_role, "conversation_id": conversation_id or "unknown", "model_provider": model_provider}
                )
            else:
                no_results = (
                    not db_docs or
                    "no results" in db_docs.lower() or
                    "error" in db_docs.lower() or
                    "not set" in db_docs.lower() or
                    db_docs.strip() == ""
                )
                if no_results:
                    schema_info = bedrock.search_db_schema("utilityPartners schema")
                    fallback_prompt = base_prompt \
                        .replace("{question}", question or "") \
                        .replace("{context_summary}", context_summary or "") \
                        .replace("{db_docs}", schema_info or "") \
                        .replace("{timeline}", timeline_str)
                    fallback_prompt += "\n\n" + self.bi_engineer_prompts["business_analysis_template"]
                    print(f"[DEBUG] Fallback LLM prompt: {fallback_prompt}")
                    response = litellm.completion(
                        model=model,
                        messages=[{"role": "system", "content": agent_role}, {"role": "user", "content": fallback_prompt}]
                    )
                    main_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
                    if not main_content or not isinstance(main_content, str) or main_content.strip().lower() == "none":
                        main_content = self.bi_engineer_prompts["no_schema_found"]
                    LLMObs.annotate(
                        input_data=[{"role": "system", "content": agent_role}, {"role": "user", "content": fallback_prompt}],
                        output_data=[{"role": "assistant", "content": main_content}],
                        tags={"agent_role": agent_role, "conversation_id": conversation_id or "unknown", "model_provider": model_provider}
                    )
                else:
                    llm_prompt = base_prompt \
                        .replace("{question}", question or "") \
                        .replace("{context_summary}", context_summary or "") \
                        .replace("{db_docs}", db_docs or "") \
                        .replace("{timeline}", timeline_str)
                    llm_prompt += "\n\n" + self.bi_engineer_prompts["business_analysis_template"]
                    print(f"[DEBUG] bi_engineer_answer LLM prompt: {llm_prompt}")
                    main_content = LLMObs.annotate(
                        input_data=[{"role": "user", "content": llm_prompt}],
                        output_data=[{"role": "assistant", "content": ""}],
                        tags={"agent_role": agent_role, "conversation_id": conversation_id or "unknown", "model_provider": model_provider}
                    )
                    print(f"[DEBUG] bi_engineer_answer LLM main_content: {main_content}")
                    if not main_content or not isinstance(main_content, str) or main_content.strip().lower() == "none":
                        main_content = self.bi_engineer_prompts["no_schema_found"]
            slack_message = format_slack_response(
                category_emoji=category_emoji,
                category_title=category_title,
                main_content=(main_content.replace('```sql', '```') if isinstance(main_content, str) else str(main_content).replace('```sql', '```')),
                agent_role=agent_role,
                agent_emoji=agent_emoji
            )
            print(f"[DEBUG] bi_engineer_answer Slack message: {slack_message}")
            return slack_message
        return _llm_call()

    def product_manager_answer(self, question, conversation_id=None, thread_history=None, channel=None, thread_ts=None):
        import litellm
        from theo.tools.jira import create_jira_ticket
        from theo.tools.slack import send_slack_message
        import logging
        import re
        import json

        agent_role = self.agents_config['product_manager']['role']
        agent_emoji = "📝"
        category_emoji = "📝"
        category_title = "Ticket Creation"
        model = self.product_manager_model
        model_provider = get_model_provider("AGENT_PRODUCT_MANAGER_MODEL", "AGENT_PRODUCT_MANAGER_PROVIDER")

        def get_ticket_type_and_summary(question):
            # Build conversation timeline with timestamps for context
            timeline = []
            if thread_history:
                for msg in thread_history:
                    user = msg.get("user") or msg.get("username") or ("bot" if msg.get("bot_id") else "unknown")
                    text = msg.get("text", "").strip()
                    ts = msg.get("ts", "")
                    if text:
                        timeline.append(f"[{ts}] {user}: {text}")
            timeline_str = "\n".join(timeline)
            
            logging.info(f"Timeline sent to LLM for ticket creation:\n{timeline_str}")
            
            base_prompt = self.product_manager_prompts.get('ticket_creation', '')
            
            # First pass: Get ticket type and initial structure
            messages = [
                {"role": "system", "content": "You are a Product Manager. You MUST respond with ONLY valid JSON. No explanations, no markdown, just pure JSON."},
                {"role": "user", "content": f"{base_prompt}\n\nConversation History:\n{timeline_str}\n\nCurrent Request: {question}"}
            ]
            
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                result = getattr(response.choices[0].message, 'content', None)
                logging.info(f"LLM Response for ticket creation:\n{result}")
                if not result or not isinstance(result, str) or not result.strip():
                    logging.error(f"LLM returned no content for ticket creation. Raw response: {response}")
                    return {
                        "error": "The LLM did not return a ticket. Please try again or rephrase your request."
                    }
                
                # Clean the result - remove any markdown formatting or extra text
                cleaned_result = result.strip()
                
                # Try to extract JSON if it's wrapped in markdown code blocks
                if cleaned_result.startswith('```'):
                    # Remove markdown code block formatting
                    lines = cleaned_result.split('\n')
                    if lines[0].startswith('```'):
                        lines = lines[1:]  # Remove first ```
                    if lines[-1].strip() == '```':
                        lines = lines[:-1]  # Remove last ```
                    cleaned_result = '\n'.join(lines).strip()
                
                logging.info(f"Cleaned LLM response for JSON parsing:\n{cleaned_result}")
                
                # Parse the JSON response
                try:
                    ticket_data = json.loads(cleaned_result)
                    
                    # Validate required fields
                    if not isinstance(ticket_data, dict):
                        raise ValueError("Response is not a JSON object")
                    
                    required_fields = ["type", "summary", "description"]
                    for field in required_fields:
                        if field not in ticket_data:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Check if we have clarifying questions
                    if "clarifying_questions" in ticket_data and ticket_data["clarifying_questions"]:
                        # Before asking questions, check if answers exist in conversation history
                        questions_to_ask = []
                        for question in ticket_data["clarifying_questions"]:
                            # Simple check if the answer might exist in the conversation
                            # This could be enhanced with more sophisticated matching
                            question_topic = re.sub(r'[^a-zA-Z0-9\s]', '', question.lower())
                            found_in_history = any(
                                any(word in msg.lower() for word in question_topic.split())
                                for msg in timeline_str.split("\n")
                            )
                            if not found_in_history:
                                questions_to_ask.append(question)
                        if questions_to_ask:
                            return {
                                "needs_clarification": True,
                                "questions": questions_to_ask
                            }
                    return ticket_data
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse LLM response as JSON: {e}")
                    logging.error(f"Raw LLM response was: {repr(result)}")
                    logging.error(f"Cleaned response was: {repr(cleaned_result)}")
                    
                    # Fallback: try to create a basic ticket from the text
                    fallback_ticket = {
                        "type": "Task",
                        "summary": question[:80] if question else "Manual ticket creation needed",
                        "description": {
                            "Context": f"LLM returned invalid JSON. Original request: {question}",
                            "Goal": "Please review the original request and create ticket manually",
                            "LLM Response": cleaned_result[:500] + "..." if len(cleaned_result) > 500 else cleaned_result
                        }
                    }
                    return fallback_ticket
                except ValueError as e:
                    logging.error(f"Invalid ticket data structure: {e}")
                    logging.error(f"Ticket data: {ticket_data}")
                    return {
                        "error": f"Invalid response format from LLM: {str(e)}"
                    }
            except Exception as e:
                logging.error(f"Error in LLM call: {e}")
                return {
                    "error": f"Failed to process ticket creation: {str(e)}"
                }

        # Get filtered thread history
        filtered_thread_history = []
        if thread_history:
            filtered_thread_history = [
                msg for msg in thread_history 
                if not any(keyword in msg.get("text", "").lower() 
                          for keyword in ["typing", "has joined"])
            ]

        # Process the ticket creation
        ticket_data = get_ticket_type_and_summary(question)
        
        if "error" in ticket_data:
            return f"❌ Error creating ticket: {ticket_data['error']}"
            
        if ticket_data.get("needs_clarification"):
            questions = ticket_data["questions"]
            response = "I need a bit more information to create a complete ticket:\n\n"
            for i, question in enumerate(questions, 1):
                response += f"{i}. {question}\n"
            return response
            
        try:
            # Prepare the description and append Slack link if available
            description = format_description(ticket_data["description"])
            if channel and thread_ts:
                slack_permalink = self.build_slack_permalink(channel, thread_ts)
                description += f"\n\nSlack conversation: {slack_permalink}"
            # Create the JIRA ticket
            jira_response = create_jira_ticket(
                summary=ticket_data["summary"],
                description=description,
                issue_type=ticket_data["type"]
            )
            
            if jira_response.get("error"):
                return f"❌ Failed to create JIRA ticket: {jira_response['error']}"
                
            ticket_url = jira_response.get("url", "")
            ticket_key = jira_response.get("key", "")
            
            return f"✅ Created {ticket_data['type']} ticket: {ticket_key}\n{ticket_url}"
            
        except Exception as e:
            logging.error(f"Error creating JIRA ticket: {e}")
            return f"❌ Failed to create JIRA ticket: {str(e)}"

    def build_slack_permalink(self, channel, thread_ts):
        workspace_domain = os.getenv("SLACK_WORKSPACE_DOMAIN", "yourdomain.slack.com")
        ts = thread_ts.replace('.', '')
        return f"https://{workspace_domain}/archives/{channel}/p{ts}"

    def get_model_provider(self, model_env_var, provider_env_var):
        provider = os.getenv(provider_env_var)
        model = os.getenv(model_env_var, "").lower()
        if provider:
            return provider
        if "gpt" in model:
            return "openai"
        if "gemini" in model:
            return "google"
        if "llama" in model or "meta" in model:
            return "meta"
        if "grok" in model:
            return "xai"
        if "claude" in model or "sonnet" in model:
            return "anthropic"
        # Add more providers as needed
        return "openai"  # Fallback to OpenAI

    def _markdown_table_to_bullet_list(self, md_text):
        """Convert a Markdown table to a bullet list. Only processes the first table found."""
        import re
        lines = md_text.splitlines()
        in_table = False
        headers = []
        bullets = []
        for i, line in enumerate(lines):
            if re.match(r'^\s*\|', line):
                if not in_table:
                    in_table = True
                    headers = [h.strip() for h in line.strip('|').split('|')]
                    continue
                # Skip separator row
                if re.match(r'^\s*\|\s*-', line):
                    continue
                # Table row
                cells = [c.strip() for c in line.strip('|').split('|')]
                if len(cells) == len(headers):
                    bullet = ', '.join(f"{headers[j]}: {cells[j]}" for j in range(len(headers)) if cells[j])
                    bullets.append(f"- {bullet}")
            elif in_table:
                # End of table
                break
        if bullets:
            # Remove the table from the text and insert the bullet list
            table_start = next(i for i, l in enumerate(lines) if re.match(r'^\s*\|', l))
            table_end = table_start + len(bullets) + 2  # +2 for header and separator
            new_lines = lines[:table_start] + bullets + lines[table_end:]
            return '\n'.join(new_lines)
        return md_text

    def create_adr_from_github(self, commit_title, commit_body, commit_url, author, date, diff_url=None, parent_adr_page_id=None):
        """
        Create a new ADR subpage and update the ADR Index table in Confluence based on a GitHub merge to main.
        """
        import litellm
        from theo.tools.confluence import create_confluence_page, add_row_to_adr_index
        from datetime import datetime
        # 1. Generate ADR content using LLM
        model = os.getenv("AGENT_TECHNICAL_WRITER_MODEL", "unknown")
        adr_prompt = (
            "IMPORTANT: Do NOT use tables. Use only a bullet list for the summary. If you use a table, your answer will be discarded.\n"
            "For the summary section, use this format (not a table):\n"
            "- Decision: Prevent duplicate activations. Consequence: Improved data integrity and sync process.\n"
            "- Decision: Remove special characters. Consequence: Cleaner data with uniform sender names.\n"
            "- ...\n"
            "\nYou are a technical writer. Draft an Architecture Decision Record (ADR) for the following code change. "
            "Summarize the context, decision, alternatives, and consequences. Use a clear, professional tone. "
            "For the summary section, output a bullet list of decisions and consequences (not a table).\n\n"
            f"Commit title: {commit_title}\n"
            f"Commit body: {commit_body}\n"
            f"Commit URL: {commit_url}\n"
            f"Author: {author}\n"
            f"Date: {date}\n"
            f"Diff URL: {diff_url or 'N/A'}\n"
        )
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a technical writer."},
                {"role": "user", "content": adr_prompt}
            ]
        )
        adr_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
        if not adr_content or adr_content == "None":
            adr_content = f"ADR for commit: {commit_title}\n\n{commit_body}"
        # Post-process: Convert any Markdown table to a bullet list
        adr_content = self._markdown_table_to_bullet_list(adr_content)
        # 2. Generate a concise ADR page title
        title_prompt = (
            "Given the following ADR content, generate a concise, descriptive Confluence page title.\n\n"
            f"Content:\n{adr_content}\n"
        )
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
        if not page_title or page_title.strip().lower() == "none":
            page_title = commit_title or 'New ADR'
        # 3. Create the ADR subpage (in UP space, parented under ADR parent page)
        space_key = os.getenv("CONFLUENCE_SPACE_KEY_UP", "UP")
        parent_id = parent_adr_page_id or os.getenv("CONFLUENCE_ADR_PARENT_PAGE_ID", "2117566465")
        base_url = os.getenv("CONFLUENCE_BASE_URL")
        # Confluence API: create child page by specifying ancestors
        from theo.tools.confluence import markdown_to_confluence_storage
        content_html = markdown_to_confluence_storage(adr_content)
        user = os.getenv("CONFLUENCE_ADMIN_USER")
        api_token = os.getenv("CONFLUENCE_API_TOKEN")
        url = f"{base_url}/rest/api/content/"
        data = {
            "type": "page",
            "title": page_title,
            "ancestors": [{"id": str(parent_id)}],
            "space": {"key": space_key},
            "body": {
                "storage": {
                    "value": content_html,
                    "representation": "storage"
                }
            }
        }
        resp = requests.post(url, json=data, auth=(user, api_token))
        resp.raise_for_status()
        new_page = resp.json()
        new_page_id = new_page.get("id")
        new_page_url = f"{base_url}/pages/viewpage.action?pageId={new_page_id}" if new_page_id else ""
        # 4. Add a new row to the ADR Index table
        # Columns: Title (link), Status, Date, Authors, Source
        date_str = date if isinstance(date, str) else datetime.utcnow().strftime("%Y-%m-%d")
        title_cell = f'<a href="{new_page_url}" title="{new_page_url}">{page_title}</a>'
        status_cell = '✅ Merged'
        authors_cell = 'LLM'
        source_cell = 'GitHub'
        new_row = [title_cell, status_cell, date_str, authors_cell, source_cell]
        add_row_to_adr_index(parent_id, new_row)
        return {"adr_page_id": new_page_id, "adr_page_url": new_page_url, "adr_title": page_title}

    def update_tbikb_for_model_changes(self, changed_models, commit_info, push_date=None):
        """
        Summarize all .model.ts changes in a push and create/update a single TBIKB Confluence page.
        changed_models: list of dicts with keys: path, diff_url, commit_message, etc.
        commit_info: dict with push metadata (e.g., pusher, commit hashes, etc.)
        push_date: optional, ISO string or datetime
        """
        import litellm
        from theo.tools.confluence import create_confluence_page
        from datetime import datetime
        model = os.getenv("AGENT_TECHNICAL_WRITER_MODEL", "unknown")
        # Build a summary prompt for all model changes
        changes_summary = []
        for model_file in changed_models:
            entry = (
                f"File: {model_file.get('path')}\n"
                f"Commit: {model_file.get('commit_hash', 'N/A')}\n"
                f"Commit message: {model_file.get('commit_message', '')}\n"
                f"Diff URL: {model_file.get('diff_url', 'N/A')}\n"
            )
            changes_summary.append(entry)
        summary_str = "\n---\n".join(changes_summary)
        prompt = (
            "IMPORTANT: Do NOT use tables. Use only a bullet list for the summary. If you use a table, your answer will be discarded.\n"
            "For the summary section, use this format (not a table):\n"
            "- Change: Prevent duplicate activations. Consequence: Improved data integrity.\n"
            "- Change: Remove special characters. Consequence: Cleaner data.\n"
            "- ...\n"
            "\nYou are a technical writer. Summarize the following DB schema/entity changes based on the .model.ts files changed in this code push. "
            "For each file, describe the entity, the nature of the change (added/removed/modified fields, etc.), and any impact on the database. "
            "For the summary section, output a bullet list of changes and their consequences (not a table).\n\n"
            f"Push info: {commit_info}\n\n"
            f"Model changes:\n{summary_str}"
        )
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a technical writer."},
                {"role": "user", "content": prompt}
            ]
        )
        doc_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
        if not doc_content or doc_content == "None":
            doc_content = f"DB schema/entity changes for this push:\n\n{summary_str}"
        # Post-process: Convert any Markdown table to a bullet list
        doc_content = self._markdown_table_to_bullet_list(doc_content)
        # Title: DB Schema Changes - <date or commit hash>
        date_str = push_date or datetime.utcnow().strftime("%Y-%m-%d")
        title = f"DB Schema Changes - {date_str}"
        # Create/update the TBIKB page
        confluence_response = create_confluence_page(title, doc_content, space_key=os.getenv("CONFLUENCE_SPACE_KEY_TBIKB", "TBIKB"))
        page_id = confluence_response.get("id")
        base_url = os.getenv("CONFLUENCE_BASE_URL")
        confluence_url = f"{base_url}/pages/viewpage.action?pageId={page_id}" if page_id and base_url else ""
        return {"tbikb_page_id": page_id, "tbikb_page_url": confluence_url, "tbikb_title": title}

    def create_adr_from_github_push(self, commits, push_info, parent_adr_page_id=None):
        """
        Create a single ADR subpage and update the ADR Index table in Confluence for a GitHub push event (all commits summarized).
        """
        import litellm
        from theo.tools.confluence import create_confluence_page, add_row_to_adr_index, markdown_to_confluence_storage
        from datetime import datetime
        import os, re
        model = os.getenv("AGENT_TECHNICAL_WRITER_MODEL", "unknown")
        # 1. Aggregate commit info
        commit_summaries = []
        for commit in commits:
            commit_message = commit.get('message', '')
            commit_title = commit_message.split('\n')[0]
            commit_summaries.append(
                f"Commit: {commit.get('id', '')}\nTitle: {commit_title}\nBody: {commit_message}\nURL: {commit.get('url', '')}\nAuthor: {commit.get('author', {}).get('name', 'Unknown')}\nDate: {commit.get('timestamp', '')}\n"
            )
        summary_str = "\n---\n".join(commit_summaries)
        adr_prompt = (
            "You are a technical writer. Draft a single Architecture Decision Record (ADR) summarizing the following code push. "
            "Summarize the context, decisions, alternatives, and consequences for all included commits. Use a clear, professional tone. "
            "Include a summary table if appropriate.\n\n"
            f"Push info: {push_info}\n\nCommits:\n{summary_str}"
        )
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a technical writer."},
                {"role": "user", "content": adr_prompt}
            ]
        )
        adr_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
        if not adr_content or adr_content == "None":
            adr_content = f"ADR for code push:\n\n{summary_str}"
        # 2. Generate a concise ADR page title
        title_prompt = (
            "Given the following ADR content, generate a concise, descriptive Confluence page title.\n\n"
            f"Content:\n{adr_content}\n"
        )
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
        if not page_title or page_title.strip().lower() == "none":
            page_title = "ADR for code push"
        # 3. Create the ADR subpage (in UP space, parented under ADR parent page)
        space_key = os.getenv("CONFLUENCE_SPACE_KEY_UP", "UP")
        parent_id = parent_adr_page_id or os.getenv("CONFLUENCE_ADR_PARENT_PAGE_ID", "2117566465")
        base_url = os.getenv("CONFLUENCE_BASE_URL")
        content_html = markdown_to_confluence_storage(adr_content)
        user = os.getenv("CONFLUENCE_ADMIN_USER")
        api_token = os.getenv("CONFLUENCE_API_TOKEN")
        url = f"{base_url}/rest/api/content/"
        data = {
            "type": "page",
            "title": page_title,
            "ancestors": [{"id": str(parent_id)}],
            "space": {"key": space_key},
            "body": {
                "storage": {
                    "value": content_html,
                    "representation": "storage"
                }
            }
        }
        resp = requests.post(url, json=data, auth=(user, api_token))
        resp.raise_for_status()
        new_page = resp.json()
        new_page_id = new_page.get("id")
        new_page_url = f"{base_url}/pages/viewpage.action?pageId={new_page_id}" if new_page_id else ""
        # 4. Add a new row to the ADR Index table
        date_str = datetime.utcnow().isoformat()
        title_cell = f'<a href="{new_page_url}" title="{new_page_url}">{page_title}</a>'
        status_cell = '✅ Merged'
        authors_cell = 'LLM'
        source_cell = 'GitHub'
        new_row = [title_cell, status_cell, date_str, authors_cell, source_cell]
        add_row_to_adr_index(parent_id, new_row)
        return {"adr_page_id": new_page_id, "adr_page_url": new_page_url, "adr_title": page_title}

    def _is_schema_change(self, diff_text):
        """
        Heuristic: Returns True if the diff contains likely DB schema changes (property or decorator add/remove/change).
        """
        import re
        schema_patterns = [
            r'^\+\s*@\w+',         # Added decorator
            r'^-\s*@\w+',          # Removed decorator
            r'^\+\s*\w+\s*[:=]', # Added property
            r'^-\s*\w+\s*[:=]',  # Removed property
        ]
        for pattern in schema_patterns:
            if re.search(pattern, diff_text, re.MULTILINE):
                return True
        return False

    def execute_sql_with_error_handling(self, sql_query, bi_engineer_prompts, db_engine):
        """
        Executes a SQL query and returns results or a formatted error message.
        """
        try:
            with db_engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(sql_query))
                # You can format the result as needed, e.g., fetchall()
                return result.fetchall()
        except Exception as e:
            # Use the new sql_error_handling prompt
            error_prompt = bi_engineer_prompts["sql_error_handling"]
            return error_prompt.format(error_message=str(e))

# Enable LLM Observability at startup
LLMObs.enable()

# Utility function for robust model provider selection
# Place this at the top level, not inside the class

def get_model_provider(model_env_var, provider_env_var):
    provider = os.getenv(provider_env_var)
    model = os.getenv(model_env_var, "").lower()
    if provider:
        return provider
    if "gpt" in model:
        return "openai"
    if "gemini" in model:
        return "google"
    if "llama" in model or "meta" in model:
        return "meta"
    if "grok" in model:
        return "xai"
    if "claude" in model or "sonnet" in model:
        return "anthropic"
    # Add more providers as needed
    return "openai"  # Fallback to OpenAI

def markdown_to_slack(text):
    # Convert **bold** to *bold*
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    # Convert [text](url) to <url|text>
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', text)
    # Convert __italic__ or _italic_ to _italic_
    text = re.sub(r'__(.*?)__', r'_\1_', text)
    text = re.sub(r'_(.*?)_', r'_\1_', text)
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
