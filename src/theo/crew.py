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
from theo.tools.confluence import update_confluence_page, create_confluence_page, get_all_confluence_pages
import html

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

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
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

    def summarize_thread(self, thread_history, question=None):
        """Summarize a Slack thread using the supervisor's LLM model, and include relevant general knowledge from Bedrock only if useful."""
        model = os.getenv("AGENT_SUPERVISOR_MODEL", "unknown")
        bedrock = BedrockClient()
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
        all_msgs = [extract_message_text(msg) for msg in thread_history if extract_message_text(msg)]
        thread_text = "\n".join(all_msgs)
        # Query general knowledge base with the question and thread context
        general_kb_query = question or thread_text
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
            general_knowledge_section = (
                f"\nHere is relevant general knowledge from previous conversations or documentation:\n---\n{general_knowledge}\n---\n"
            )
        @llm(model_name=model, name="supervisor_thread_summary", model_provider="openai")
        def _llm_call():
            prompt = (
                f"Summarize the following Slack thread for context. Focus on user questions, requests, and key information.\n\n"
                f"Messages:\n{thread_text}\n"
                f"{general_knowledge_section}"
            )
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
                summary = thread_text[:1000]  # Use the actual messages as fallback
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
        doc_prompt = (
            "You are a technical writer. Write a clear, professional, and well-structured technical documentation page based on the following conversation and summary. The documentation should be suitable for both product and engineering team members. Do not include Slack formatting, agent signatures, or conversational filler.\n\n"
            f"Conversation context:\n{timeline_str}\n\n"
            f"Summary:\n{context_summary}\n\n"
            "Write only the documentation content."
        )
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
        title_prompt = (
            "Given the following documentation content, generate a concise, descriptive Confluence page title suitable for both product and engineering audiences.\n\n"
            f"Content:\n{doc_content}\n"
        )
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
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

    def support_engineer_answer(self, question, conversation_id=None, context_summary=None, thread_history=None):
        import litellm
        bedrock = BedrockClient()
        agent_role = "Support Engineer"
        agent_emoji = "👨‍💻"
        category_emoji = "📄"
        category_title = "Support Request"
        model = os.getenv("AGENT_SUPPORT_ENGINEER_MODEL", "gpt-4o")
        model_provider = get_model_provider("AGENT_SUPPORT_ENGINEER_MODEL", "AGENT_SUPPORT_ENGINEER_PROVIDER")
        context_summary_clean = "" if context_summary in (None, "None") else context_summary
        docs = bedrock.search_code_documentation(question)
        print(f"[DEBUG] Bedrock docs: {docs}")
        llm_prompt = (
            f"You are a Support Engineer.\n"
            f"A user asked: '{question}'\n"
            f"\nHere is relevant documentation:\n---\n{docs}\n---\n"
            f"Conversation context (summarized thread):\n{context_summary_clean}\n"
            f"Using the documentation above, answer the user's question in a clear, concise, and user-friendly way. "
            f"If the answer is not directly in the documentation, say so and suggest next steps.\n"
            f"Format your answer for Slack, using bullet points or sections if helpful."
        )
        print(f"[DEBUG] LLM prompt sent to support engineer:\n{llm_prompt}")
        from ddtrace.llmobs import LLMObs
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Support Engineer."},
                    {"role": "user", "content": llm_prompt}
                ]
            )
            main_content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
            if not main_content or not isinstance(main_content, str) or main_content.strip().lower() == "none":
                main_content = "Sorry, I couldn't find a relevant answer in the documentation. Please provide more details or contact support."
            # Log to LLMObs for observability
            LLMObs.annotate(
                input_data=[{"role": "system", "content": "You are a Support Engineer."}, {"role": "user", "content": llm_prompt}],
                output_data=[{"role": "assistant", "content": main_content}],
                tags={"agent_role": agent_role, "model_provider": model_provider}
            )
            print(f"[DEBUG] LLM raw output: {main_content}")
            main_content = markdown_to_slack(main_content)
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            main_content = "Sorry, I couldn't find a relevant answer in the documentation. Please provide more details or contact support."
        return format_slack_response(
            category_emoji=category_emoji,
            category_title=category_title,
            main_content=main_content if isinstance(main_content, str) else str(main_content),
            agent_role=agent_role,
            agent_emoji=agent_emoji
        )

    def supervisor_answer(self, question, conversation_id=None, context_summary=None):
        agent_role = "supervisor"
        model = os.getenv("AGENT_SUPERVISOR_MODEL", "gpt-4o")
        model_provider = get_model_provider("AGENT_SUPERVISOR_MODEL", "AGENT_SUPERVISOR_PROVIDER")
        context_summary = context_summary or ""
        valid_tasks = {"support_request", "documentation_update", "bi_report", "ticket_creation", "clarification_needed"}
        prompt = (
            "You are an agent router. Your job is to select the most appropriate agent for each user request, based on the intent of the message.\n"
            "\n"
            "Agent mapping:\n"
            "- support_request: for troubleshooting, help, errors, support, or general product/system questions\n"
            "- bi_report: for reports, analytics, SQL, data, metrics\n"
            "- documentation_update: for explicit requests to document, update documentation, write docs, or similar\n"
            "- ticket_creation: for explicit requests to create, add, or file a ticket, bug, or feature\n"
            "- clarification_needed: if the request is ambiguous or unclear\n"
            "\n"
            "Instructions:\n"
            "- If the user's request is a follow-up in a thread, use the conversation context to determine the correct agent. Do not ask for clarification if the context makes the intent clear.\n"
            "- Route to support_request for most user questions about how to use, troubleshoot, or understand a system, feature, or process, unless the user is explicitly asking for documentation to be written or updated.\n"
            "- Route to documentation_update only if the user explicitly asks to document, update documentation, write docs, or similar.\n"
            "- Route to bi_report for requests about reports, analytics, SQL, data, or metrics.\n"
            "- Route to ticket_creation only for explicit requests to create, add, or file a ticket, bug, or feature.\n"
            "- If you are unsure, choose clarification_needed.\n"
            "\n"
            "Examples:\n"
            "- 'How does the PMSSync work?' => support_request\n"
            "- 'Can you help me with PMSSync errors?' => support_request\n"
            "- 'Update the onboarding docs' => documentation_update\n"
            "- 'Can you give me a SQL query for user signups?' => bi_report\n"
            "- 'Create a ticket for this bug' => ticket_creation\n"
            "\n"
            f"Conversation context (summarized thread):\n{context_summary}\n"
            f"User request: {question}\n"
            "\nRespond with ONLY one of: support_request, documentation_update, bi_report, ticket_creation, clarification_needed."
        )
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
            # Log to LLMObs for observability
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
        # Heuristic fallback: if answer is clarification_needed but context_summary contains a recent support request, override to support_request
        if answer == "clarification_needed" and context_summary:
            lowered_context = context_summary.lower()
            if ("support request" in lowered_context or "troubleshoot" in lowered_context or "error" in lowered_context or "help" in lowered_context):
                print("[DEBUG] Heuristic fallback: context indicates support, overriding to support_request")
                answer = "support_request"
        downstream_prompt = f"Conversation context (summarized thread):\n{context_summary}\n\nUser request: {question}"
        return answer, downstream_prompt

    def technical_writer_answer(self, question, conversation_id=None, context_summary=None, thread_history=None):
        print(f"[DEBUG] technical_writer_answer called with question={question}, conversation_id={conversation_id}, context_summary={context_summary}, thread_history={thread_history}")
        agent_role = "Technical Writer"
        agent_emoji = "📝"
        category_emoji = "📚"
        category_title = "Documentation Update"
        model = os.getenv("AGENT_TECHNICAL_WRITER_MODEL", "unknown")
        @llm(model_name=model, name="technical_writer_answer", model_provider="anthropic")
        def _llm_call():
            # Use the raw thread only as part of the LLM prompt, not in the Slack message
            timeline = []
            if thread_history:
                for msg in thread_history:
                    user = msg.get("user") or msg.get("username") or ("bot" if msg.get("bot_id") else "unknown")
                    text = msg.get("text", "").strip()
                    ts = msg.get("ts", "")
                    if text:
                        timeline.append(f"[{ts}] {user}: {text}")
            timeline_str = "\n".join(timeline) if timeline else "No timeline available."
            # Build a rich prompt for the LLM
            llm_prompt = (
                f"You are a Technical Writer.\n"
                f"User question or documentation request: {question}\n"
                f"Summary of conversation context: {context_summary}\n"
                f"Conversation timeline:\n{timeline_str}\n"
                f"Generate a clear, user-friendly documentation update or summary for the user."
            )
            print(f"[DEBUG] technical_writer_answer LLM prompt: {llm_prompt}")
            # The LLM generates the main content, but only the answer and summary are sent to Slack
            main_content = LLMObs.annotate(
                input_data=[{"role": "user", "content": llm_prompt}],
                output_data=[{"role": "assistant", "content": ""}],
                tags={"agent_role": agent_role, "conversation_id": conversation_id or "unknown", "model_provider": model_provider}
            )
            print(f"[DEBUG] technical_writer_answer LLM main_content: {main_content}")
            # Only show the LLM's answer and summary in Slack, not the full timeline
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
        agent_role = "BI Engineer"
        agent_emoji = "📊"
        category_emoji = "📈"
        category_title = "BI Report"
        model = os.getenv("AGENT_BI_ENGINEER_MODEL", "unknown")
        model_provider = get_model_provider("AGENT_BI_ENGINEER_MODEL", "AGENT_BI_ENGINEER_PROVIDER")
        bedrock = BedrockClient()
        db_docs = bedrock.search_db_schema(question)
        from ddtrace.llmobs import LLMObs
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
            llm_prompt = (
                f"You are a BI Engineer.\n"
                f"User question: {question}\n"
                f"Summary of conversation context: {context_summary}\n"
                f"\nHere is relevant database schema and query documentation:\n---\n{db_docs}\n---\n"
                f"Conversation timeline:\n{timeline_str}\n"
                f"Generate a concise, context-aware BI report for the user."
            )
            print(f"[DEBUG] bi_engineer_answer LLM prompt: {llm_prompt}")
            main_content = LLMObs.annotate(
                input_data=[{"role": "user", "content": llm_prompt}],
                output_data=[{"role": "assistant", "content": ""}],
                tags={"agent_role": agent_role, "conversation_id": conversation_id or "unknown", "model_provider": model_provider}
            )
            print(f"[DEBUG] bi_engineer_answer LLM main_content: {main_content}")
            # If LLM returns None, empty, or 'none', provide a helpful fallback message
            if not main_content or not isinstance(main_content, str) or main_content.strip().lower() == "none":
                main_content = ("Sorry, I could not find any relevant database schema or queries for your request. "
                                "Please provide more details or check back later.")
            slack_message = format_slack_response(
                category_emoji=category_emoji,
                category_title=category_title,
                main_content=main_content if isinstance(main_content, str) else str(main_content),
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
        agent_role = "Product Manager"
        agent_emoji = "📝"
        category_emoji = "📝"
        category_title = "Ticket Creation"
        model = os.getenv("AGENT_PRODUCT_MANAGER_MODEL", "gpt-4o")
        model_provider = get_model_provider("AGENT_PRODUCT_MANAGER_MODEL", "AGENT_PRODUCT_MANAGER_PROVIDER")
        admin_slack_user_id = os.getenv("ADMIN_SLACK_USER_ID")
        admin_tag = f"<@{admin_slack_user_id}> " if admin_slack_user_id else ""
        VALID_ISSUE_TYPES = ["Story", "Spike", "Bug", "Hotfix"]
        MAX_SUMMARY_LEN = 80
        MAX_TIMELINE_MESSAGES = 10
        def extract_json_from_llm_output(content):
            import json
            try:
                return json.loads(content)
            except Exception:
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except Exception as e:
                        logging.warning(f"Failed to parse extracted JSON: {e}")
                logging.warning("LLM output could not be parsed as JSON. Raw output: %s", content)
                return {}
        def filter_thread_history(thread_history, current_channel=None, current_thread_ts=None):
            filtered = []
            for msg in thread_history or []:
                if current_thread_ts and msg.get("thread_ts") != current_thread_ts:
                    continue
                if msg.get("bot_id") and "Product Manager" not in msg.get("text", ""):
                    continue
                filtered.append(msg)
            return filtered
        filtered_thread_history = filter_thread_history(thread_history, channel, thread_ts)
        logging.info(f"Filtered thread history for LLM: {filtered_thread_history}")
        def get_ticket_type_and_summary():
            timeline = []
            if filtered_thread_history:
                for msg in filtered_thread_history:
                    user = msg.get("user") or msg.get("username") or ("bot" if msg.get("bot_id") else "unknown")
                    text = msg.get("text", "").strip()
                    ts = msg.get("ts", "")
                    if text:
                        timeline.append(f"[{ts}] {user}: {text}")
            timeline_str = "\n".join(timeline)
            logging.info(f"Timeline sent to LLM for ticket creation:\n{timeline_str}")
            llm_prompt = (
                "You must respond ONLY with a valid JSON object.\n"
                "You are a Product Manager.\n"
                "Given the following conversation, classify the ticket as one of: Story, Spike, Bug, Hotfix.\n"
                "Generate a JIRA-ready summary (one short, action-oriented, title-style sentence, max 80 chars) and a detailed description (include all relevant context, steps, and expected behavior).\n"
                "IMPORTANT: The summary and description must be based on the technical details, troubleshooting steps, and all relevant information discussed by all participants in the conversation. Do NOT focus only on the last message or only on support engineer responses. Synthesize the ticket from the entire thread.\n"
                "When listing steps or items, use Markdown format (`- item` for bullets, `1. item` for numbered lists).\n"
                "Respond in JSON with keys: type, summary, description.\n"
                f"Conversation timeline:\n{timeline_str if timeline_str else ''}\n"
                f"User request: {question}\n"
            )
            logging.info(f"LLM prompt: {llm_prompt}")
            import litellm
            from ddtrace.llmobs import LLMObs
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You must respond ONLY with a valid JSON object."},
                    {"role": "system", "content": "You are a Product Manager."},
                    {"role": "user", "content": llm_prompt}
                ]
            )
            content = response['choices'][0]['message']['content'] if response and 'choices' in response else None
            if not content:
                content = "{}"
            # Log to LLMObs for observability
            LLMObs.annotate(
                input_data=[{"role": "system", "content": "You must respond ONLY with a valid JSON object."}, {"role": "system", "content": "You are a Product Manager."}, {"role": "user", "content": llm_prompt}],
                output_data=[{"role": "assistant", "content": content}],
                tags={"agent_role": agent_role, "model_provider": model_provider}
            )
            logging.info(f"Raw LLM output: {content}")
            ticket_info = extract_json_from_llm_output(content)
            issue_type = ticket_info.get("type", "Story")
            if issue_type not in VALID_ISSUE_TYPES:
                logging.warning(f"Invalid or missing issue type '{issue_type}', defaulting to 'Story'.")
                issue_type = "Story"
            summary = ticket_info.get("summary", "").strip()
            if not summary:
                summary = f"{question[:MAX_SUMMARY_LEN]}"
                logging.warning("Summary missing from LLM, using fallback.")
            if len(summary) > MAX_SUMMARY_LEN:
                summary = summary[:MAX_SUMMARY_LEN-3] + "..."
            description = ticket_info.get("description", "").strip()
            if not description:
                if timeline_str:
                    description = timeline_str + f"\n\nUser request: {question}"
                else:
                    description = f"User request: {question}"
                logging.warning("Description missing from LLM, using fallback.")
            return {"type": issue_type, "summary": summary, "description": description}
        ticket_info = get_ticket_type_and_summary()
        issue_type = ticket_info["type"]
        summary = ticket_info["summary"]
        description = ticket_info["description"]
        description = re.sub(r'^Slack conversation:.*$', '', description, flags=re.MULTILINE).strip()
        slack_link = None
        if channel and thread_ts:
            slack_link = self.build_slack_permalink(channel, thread_ts)
            logging.info(f"Slack permalink being built: {slack_link}")
            description += f"\n\nSlack conversation: {slack_link}"
        else:
            description += "\n\nSlack conversation: None provided"
        logging.info(f"[CREW] slack_link: {slack_link}")
        logging.info(f"[CREW] Final JIRA description (before ticket creation): {description!r}")
        jira_result = create_jira_ticket(summary, description, issue_type=issue_type)
        if "error" in jira_result:
            main_content = f"Failed to create JIRA ticket: {jira_result['error']}"
            ticket_url = None
            ticket_key = None
        else:
            ticket_key = jira_result.get("key")
            ticket_url = jira_result.get("url")
            main_content = f"Created JIRA ticket: <{ticket_url}|{ticket_key}> ({issue_type})\nSummary: {summary}\nPlease assign this ticket."
        return format_slack_response(
            category_emoji=category_emoji,
            category_title=category_title,
            main_content=admin_tag + main_content,
            agent_role=agent_role,
            agent_emoji=agent_emoji
        )

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
