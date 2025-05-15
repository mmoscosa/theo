# Theo Crew

Theo Crew is a multi-agent, production-grade FastAPI service orchestrating LLM-powered agents for support, documentation, BI, and product management workflows. It features:
- **Semantic documentation updates:** LLM+Bedrock finds and updates the most relevant Confluence page for a topic, using both semantic search and exact/fuzzy title matching. No duplicate pages are created—if a page exists, it is always updated.
- **White-checkmark reaction deduplication:** The ✅ reaction only triggers a documentation update if it hasn't already been done for the thread, preventing redundant updates.
- **Robust Slack notifications:** Slack messages about documentation updates are only sent if the update succeeds. Clear error messages are sent if something goes wrong.
- **Error handling & observability:** All errors are logged, surfaced in Slack, and traced in Datadog LLM Observability for full traceability.
- **Production-grade automation:** All configuration is via environment variables. The system is robust, observable, and user-friendly for automated documentation, support, and analytics workflows.
- **Slack-first conversational UX** (reactions, thread-based, admin escalation)
- **Hot-reload dev workflow** with Docker Compose
- **AWS Bedrock** for vector search knowledge bases
- **Extensible agent system** powered by [crewAI](https://crewai.com)

---

## Features
- **Slack Integration:**
  - Thread-based conversations, reactions for status (🤖, ⏳, 🌐), and admin escalation
  - Conversation conclusion via ✅ or ❌ reactions, with rating prompt or admin tag
  - Bot only responds in concluded threads if explicitly tagged
- **LLM Observability:**
  - Full workflow and LLM call tracing in Datadog
  - Audit trail and error metrics
- **Agents:**
  - **Supervisor**: routes and confirms requests
  - **Support Engineer**: answers support questions, consults KBs
  - **Technical Writer**: updates documentation
  - **BI Engineer**: generates reports and analytics
  - **Product Manager**: creates and manages tickets
  - **LLM-based thread summarization**: Supervisor uses an LLM to summarize Slack thread context for downstream agents, improving context focus and reducing noise.
  - **Advanced agent context**: BI Engineer and Technical Writer receive both the LLM summary and the full raw thread as context for the LLM prompt, but only the LLM-generated answer and summary are shown in Slack (the raw thread is not included in Slack messages).
- **Knowledge Bases:**
  - AWS Bedrock (us-east-1) for code, DB schema, and general knowledge
- **Modern Dev Workflow:**
  - Hot reload with `docker-compose.dev.yml`
  - All secrets/config from `.env` (see `.env.example`)

---

## Quickstart

### 1. Clone & Configure
- Copy `.env.example` to `.env` and fill in all required keys (OpenAI, Anthropic, AWS, Datadog, Slack, etc.)
- Invite your bot to your Slack workspace and channel

### 2. Run in Dev Mode (with Hot Reload)
```sh
docker compose -f docker-compose.dev.yml up
```
- Code changes are picked up instantly
- FastAPI runs on [http://localhost:8000](http://localhost:8000)

### 3. Expose Locally (for Slack)
- Use [ngrok](https://ngrok.com/) or similar:
  ```sh
  ngrok http 8000
  ```
- Set your Slack app's event/webhook URL to the ngrok HTTPS URL (e.g., `https://xxxx.ngrok.io/slack/events`)

### 4. Test via Slack
- Mention your bot in a channel or thread (e.g., `@Theo Can I get a report on user signups?`)
- Watch for reactions and responses
- Conclude a conversation with ✅ or ❌
- To bring the bot back after conclusion, mention it again in the thread

### 5. Test via API
- Use the FastAPI docs at [http://localhost:8000/docs](http://localhost:8000/docs)
- Try `/slack/events` and `/github/webhook` endpoints

---

## Environment Variables
See `.env.example` for all required variables, including:
- LLM API keys (OpenAI, Anthropic, Google, AWS Bedrock)
- Datadog (DD_API_KEY, DD_LLMOBS_ML_APP, etc.)
- Slack (SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, SLACK_BOT_USER_ID, ADMIN_SLACK_USER_ID)
- GitHub, Jira, Confluence, and agent model configs

---

## Agent Roles
- **Supervisor** (`gemini-2.5-pro-preview-05-06`): routes and confirms
- **Support Engineer** (`o3-2025-04-16`): support, KB search
- **Technical Writer** (`claude-3-5-sonnet-20240620`): documentation
- **BI Engineer** (`claude-3-5-sonnet-20240620`): analytics/reports
- **Product Manager** (`chatgpt-4o-latest-20250326`): tickets/issues

### Advanced Context Handling
- **Supervisor**: Summarizes thread context using LLM before routing.
- **BI Engineer**: Receives both the summary and the raw thread as LLM context, but only the LLM-generated answer and summary are shown in Slack. Timeline and raw thread are not included in Slack messages.
- **Technical Writer**: Receives both the summary and the raw thread as LLM context, but only the LLM-generated answer and summary are shown in Slack. Structured context is used for documentation, but not shown in Slack messages.

---

## Slack UX & Conversation Flow
- **Reactions:**
  - 🤖 = Request received
  - ⏳ = Processing (removed when done)
  - 🌐 = Error (admin tagged)
- **Conversation Conclusion:**
  - ✅ = Thank you + rating prompt, bot stops responding unless tagged
  - ❌ = Apology + admin tag, bot stops responding unless tagged
- **Thread-based:**
  - Bot only responds in a thread after conclusion if explicitly tagged (e.g., `@Theo`)

## Advanced Agent Context Example

**BI Engineer Timeline Extraction:**
When a BI report is requested, the BI Engineer agent receives both the LLM-generated summary and the full Slack thread as context for the LLM prompt. The agent may build a timeline of all user and agent messages for analytics and reporting, but only the LLM-generated answer and summary are included in the Slack response (not the full timeline or raw thread). This enables:
- Richer analytics (e.g., event sequence, user intent, escalation path)
- Traceable, context-aware reports

**Technical Writer Structured Context:**
The Technical Writer uses the raw thread as LLM context to extract all relevant technical changes, user clarifications, and agent responses, ensuring documentation is accurate and complete. Only the LLM-generated answer and summary are shown in Slack.

---

## LLM Observability
- All workflows and LLM calls are traced in Datadog
- Audit events and error metrics are sent for every message
- See [Datadog LLM Observability docs](https://docs.datadoghq.com/llm_observability/)

---

## Troubleshooting
- **No Slack replies?**
  - Check your `.env` for correct `SLACK_BOT_TOKEN` and `SLACK_BOT_USER_ID`
  - Make sure your bot is invited to the channel
  - Check ngrok or tunnel logs for incoming requests
- **No Datadog traces?**
  - Check Datadog API key and LLMObs config in `.env`
  - Make sure you are using the correct region/site
- **Bot spamming or duplicate replies?**
  - The system deduplicates by message timestamp and only responds once per event
- **Conversation not concluding?**
  - Make sure you react with ✅ or ❌ to the parent message in the thread

---

## Contributing & Extending
- Define new agents in `src/theo/config/agents.yaml`
- Add new tasks in `src/theo/config/tasks.yaml`
- Extend agent logic in `src/theo/crew.py`
- Add new tools in `src/theo/tools/`
- PRs and issues welcome!

---

## License
MIT
