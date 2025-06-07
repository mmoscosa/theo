# Theo Crew

Theo Crew is a **high-performance, multi-agent FastAPI service** that orchestrates LLM-powered agents for support, documentation, BI, and product management workflows. Built for production with **concurrent processing** and **real-time progress indicators**.

## 🚀 **Key Features**

### **Concurrent Processing Architecture**
- **Up to 10 simultaneous conversations** with thread-safe processing
- **< 10ms initial response time** to Slack events
- **Background task execution** prevents blocking and timeouts
- **45-second conversation timeout** with automatic cleanup
- **Duplicate detection** and conversation state management

### **Smart Progress Indicators**
- **🤖** = Agent ownership (persistent throughout conversation)  
- **⏳ → 🧠 → 🔍 → 📝** = Processing stages (auto-cleaned when complete)
- **🌐** = Error occurred (admin notification triggered)
- **Real-time status updates** without overwhelming users

### **Production-Grade Reliability**
- **Semantic documentation updates:** LLM+Bedrock finds and updates the most relevant Confluence page for a topic, using both semantic search and exact/fuzzy title matching. No duplicate pages are created—if a page exists, it is always updated.
- **White-checkmark reaction deduplication:** The ✅ reaction only triggers a documentation update if it hasn't already been done for the thread, preventing redundant updates.
- **Robust error handling:** All failures are logged, surfaced in Slack, and traced in Datadog LLM Observability for full traceability.
- **Thread-safe operations** with semaphore-based resource management
- **Automatic conversation cleanup** after timeout or completion

### **Advanced Agent System**
- **LLM-based thread summarization**: Supervisor uses an LLM to summarize Slack thread context for downstream agents
- **Context-aware routing**: Agents receive both summarized and raw thread context for optimal responses
- **Agent signature filtering**: Prevents LLMs from copying previous agent signatures
- **Bedrock Knowledge Bases**: AWS Bedrock (us-east-1) for code, DB schema, and general knowledge

---

## 🏗 **Architecture Overview**

### **Agent Roles & Models**
- **Supervisor** (`gemini-2.5-pro-preview-05-06`): Intelligent routing and conversation orchestration
- **Support Engineer** (`o3-2025-04-16`): Technical support with Bedrock KB integration  
- **Technical Writer** (`claude-3-5-sonnet-20240620`): Documentation automation
- **BI Engineer** (`claude-3-5-sonnet-20240620`): Analytics and SQL generation
- **Product Manager** (`chatgpt-4o-latest-20250326`): Ticket creation and management

### **Concurrent Processing Flow**
1. **Immediate Response** (< 10ms): Slack receives 200 OK, 🤖 reaction added
2. **Background Processing**: Conversation routed to appropriate agent with progress indicators
3. **Stage Updates**: ⏳ → 🧠 → 🔍 → 📝 show real-time progress
4. **Completion**: Progress emojis removed, 🤖 remains as ownership indicator
5. **Cleanup**: Automatic timeout and resource management

### **Thread-Safe State Management**
- **In-memory conversation tracking** with automatic expiration
- **Reaction conflict prevention** via centralized reaction manager  
- **Resource pooling** with semaphore limits (max 10 concurrent)
- **Graceful failure handling** with retry logic and admin notifications

---

## 🎯 **Slack UX & Interaction Flow**

### **Conversation Lifecycle**
1. **Initiation**: Mention `@Theo` or message in thread
2. **Acknowledgment**: 🤖 reaction appears immediately (< 10ms)
3. **Processing**: Stage indicators show progress (⏳ → 🧠 → 🔍 → 📝)
4. **Response**: Agent delivers answer, progress emojis removed
5. **Conclusion**: ✅ (success) or ❌ (escalation) to end conversation

### **Progress Reaction System**
```
🤖               # Agent ownership (persistent)
🤖 + ⏳          # Processing started  
🤖 + 🧠          # Thinking/analyzing
🤖 + 🔍          # Researching knowledge bases
🤖 + 📝          # Writing response
🤖               # Completed (progress emojis removed)
```

### **Error Handling & Admin Escalation**
- **🌐** reaction + admin tag for critical errors
- **Timeout warnings** after 30 seconds of processing
- **Automatic retry** (2 attempts) before graceful failure
- **Admin notifications** via `@ADMIN_SLACK_USER_ID` for unhandled issues

---

## 🚀 **Quick Start**

### **1. Setup & Configuration**
```bash
# Clone and configure
cp .env.example .env
# Fill in all required keys (OpenAI, Anthropic, AWS, Datadog, Slack, etc.)

# Invite bot to Slack workspace and channels
```

### **2. Development Mode (Hot Reload)**
```bash
# Start with automatic code reloading
docker compose -f docker-compose.dev.yml up

# FastAPI available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### **3. Slack Integration**
```bash
# Expose locally for Slack webhooks
ngrok http 8000

# Configure Slack app webhook URL: https://xxxx.ngrok.io/slack/events
```

### **4. Testing & Monitoring**
```bash
# Test conversations in Slack
@Theo Can you help me with [question]?

# Monitor concurrent conversations
curl http://localhost:8000/conversations/stats

# View processing logs
docker compose -f docker-compose.dev.yml logs -f
```

---

## 📊 **Performance & Monitoring**

### **Response Times**
- **Slack Event Response**: < 10ms (immediate acknowledgment)
- **Background Processing**: 15-45 seconds (with progress indicators)
- **Concurrent Capacity**: Up to 10 simultaneous conversations
- **Timeout Handling**: 45-second limit with graceful degradation

### **Observability Stack**
- **Datadog LLM Observability**: Full workflow tracing and metrics
- **Conversation Statistics**: Real-time processing status via `/conversations/stats`
- **Audit Trail**: All events, handoffs, and errors logged to Datadog
- **Error Metrics**: Success/fail rates, timeout tracking, retry attempts

### **System Health Checks**
- `/health` and `/healthcheck` endpoints for monitoring
- Automatic conversation cleanup task
- Resource leak prevention with semaphore limits
- Graceful shutdown handling for container deployments

---

## 🔧 **Environment Configuration**

### **Required Variables**
```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# Datadog Observability  
DD_API_KEY=...
DD_LLMOBS_ML_APP=Theo_Agent_Production
DD_LLMOBS_ENABLED=1

# Slack Integration
SLACK_BOT_TOKEN=xoxb-...
SLACK_BOT_USER_ID=U...
ADMIN_SLACK_USER_ID=U...
SLACK_SIGNING_SECRET=...

# Agent Models (customizable)
AGENT_SUPERVISOR_MODEL=gemini-2.5-pro-preview-05-06
AGENT_SUPPORT_ENGINEER_MODEL=o3-2025-04-16
AGENT_TECHNICAL_WRITER_MODEL=claude-3-5-sonnet-20240620
AGENT_BI_ENGINEER_MODEL=claude-3-5-sonnet-20240620
AGENT_PRODUCT_MANAGER_MODEL=chatgpt-4o-latest-20250326
```

---

## 🏗 **Advanced Features**

### **Intelligent Agent Routing**
- **Context-aware supervisor** analyzes conversation intent
- **Automatic handoff** to appropriate specialist agent
- **Ambiguity detection** with clarification requests
- **Multi-step workflow support** for complex requests

### **Knowledge Base Integration**
- **AWS Bedrock vector search** across code documentation
- **Database schema queries** for BI analysis
- **General knowledge retrieval** for support questions
- **Semantic + fuzzy matching** for optimal document discovery

### **Production Deployment Features**
- **Docker containerization** with health checks
- **Hot-reload development** workflow  
- **Environment-based configuration** (no hardcoded values)
- **Graceful error handling** with user-friendly messages
- **Admin escalation patterns** for unhandled scenarios

---

## 🛠 **Troubleshooting**

### **Common Issues**
| Issue | Solution |
|-------|----------|
| No Slack responses | Check `SLACK_BOT_TOKEN` and bot channel permissions |
| Slow responses | Monitor `/conversations/stats` for resource bottlenecks |
| Missing progress indicators | Verify bot has reaction permissions in channel |
| Timeout errors | Check Bedrock/LLM API connectivity and rate limits |
| Duplicate conversations | System auto-deduplicates; check for timestamp issues |

### **Development Tips**
- Use `docker compose -f docker-compose.dev.yml logs -f` to watch real-time processing
- Test concurrent load with multiple Slack conversations simultaneously  
- Monitor conversation stats endpoint to verify resource management
- Check Datadog traces for detailed LLM call performance analysis

---

## 🤝 **Contributing & Extension**

### **Adding New Agents**
1. Define agent in `src/theo/config/agents.yaml`
2. Add prompts in `src/theo/config/{agent}_prompts.yaml` 
3. Implement methods in `src/theo/crew.py`
4. Update supervisor routing in `src/theo/config/supervisor_prompts.yaml`

### **Custom Tools & Integrations**
- Add new tools in `src/theo/tools/`
- Extend knowledge bases via Bedrock configuration
- Integrate additional APIs (Jira, Confluence, GitHub, etc.)
- Implement custom reaction patterns and UX flows

---

## 📄 **License**
MIT License - See [LICENSE](LICENSE) for details.

---

## 🎯 **Production Ready**
Theo Crew is built for production deployment with enterprise-grade reliability, observability, and performance. The concurrent architecture ensures your team can have multiple simultaneous conversations without blocking or degraded performance.
