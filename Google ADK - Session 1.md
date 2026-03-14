# Google Agent Development Kit (ADK)

## What is ADK?

**Agent Development Kit (ADK)** is an open-source framework designed to streamline the development, orchestration, and deployment of AI agents.

While optimized for Gemini models and the Google ecosystem, ADK is:

- **Model-agnostic** – Works with Gemini, OpenAI, and others
- **Deployment-agnostic** – Run locally, on cloud, or in containers
- **Framework-compatible** – Integrates with tools like LangChain, CrewAI, etc.

> The foundation for building intelligent, modular, and production-ready agents.

---

## Why Use ADK? – Core Capabilities

| Capability | Description |
|------------|-------------|
| **Flexible Orchestration** | Design predictable pipelines or adaptive flows using workflow agents and LLM-driven routing |
| **Multi-Agent Architecture** | Build scalable systems by composing specialized agents with clear roles and coordination |
| **Rich Tool Ecosystem** | Extend agent capabilities using built-in tools, custom functions, or external libraries like LangChain and CrewAI |
| **Deployment-Ready** | Easily containerize and deploy agents locally, in the cloud, or with Google's Vertex AI Agent Engine |
| **Built-in Evaluation** | Systematically assess agent outputs and execution steps using configurable test cases |
| **Safety & Security Patterns** | Incorporate design best practices to ensure reliable, secure, and trustworthy agent behavior |

> Build, orchestrate, evaluate, and deploy multi-agent systems with ease

---

## Understanding the Agent Workflow

When a user interacts with an agent, the ADK orchestrates a streamlined, observable execution pipeline:

```
┌─────────────────┐
│  Input Reception │ ← User input captured via interface or API
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent Routing   │ ← ADK determines appropriate agent (single or multi-agent)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Context Injection│ ← Agent receives input + predefined config & instructions
└────────┬────────┘
         ↓
┌─────────────────┐
│  LLM Execution   │ ← Language model processes context, generates response
└────────┬────────┘
         ↓
┌─────────────────┐
│ Response Delivery│ ← Output returned to user interface or downstream system
└────────┬────────┘
         ↓
┌─────────────────┐
│  Event Tracing   │ ← ADK logs granular events for debugging & analysis
└─────────────────┘
```

---

## Key Architectural Advantages

| Advantage | Description |
|-----------|-------------|
| **Separation of Concerns** | Agent logic is decoupled from runtime infrastructure, simplifying development and testing |
| **Modular Design** | Each agent is encapsulated with its own configuration, dependencies, and execution logic |
| **Scalable Extensibility** | New agents, tools, or capabilities can be integrated seamlessly as project requirements evolve |
| **Operational Transparency** | Built-in event logging surfaces the full decision-making trail, enhancing explainability and trust |

> ADK's architecture is engineered for flexibility, clarity, and scale.

---

## Core Components of an Agent

Every agent in ADK consists of four core components:

1. **Prompt / Instructions / Output Format** – Defines agent behavior
2. **Models** – The LLM powering the agent
3. **Tools** – Functions the agent can call
4. **Memory** – Session state and context

---

## ADK Memory Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| **Sessions** | Persistent containers for chat history, user data, and system context |
| **State** | A key-value store for dynamic memory—used to track preferences, history, or structured outputs |
| **Runners** | Orchestrators that connect agents with session state and manage execution flow |

> Enables agents to remember, personalize, and persist knowledge across interactions.

---

## Sessions in Detail

### What is a Session?

A session in ADK includes:

- **ID** – Unique identifier per user or thread
- **State** – Persisted data like preferences or structured outputs
- **Events** – Logged messages, tool invocations, responses
- **Metadata** – App name, user ID, timestamps

> Enables agents to maintain multi-turn context and coherent interactions.

### Session Types

| Type | Description |
|------|-------------|
| **In-Memory Sessions** | Ideal for development, testing, and lightweight use cases. Session data resides in memory and is lost when the application stops |
| **Database Sessions** | Enables persistent storage of session data on local or remote databases. Suitable for applications requiring state retention across restarts |
| **Vertex AI Sessions** | Seamlessly integrates with Google Cloud's Vertex AI for fully managed, cloud-native session persistence and scalability |

---

## Understanding State

State can include:

- User metadata (e.g., name, preferences)
- Results from previous queries
- Structured output (via `output_key`)
- Custom session flags (e.g., `first_interaction`)

> The agent's "memory layer" throughout a session

**Accessing State:**
- Use templates like `{username}` or `{user_preferences.favorite_food}` in prompts

---

## Role of Runners

Runners are responsible for executing agent logic with session context.

### Runners Manage:

- Agent selection and input routing
- Injecting state into agent prompts
- Executing tool calls
- Updating state and event logs
- Returning final output to the user

**Usage:**
```python
runner.run()        # Synchronous execution
runner.run_async()  # Asynchronous execution
```

### The Runner Lifecycle

Step-by-step request flow (all handled internally by Runner):

1. User message received
2. Runner locates session
3. Agent is selected
4. State is injected into prompt
5. Agent executes
6. Events and state updated
7. Final response returned

---

## Session Management: Best Practices

### Managing Multiple Sessions

Real-world apps may serve many users simultaneously:

- Check for existing session
- Resume or create new session dynamically
- Periodically expire old sessions

### Session Expiration & Cleanup

For long-running apps:

- Set age limits (e.g., 24 hours)
- Periodically clean expired sessions
- Reduce memory usage and improve performance

**Key Methods:**
```python
list_sessions()     # List all sessions
create_session()    # Create new session
delete_session()    # Remove session
```

> Use timestamp diff between `now` and `last_update_time` for cleanup.

---

## Introduction to Tool Calling

The ADK enables agents to perform actions beyond text generation by integrating **tools**—external functions or services that the agent can call during execution.

### Three Categories of Tools

| Category | Description |
|----------|-------------|
| **Function Calling Tools** | Custom Python functions for business logic |
| **Built-in Tools** | Predefined tools offered by Google (e.g., Search, Code Execution) |
| **Third-party Tools** | External integrations (e.g., LangChain, CrewAI) |

> Empowering agents with real-world capabilities

### How Agents Use Tools

```
User Request → Agent Analyzes → Tool Selected → Tool Called → Response Incorporated
```

1. User sends a request
2. Agent analyzes request and available tools
3. Relevant tool is selected based on docstring/context
4. Tool is called with appropriate parameters
5. Response is incorporated into final output

---

## Function Calling Tools

Function tools are developer-defined Python functions passed into agents to execute specific tasks.

### Variations

| Type | Description |
|------|-------------|
| **Standard Functions** | Basic operations |
| **Agents-as-Tools** | Agent reuse in multi-agent systems |
| **Long-running Functions** | For tasks with extended durations |

### Best Practices

- ✅ Include rich docstrings and type annotations
- ✅ Return structured dictionaries
- ❌ Avoid default parameters

---

## Built-in Tools

ADK includes pre-built tools that extend agents' capabilities with zero setup.

### Available Tools

| Tool | Description |
|------|-------------|
| **Google Search** | Perform Google web searches |
| **Code Execution** | Execute Python code in a sandbox |
| **RAG** | Retrieve content from vector databases |

### Limitations

- Built-in tools only work with **Gemini models**
- You can use only **one built-in tool per agent**
- Cannot combine built-in tools with custom ones

---

## Third-party Tool Integrations

ADK supports integration with popular tool libraries:

| Library | Capabilities |
|---------|--------------|
| **LangChain tools** | Chains, retrievers, document loaders |
| **CrewAI tools** | Role-based coordination and planning |
| **Custom wrappers** | Interface with internal systems or APIs |

> Leverage the wider agent ecosystem

---

## Tool Debugging & Observability

Tool usage is fully observable in the ADK web UI:

- ✅ See which tool was called
- ✅ Inspect input arguments
- ✅ View returned output
- ✅ Debug failures and mismatches

> This makes tool-driven agents explainable, debuggable, and safe.

---

## Tool Limitations & Best Practices

| Limitation | Recommendation |
|------------|----------------|
| One built-in tool per agent | Split logic across agents |
| No mixing of built-in and custom | Use agent-as-tool to chain functionality |
| Avoid default args in tools | Use required, clearly typed parameters |
| Must return dicts, not raw values | Always use descriptive keys and structure |

---

## Using Different Models in ADK

ADK allows seamless integration of multiple LLM providers, empowering developers to:

- Choose the best model for the task
- Optimize cost vs performance
- Reduce reliance on a single vendor
- Experiment with model features and output styles

### Powered By

| Library | Description |
|---------|-------------|
| **LiteLLM** | Unified interface to multiple LLM APIs |
| **OpenRouter** | Proxy for accessing top providers with a single API key |

> Multi-model flexibility using LiteLLM + OpenRouter

---

## Best Practices for Multi-Model Agents

1. **Define a Clear Model Selection Strategy**
   - Choose models based on task complexity, cost-efficiency, latency requirements, and domain-specific strengths

2. **Implement Fallback Mechanisms**
   - Design agents to gracefully fall back to alternate models if the primary provider is unavailable or throttled

3. **Adapt Instructions for Each Model**
   - Different models respond better to different prompt styles. Customize agent instructions accordingly

4. **Monitor Performance and Cost Metrics**
   - Track response time, token usage, error rates, and user satisfaction

5. **Avoid Feature Assumptions**
   - Not all models support the same capabilities (e.g., tool calling, vision). Design defensively

6. **Maintain Provider Diversity Thoughtfully**
   - Use multiple providers strategically to balance cost, access, and performance

> **Example:** Use Claude for creative generation and GPT-4 for reasoning-intensive tasks.

---

## Structured Outputs with ADK

ADK provides multiple mechanisms to enforce output formats:

| Mechanism | Description |
|-----------|-------------|
| **Output Schema** | Validate output using a Pydantic model |
| **Output Key** | Store structured output in agent state for reuse |
| **Input Schema** | Define expected input format (less common) |

> Structured outputs enforce schemas to avoid inconsistent formatting, missing or extra fields, unusable data for APIs or downstream systems.

### How It Works (Behind the Scenes)

When structured output is enabled, ADK:

1. Collects raw model output
2. Parses it as JSON
3. Validates against the defined schema
4. Stores it under a specified `output_key` in state
5. Raises error or retry logic if validation fails

---

## The Output Schema Pattern

Using `output_schema`, you define the exact structure the agent must follow:

```python
from pydantic import BaseModel

class MyOutput(BaseModel):
    name: str
    score: int
    summary: str

agent = Agent(
    name="structured_agent",
    output_schema=MyOutput,
    ...
)
```

**Benefits:**
- ✅ Enforces type safety
- ✅ Required fields validation
- ✅ Consistent formats

---

## Benefits of Storing in output_key

Storing output in agent state allows:

- Easy access by downstream agents
- Persistent trace of structured responses
- Enhanced debugging and transparency
- Direct integration into orchestrated workflows

---

## Structured Outputs: Limitations & Best Practices

| Topic | Guidance |
|-------|----------|
| **Tool Compatibility** | `output_schema` cannot be used with built-in tools or agent transfer |
| **Explicit Instructions** | Always describe output format clearly in prompts |
| **Error Handling** | Plan for failures—use retries or fallback logic |
| **Start Simple** | Avoid overly complex schemas early on |

---

## Summary

Google ADK provides a comprehensive framework for building production-ready AI agents with:

- 🔧 **Flexible tool integration**
- 🧠 **Robust memory management**
- 🔄 **Multi-model support**
- 📊 **Structured outputs**
- 🔍 **Full observability**

> Build smarter, more resilient agent systems using the right model for the right task.
