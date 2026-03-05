---
type: fetched_content
source: https://github.com/orgs/langfuse/discussions/11934
title: LangChain Integration Issue After Upgrading from Langfuse v2 to v3 - Traces Not Being Captured
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: 并发环境回调管理
---

# LangChain Integration Issue After Upgrading from Langfuse v2 to v3 - Traces Not Being Captured

**langfuse · Discussion #11934**

**Labels**: [sdk-python](https://github.com/orgs/langfuse/discussions?discussions_q=label%3Asdk-python) [integration-langchain](https://github.com/orgs/langfuse/discussions?discussions_q=label%3Aintegration-langchain)

**Participants**: 2

## Environment

- Langfuse Python SDK: 3.12.1
- LangChain version: 0.3.27
- Python version: 3.12.7
- Self-hosted Langfuse at: http://xx.xxx.xxx.xxx:xxxx/
- Framework: Django REST Framework

## Working Code (v2.x)

```python
from langfuse.callback import CallbackHandler
from langfuse import Langfuse

# Initialization
langfuse_sdk = Langfuse(
    secret_key="sk-lf-xxx",
    public_key="pk-lf-xxx",
    host="http://xx.xxx.xxx.xxx:xxxx/"
)

# Custom wrapper class (not standard Langfuse)
langfuse_handler = RealTimeTraceCaptureWrapper(
    public_key="pk-lf-xxx",
    secret_key="sk-lf-xxx",
    host="http://xx.xxx.xxx.xxx:xxxx/"
)

# Usage in Django view
def reconciliation(request):
    chain = build_chain("get_python", tenant_id)
    response = chain.invoke(
        input={...},
        config={
            "run_name": "test",
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_user_id": user_name}
        }
    )
    trace_id = langfuse_handler.trace_id
    return JsonResponse({"trace_id": trace_id, "response": response})
```

This setup consistently produced traces in the Langfuse UI and returned a valid trace ID.

## Non-Working Code (v3.x - Following Migration Guide)

```python
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

# Initialization
langfuse_sdk = Langfuse(
    secret_key="sk-lf-xxx",
    public_key="pk-lf-xxx",
    base_url="http://xx.xxx.xxx.xxx:xxxx/"  # Changed from 'host'
)

langfuse_handler = CallbackHandler()

def reconciliation(request):
    chain = build_chain("get_python", tenant_id)
    response = chain.invoke(
        input={
            "source_csv": source_csv,
            "target_csv": target_csv,
            "metadata_csv": metadata_csv
        },
        config={
            "runName": "test",  # Changed from 'run_name'
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_user_id": user_name}
        }
    )
    trace_id = langfuse_sdk.get_current_trace_id()
    langfuse_sdk.flush()
    return JsonResponse({"trace_id": trace_id, "response": response})
```

## Issues

- No traces appear in the Langfuse UI
- `langfuse_sdk.get_current_trace_id()` returns None
- No errors in console/logs. The response is returned as expected.
- LangChain execution works fine, but Langfuse doesn't capture it

## What I've Tried

- Changed host to base_url parameter
- Changed import from langfuse.callback to langfuse.langchain
- Added langfuse_sdk.flush() before returning response
- Changed run_name to runName
- Re-initializing the client inside the request scope

None of the above resulted in traces being captured.

## Questions

1. Is there additional OpenTelemetry setup required for Django applications in v3?
2. Are there known issues with self-hosted Langfuse and the v3 SDK?
3. How should I have Langfuse capture the LLM calls and also properly retrieve trace IDs in v3 for short-lived Django requests?
4. Is the simple callback handler approach compatible with Django's request-response cycle?
5. If I need to later extend this to capture LLM calls made via Celery, will I need to make any additional changes?

## Important note on design constraints

I'm aware of the start_as_current_observation / propagate_attributes wrapper-based approach suggested in the documentation. However, due to internal architectural and compliance constraints, we cannot wrap LLM execution logic in Langfuse-specific context managers. We need a solution that works with the LangChain callback-based integration only, similar to how this functioned in v2.

Any guidance on what is missing or changed behavior in v3 would be greatly appreciated.

## Reply 1: Inkeep AI Assistant

Based on the migration guide and LangChain integration documentation, there are several critical changes in v3 that affect your setup.

### Key Changes in v3 LangChain Integration

**Import Changes:**
Your import is correct - the v3 SDK uses `from langfuse.langchain import CallbackHandler`.

**Trace ID Retrieval:**
In v3, you cannot use `langfuse_sdk.get_current_trace_id()` directly after a LangChain invocation. Instead, you should access `langfuse_handler.last_trace_id` immediately after the chain invocation:

```python
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

response = chain.invoke(
    input={...},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": user_name
        }
    }
)

trace_id = langfuse_handler.last_trace_id  # Access trace ID from handler
```

**Important Note on last_trace_id:**
The documentation warns that "care is needed in concurrent environments where handler is reused". In Django REST Framework, each request should create a new CallbackHandler instance to avoid conflicts.

### Recommended Approach for Django

Given your constraint about not using context managers, here's the callback-only approach:

```python
from langfuse import get_client
from langfuse.langchain import CallbackHandler

def reconciliation(request):
    # Create a new handler instance per request
    langfuse_handler = CallbackHandler()

    chain = build_chain("get_python", tenant_id)

    response = chain.invoke(
        input={
            "source_csv": source_csv,
            "target_csv": target_csv,
            "metadata_csv": metadata_csv
        },
        config={
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_user_id": user_name,
                "langfuse_session_id": "session_456",  # Optional
                "langfuse_tags": ["django-request"]  # Optional
            }
        }
    )

    # Get trace ID from the handler
    trace_id = langfuse_handler.last_trace_id

    # Flush in short-lived environments
    langfuse = get_client()
    langfuse.flush()

    return JsonResponse({"trace_id": trace_id, "response": response})
```

### Addressing Your Specific Questions

**1. OpenTelemetry Setup:**
The v3 SDK is built on OpenTelemetry, but for the LangChain callback-based integration, no additional OTEL setup is required in your application code. The CallbackHandler handles this internally.

**2. Self-Hosted Compatibility:**
The documentation doesn't indicate known issues with self-hosted Langfuse and v3. Ensure your base_url parameter is correct (you've already changed from host to base_url, which is correct).

**3. Short-Lived Django Requests:**
The key changes are:
- Use `langfuse_handler.last_trace_id` to retrieve the trace ID
- Call `get_client().flush()` before returning the response to ensure events are sent
- Create a new CallbackHandler instance per request to avoid concurrency issues

**4. Callback Handler Compatibility:**
Yes, the callback handler approach is compatible with Django's request-response cycle. The main requirement is calling `flush()` before the response is returned.

**5. Celery Extension:**
For Celery tasks, you'll need to ensure `flush()` is called at the end of each task. The same callback-based pattern should work, but each Celery task should instantiate its own CallbackHandler.

### Troubleshooting Checklist

If traces still don't appear:
- Verify initialization: Ensure environment variables are set correctly or the client is initialized with the correct parameters
- Check network connectivity: Verify your Django application can reach your self-hosted Langfuse instance
- Confirm flush timing: Make sure `flush()` is called before the Django response is returned
- Handler reuse: Ensure you're creating a new CallbackHandler instance for each request, not reusing a global instance
- Check logs: Look for any SDK warnings or errors that might indicate connection issues

The critical difference from v2 is that trace ID retrieval now happens via `langfuse_handler.last_trace_id` rather than through the main client instance.

## Reply 2: Langfuse Team

Hi @javin-goyal, can you enable the debug mode of the SDK and share the logs? https://langfuse.com/docs/observability/sdk/advanced-features#logging--debugging
