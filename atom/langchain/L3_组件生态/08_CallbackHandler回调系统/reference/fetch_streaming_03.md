---
type: fetched_content
source: https://www.reddit.com/r/LangChain/comments/19enjxr/streaming_local_llm_with_fastapi_llamacpp_and/
title: Streaming local LLM with FastAPI, Llama.cpp and Langchain
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: 流式输出处理
---

# Streaming local LLM with FastAPI, Llama.cpp and Langchain

**r/LangChain**

**Submitted 2 years ago**

## 帖子正文

Hi,

I have setup FastAPI with Llama.cpp and Langchain. Now I want to enable streaming in the FastAPI responses. Streaming works with Llama.cpp in my terminal, but I wasn't able to implement it with a FastAPI response. See this Stackoverflow-Question (for code etc.): https://stackoverflow.com/questions/77867894/streaming-local-llm-with-fastapi-llama-cpp-and-langchain?noredirect=1#comment137276485_77867894

Most tutorials focused on enabling streaming with an OpenAI model, but I am using a local LLM (quantized Mistral) with llama.cpp. I think I have to modify the Callbackhandler, but no tutorial worked.

Does anyone know how I can make Streaming working? I have a project deadline on Friday and unitl then I have to make it work...

## 评论区

### Comment 1 (3 points • 2 years ago)

This works for me (other examples here):

```python
"""
Demonstrates how to use the `ChatInterface` to create a chatbot using
[LangChain Expression Language](https://python.langchain.com/docs/expression_language/) (LCEL)
with streaming and memory.
"""

from operator import itemgetter

import panel as pn
from huggingface_hub import hf_hub_download
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms.llamacpp import LlamaCpp

pn.extension()

REPO_ID = "TheBloke/zephyr-7B-beta-GGUF"
FILENAME = "zephyr-7b-beta.Q5_K_M.gguf"
SYSTEM_PROMPT = "Try to be a silly comedian."


def load_llm(repo_id: str = REPO_ID, filename: str = FILENAME, **kwargs):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    llm = LlamaCpp(model_path=model_path, **kwargs)
    return llm


def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    message = ""
    inputs = {"input": contents}
    for token in chain.stream(inputs):
        message += token
        yield message
    memory.save_context(inputs, {"output": message})


model = load_llm(
    repo_id=REPO_ID,
    filename=FILENAME,
    streaming=True,
    n_gpu_layers=1,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
)
memory = ConversationSummaryBufferMemory(return_messages=True, llm=model)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
output_parser = StrOutputParser()
chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
    | output_parser
)

chat_interface = pn.chat.ChatInterface(
    pn.chat.ChatMessage(
        "Offer a topic and Mistral will try to be funny!", user="System"
    ),
    callback=callback,
    callback_user="Mistral",
)
chat_interface.servable()
```

The only issue with this is that the model seems to go off. I presume something is not right with the formatting.

### Comment 2 (OP Reply)

> from langchain_core.runnables import RunnableLambda, RunnablePassthrough

thanks! But this is not how it coulbe be done with FastAPI right?

### Comment 3 (2 points • 2 years ago)

You can adapt it, replacing panel chat with your desired API endpoints.

### Comment 4 (2 points • 2 years ago)

Did you have a look on this tutorial?

### Comment 5 (OP Reply)

yes already had a look on it but the streaming looked weird for me and did't work as expected.
