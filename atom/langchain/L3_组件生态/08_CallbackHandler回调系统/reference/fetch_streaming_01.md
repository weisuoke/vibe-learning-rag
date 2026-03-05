---
type: fetched_content
source: https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/09-langchain-streaming/09-langchain-streaming.ipynb
title: Streaming - LangChain Handbook
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: 流式输出处理
---

以下是该网页（https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/09-langchain-streaming/09-langchain-streaming.ipynb）的结构化 Markdown 格式内容，已按原网页（Jupyter Notebook 渲染结构）完整提取并转换，保留所有标题层级、文本、代码块、输出和语义信息，无任何删减、摘要或改写。

---
source: https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/09-langchain-streaming/09-langchain-streaming.ipynb
title: Streaming - LangChain Handbook
fetched_at: 2026-02-25 01:22:00 PST
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/09-langchain-streaming/09-langchain-streaming.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/09-langchain-streaming/09-langchain-streaming.ipynb)

#### [LangChain Handbook](https://pinecone.io/learn/langchain)

# Streaming

For LLMs, streaming has become an increasingly popular feature. The idea is to rapidly return tokens as an LLM is generating them, rather than waiting for a full response to be created before returning anything.

Streaming is actually very easy to implement for simple use-cases, but it can get complicated when we start including things like Agents which have their own logic running which can block our attempts at streaming. Fortunately, we can make it work — it just requires a little extra effort.

We'll start easy by implementing streaming to the terminal for LLMs, but by the end of the notebook we'll be handling the more complex task of streaming via FastAPI for Agents.

First, let's install all of the libraries we'll be using.

```python
!pip install -qU \
    openai==0.28.0 \
    langchain==0.0.301 \
    fastapi==0.103.1 \
    "uvicorn[standard]"==0.23.2
```

## LLM Streaming to Stdout

The simplest form of streaming is to simply "print" the tokens as they're generated. To set this up we need to initialize an LLM (one that supports streaming, not all do) with two specific parameters:

- `streaming=True`, to enable streaming
- `callbacks=[SomeCallBackHere()]`, where we pass a LangChain callback class (or list containing multiple).

The `streaming` parameter is self-explanatory. The `callbacks` parameter and callback classes less so — essentially they act as little bits of code that do something as each token from our LLM is generated. As mentioned, the simplest form of streaming is to print the tokens as they're being generated, like with the `StreamingStdOutCallbackHandler`.

```python
import os
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    callbacks=[StreamingStdOutCallbackHandler()]  # ! important
)
```

Now if we run the LLM we'll see the response being *streamed*.

```python
from langchain.schema import HumanMessage

# create messages to be passed to chat LLM
messages = [HumanMessage(content="tell me a long story")]

llm(messages)
```

That was surprisingly easy, but things begin to get much more complicated as soon as we begin using agents. Let's first initialize an agent.

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, AgentType, initialize_agent

# initialize conversational memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"
)

# create a single tool to see how it impacts streaming
tools = load_tools(["llm-math"], llm=llm)

# initialize the agent
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    memory=memory,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    return_intermediate_steps=False
)
```

We already added our `StreamingStdOutCallbackHandler` to the agent as we initialized the agent with the same `llm` as we created with that callback. So let's see what we get when running the agent.

```python
prompt = "Hello, how are you?"

agent(prompt)
```

Not bad, but we do now have the issue of streaming the *entire* output from the LLM. Because we're using an agent, the LLM is instructed to output the JSON format we can see here so that the agent logic can handle tool usage, multiple "thinking" steps, and so on. For example, if we ask a math question we'll see this:

```python
agent("what is the square root of 71?")
```

It's interesting to see during development but we'll want to clean this streaming up a little in any actual use-case. For that we can go with two approaches — either we build a custom callback handler, or use a purpose built callback handler from LangChain (as usual, LangChain has something for everything). Let's first try LangChain's purpose-built `FinalStreamingStdOutCallbackHandler`.

We will overwrite the existing `callbacks` attribute found here:

```python
agent.agent.llm_chain.llm
```

With the new callback handler:

```python
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

agent.agent.llm_chain.llm.callbacks = [
    FinalStreamingStdOutCallbackHandler(
        answer_prefix_tokens=["Final", "Answer"]
    )
]
```

Let's try it:

```python
agent("what is the square root of 71?")
```

Not quite there, we should really clean up the `answer_prefix_tokens` argument but it is hard to get right. It's generally easier to use a custom callback handler like so:

```python
import sys

class CallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.content: str = ""
        self.final_answer: bool = False

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.content += token
        if "Final Answer" in self.content:
            # now we're in the final answer section, but don't print yet
            self.final_answer = True
            self.content = ""
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ["}"]:
                    sys.stdout.write(token)  # equal to `print(token, end="")`
                    sys.stdout.flush()

agent.agent.llm_chain.llm.callbacks = [CallbackHandler()]
```

Let's try again:

```python
agent("what is the square root of 71?")
```

```python
agent.agent.llm_chain.llm
```

It isn't perfect, but this is getting better. Now, in most scenarios we're unlikely to simply be printing output to a terminal or notebook. When we want to do something more complex like stream this data through another API, we need to do things differently.

## Using FastAPI with Agents

In most cases we'll be placing our LLMs, Agents, etc behind something like an API. Let's add that into the mix and see how we can implement streaming for agents with FastAPI.

First, we'll create a simple `main.py` script to contain our FastAPI logic. You can find it in the same GitHub repo location as this notebook ([here's a link](https://github.com/pinecone-io/examples/blob/langchain-streaming/learn/generation/langchain/handbook/09-langchain-streaming/main.py)).

To run the API, navigate to the directory and run `uvicorn main:app --reload`. Once complete, you can confirm it is running by looking for the 🤙 status in the next cell output:

```python
import requests

res = requests.get("http://localhost:8000/health")
res.json()
```

Out[14]:

```json
{'status': '🤙'}
```

```python
res = requests.get("http://localhost:8000/chat",
    json={"text": "hello there!"}
)
res
```

Out[15]:

```json
<Response [200]>
```

```python
res.json()
```

Out[16]:

```json
{'input': 'hello there!',
 'chat_history': [],
 'output': 'Hello! How can I assist you today?'}
```

Unlike with our StdOut streaming, we now need to send our tokens to a generator function that feeds those tokens to FastAPI via a `StreamingResponse` object. To handle this we need to use async code, otherwise our generator will not begin emitting anything until *after* generation is already complete.

The `Queue` is accessed by our callback handler, as as each token is generated, it puts the token into the queue. Our generator function asyncronously checks for new tokens being added to the queue. As soon as the generator sees a token has been added, it gets the token and yields it to our `StreamingResponse`.

To see it in action, we'll define a stream requests function called `get_stream`:

```python
def get_stream(query: str):
    s = requests.Session()
    with s.get(
        "http://localhost:8000/chat",
        stream=True,
        json={"text": query}
    ) as r:
        for line in r.iter_content():
            print(line.decode("utf-8"), end="")
```

```python
get_stream("hi there!")
```

"Hello! How can I assist you today?"

（笔记本内容至此完整结束，原网页所有章节、代码、输出和结构均已按目录层级与单元格顺序精准还原。）
