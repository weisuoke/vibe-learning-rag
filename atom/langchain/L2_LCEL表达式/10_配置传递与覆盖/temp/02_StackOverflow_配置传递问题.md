---
source: https://stackoverflow.com/questions/79640638/how-to-pass-runnableconfig-as-arg-when-call-langchain-invoke-method
title: how to pass RunnableConfig as arg when call langchain invoke method
fetched_at: 2026-02-20 18:26:00 PST
---

# how to pass RunnableConfig as arg when call langchain invoke method

**Asked** 8 months ago
**Modified** 7 months ago
**Viewed** 804 times

## Question

I'm trying to make a RAG application using langchain and streamlit,I trying to manage the chathistory within this app,but I came across an unexpeted issue, I will explain :

```python
from langchain.schema.runnable import RunnableConfig
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler

def create_full_chain(retriever, groq_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model("Groq", groq_api_key=groq_api_key)
    # model = get_model()
    system_prompt = """my own promt.

    Context: {context}

    Question: """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever,rag_prompt=prompt) #add by youssef rag_prompt=prompt
    chain = create_memory_chain(model, rag_chain, chat_memory)
    return chain

def ask_question(chain, query,config=None):
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"},},
        config=config
    )
    return response
```

here's i create a runnableConfig :

```python
run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
            callbacks=[run_collector],
            tags=["Streamlit Chat"],
        )
```

now when the user asque a question this methode execute:

```python
response = ask_question(chain, prompt,config=runnable_config)
```

now my issue is, I have to pass to the `invoke` method two args with the same name which is config, because this one `config={"configurable": {"session_id": "foo"}` ,},is managed by the memory(chathistory (RunnableWithMessageHistory)) and and to be able to pass `runnable_config` i have to pass it like this `config=runnable_config` inside the invoke methode, Does anyone please have any idea how to resolve this issue please ?
thnaks

**Comment** (May 28, 2025 at 21:39):
RunnableConfig is just a dictionary, so merge it with the other dictionary.

## Answer (Score: 0)

**Answered** Jul 21, 2025 at 7:59

User's custom config can be injected to overall RunnableConfig:

```python
from typing import TypedDict

class UserConfig(TypedDict):
    user_id: str

user_config = UserConfig(user_id = "user-123")
config: RunnableConfig = {
    "configurable": {
        "thread_id": "thread-123",
        **user_config
    }
}
```
