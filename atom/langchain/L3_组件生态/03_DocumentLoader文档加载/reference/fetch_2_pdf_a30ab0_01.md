---
type: fetched_content
source: https://github.com/timerring/rag101
title: GitHub - timerring/rag101: LangChain and RAG best practices
fetched_at: 2026-02-24T15:17:49.559292+00:00
status: success
author: 
published_date: 
knowledge_point: 03_DocumentLoader文档加载
content_type: code
fetch_tool: Grok-mcp___web_fetch
priority: high
word_count: 1890
knowledge_point_tag: 实战代码_场景2_PDF文档加载
---

---
source: https://github.com/timerring/rag101
title: GitHub - timerring/rag101: LangChain and RAG best practices
fetched_at: 2026-02-24 07:14 AM PST
---

# GitHub - timerring/rag101: LangChain and RAG best practices

**timerring / rag101**  
**LangChain and RAG best practices**

[Branches](https://github.com/timerring/rag101/branches)  
[Tags](https://github.com/timerring/rag101/tags)  

**Go to file**  
**Code** (当前选中)

## Folders and files

| Name | Latest commit | History |
|------|---------------|---------|
| [conversational_retrieval_chain](/timerring/rag101/tree/main/conversational_retrieval_chain) |  |  |
| [embeddings](/timerring/rag101/tree/main/embeddings) |  |  |
| [loader](/timerring/rag101/tree/main/loader) |  |  |
| [question_answering](/timerring/rag101/tree/main/question_answering) |  |  |
| [retrieval](/timerring/rag101/tree/main/retrieval) |  |  |
| [splitter](/timerring/rag101/tree/main/splitter) |  |  |
| [vectorstores](/timerring/rag101/tree/main/vectorstores) |  |  |
| [LICENSE](/timerring/rag101/blob/main/LICENSE) |  |  |
| [README.md](/timerring/rag101/blob/main/README.md) |  |  |
| [requirements.txt](/timerring/rag101/blob/main/requirements.txt) |  |  |

**View all files**

## Repository files navigation

# LangChain and RAG best practices

## Introduction

This is a quick start guide essay for LangChain and RAG which mainly refers to the [Langchain chat with your data](https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/lesson/snupv/introduction?courseName=langchain-chat-with-your-data) course.

You can check the entire code in the [rag101 repository](https://github.com/timerring/rag101/).

### LangChain

LangChain is an Open-source developer framework for building LLM applications.

It components are as below:

#### Prompt

- Prompt Templates: used for generating model input.
- Output Parsers: implementations for processing generated results.
- Example Selectors: selecting appropriate input examples.

#### Models

- LLMs
- Chat Models
- Text Embedding Models

#### Indexes

- Document Loaders
- Text Splitters
- Vector Stores
- Retrievers

#### Chains

- Can be used as a building block for other chains.
- Provides over 20 types of application-specific chains.

#### Agents

- Supports 5 types of agents to help language models use external tools.
- Agent Toolkits: provides over 10 implementations, agents execute tasks through specific tools.

### RAG process

The whole RAG process lays on the Vector Store Loading and Retrieval-Augmented Generation.

#### Vector Store Loading

Load the data from different sources, split and convert them into vector embeddings.

#### Retrieval-Augmented Generation

1) After the user's input **Query**, the system will retrieve the most relevant document fragments (Relevant Splits) from the vector store.  
2) The retrieved relevant fragments will be combined into a **Prompt**, which will be passed along with the context to the large language model (LLM).  
3) Finally, the language model will generate an answer based on the retrieved fragments and return it to the user.

## Loaders

You can use loaders to deal with different kind and format of data.

Some are public and some are proprietary. Some are structured and some are not.

Some useful lib:

- pdf: pypdf  
- youtube audio: yt_dlp pydub  
- web page: beautifulsoup4  

For more loaders, you can check the [official docs](https://python.langchain.com/api_reference/community/document_loaders.html#module-langchain_community.document_loaders).

You can check the entire code [here](https://github.com/timerring/rag101/tree/main/loader).

### PDF

Now, we can practice:

First, install the lib:

```bash
pip install langchain-community 
pip install pypdf
```

You can check the demo in the

```python
from langchain.document_loaders import PyPDFLoader

# In fact, the langchain calls the pypdf lib to load the pdf file
loader = PyPDFLoader("ProbRandProc_Notes2004_JWBerkeley.pdf")
pages = loader.load()

print(type(pages))
# <class 'list'>
print(len(pages))
# Print the total num of pages

# Using the first page as an example
page = pages[0]
print(type(page))
# <class 'langchain_core.documents.base.Document'>

# What is inside the page:
# 1. page_content
# 2. meta_data: the description of the page

print(page.page_content[0:500])
print(page.metadata)
```

### Web Base Loader

Also we install the lib first:

```bash
pip install beautifulsoup4
```

The WebBaseLoader is based on the beautifulsoup4 lib.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://zh.d2l.ai/")
pages = loader.load()
print(pages[0].page_content[:500])

# You can also use json as the post processing
# import json
# convert_to_json = json.loads(pages[0].page_content)
```

## Splitters

Splitting Documents into smaller chunks. Retaining the meaningful relationships.

### Why split?

- The limitation of GPU: the GPT model with more than 1B parameters. The forward propagation cannot process such a large parameters. So the split is necessary.
- More efficient computation.
- Some fixed size of sequence.
- Better generalization.

> However, the split points may lose some information. So we split should consider the semantic.

### Type of splitters

- CharacterTextSplitter
- MarkdownHeaderTextSplitter
- TokenTextsplitter
- SentenceTransformersTokenTextSplitter
- **RecursiveCharacterTextSplitter** : Recursively tries to split by different characters to find one that works.
- Language: for CPP, Python, Ruby, Markdown etc
- NLTKTextSplitter: sentences using NLTK(Natural Language Tool Kit)
- SpacyTextSplitter: sentences using Spacy

For more, check the [docs](https://python.langchain.com/api_reference/text_splitters/index.html#module-langchain_text_splitters).

### Example CharacterTextSplitter and RecursiveCharacterTextSplitter

You can check the entire code [here](https://github.com/timerring/rag101/blob/main/splitter/text_splitter.py).

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

example_text = """When writing documents, writers will use document structure to group content. This can convey to the reader, which idea's are related. For example, closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. 

  Paragraphs are often delimited with a carriage return or two carriage returns. Carriage returns are the "backslash n" you see embedded in this string. Sentences have a period at the end, but also, have a space.and words are separated by space."""

c_splitter = CharacterTextSplitter(
    chunk_size=450, # the size of the chunk
    chunk_overlap=0, # the overlap of the chunk, which can be shared with the previous chunk
    separator = ' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["

", "
", " ", ""] # priority of the separators
)

print(c_splitter.split_text(example_text))
# split at 450 characters
print(r_splitter.split_text(example_text))
# split at first 


```

## Vectorstores and Embeddings

Review the RAG process:

Benefits:

1) Improve the accuracy of the query. When query the similar chunks, the accuracy will be higher.
2) Improve the efficiency of the query. Minimize the computation when query the similar chunks.
3) Improve the coverage of the query. The chunks can cover every point of the document.
4) Improve the Embeddings.

### Embeddings

If two sentences have similar meanings, then they will be closer in the high-dimensional semantic space.

### Vector Stores

Store every chunk in a vector store. When customer query, the query will be embedded and then find the most similar vectors which means the index of these chunks, and then return the chunks.

### Practice

#### Embeddings

You can check the entire code [here](https://github.com/timerring/rag101/blob/main/embeddings/zhipu.py).

First, install the lib:

The `chromadb` is a lightweight vector database.

```bash
pip install chromadb
```

What we need is a good embedding model, you can select what you like. Refer to the [docs](https://python.langchain.com/api_reference/community/embeddings.html#module-langchain_community.embeddings).

Here I use the `ZhipuAIEmbeddings`. So you should install the lib:

```bash
pip install zhipuai
```

Here is the test code:

```python
from langchain_community.embeddings import ZhipuAIEmbeddings

embed = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key="Entry your own api key"
)

input_texts = ["This is a test query1.", "This is a test query2."]
print(embed.embed_documents(input_texts))
```

#### Vector Stores

You can check the entire code [here](https://github.com/timerring/rag101/blob/main/vectorstores/chroma.py).

```bash
pip install langchain-chroma
```

Then we can use the `Chroma` to store the embeddings.

```python
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings

# load the web page
loader = WebBaseLoader("https://en.d2l.ai/")
docs = loader.load()

# split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs)
# print(len(splits))

# set the embeddings models
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key="your own api key"
)

# set the persist directory
persist_directory = r'.'

# create the vector database
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)
# print(vectordb._collection.count())

# query the vector database
question = "Recurrent"
docs = vectordb.similarity_search(question, k=3)
# print(len(docs))
print(docs[0].page_content)
```

Then you can find the `chorma.sqlite3` file in the specific directory.

## Retrieval

This part is the core part of the RAG.

Last part we have already used the `similarity_search` method. On top of that, we also have other methods.

- Basic semantic similarity
- Maximum Marginal Relevance(MMR)
- Metadata
- LLM Aided Retrieval

### Similarity Search

Similarity Search calculates the similarity between the query vector and all document vectors in the database to find the most relevant document.

The similarity measurement methods include **cosine similarity** and **Euclidean distance**, which can effectively measure the closeness of two vectors in a high-dimensional space.

However, relying solely on similarity search may result in insufficient diversity, as it only focuses on the match between the query and the content, ignoring the differences between different pieces of information. In some applications, especially when it is necessary to cover **multiple different aspects of information**, the extended method of Maximum Marginal Relevance (MMR) can better balance relevance and diversity.

#### Practice

The practice part is on the pervious part.

### Maximum Marginal Relevance (MMR)

Retrieving only the most relevant documents may overlook the diversity of information. For example, if only the most similar response is selected, the **results may be very similar or even contain duplicate content**. The core idea of MMR is to balance relevance and diversity, that is, to select the information most relevant to the query while ensuring that the information is diverse in content. **By reducing the repetition of information between different pieces**, MMR can provide a more comprehensive and diverse set of results.

The process of MMR is as follows:

1) Query the Vector Store: First convert the query into vectors using the embedding model.  
2) Choose the fetch_k most similar responses. Find the top k most similar vectors from the vector store.  
3) Within those responses choose the k most diverse. By calculating the similarity between each response, MMR will prefer results that are **more different from each other**, thus increasing the coverage of information. This process ensures that the returned results are not only "most similar", but also "complementary".

The key parameter is the `lambda` which is the weight of the relevance vs diversity.

```python
docs_mmr = vectordb.max_marginal_relevance_search("How the neural network works?", fetch_k=8, k=2)
print(docs_mmr[0].page_content[:100])
```

### Metadata Filtering

```python
new_loader = WebBaseLoader("https://www.deeplearning.ai/")
new_docs = new_loader.load()
new_splits = text_splitter.split_documents(new_docs)
vectordb.add_documents(new_splits)

docs_meta = vectordb.similarity_search("how the neural network works?", k=1, filter={"source": "https://www.deeplearning.ai/"})
print(docs_meta[0].page_content[:100])
```

### LLM Aided Retrieval (`SelfQueryRetriever`)

Uses LLM to extract search terms and filters from queries.

```python
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

llm = OpenAI(temperature=0)
metadata_field_info = [
    AttributeInfo(name="source", description="Source of the chunk", type="string"),
    AttributeInfo(name="page", description="Page number", type="integer"),
]
document_content_description = "lectures on retrieval augmentation generation"

retriever = SelfQueryRetriever.from_llm(
    llm, vectordb, document_content_description, metadata_field_info, verbose=True
)
```

### Compression

Reduces document size by extracting only relevant parts using an LLM.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(OpenAI(temperature=0))
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vectordb.as_retriever()
)

question = "What is the main topic of second lecture?"
compressed_docs = compression_retriever.get_relevant_documents(question)
```

## Question Answering

Combines retrieved documents with LLM to generate answers.

### RetrievalQA Chain

Improves accuracy, supports real-time updates, reduces model memory.

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

template = """
Please answer the question based on the following context.
If you don't know the answer, just say you don't know...
Context: {context}
Question: {question}
Helpful answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    chat,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

result = qa_chain({"query": "What is this book about?"})
print(result["result"])
```

### Chain Types
- **Map_reduce**: Parallel processing, fast.  
- **Refine**: Sequential refinement, high quality.  
- **Map_rerank**: Selects best answer by score.

## Conversational Retrieval Chain

Maintains chat history and context.

### Memory

```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
```

### Full Example

```python
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    chat, retriever=vectordb.as_retriever(), memory=memory
)

question = "What is the main topic of this book?"
result = qa.invoke({"question": question})
print(result['answer'])

question = "Can you tell me more about it?"
result = qa.invoke({"question": question})
print(result['answer'])
```

## Resources

- [Blog Post](https://blog.timerring.com/posts/langchain-and-rag-best-practices/)
- License: MIT
- Language: Python (100%)

## 内容摘要
(待后续人工精炼；此处保留抓取原文为主)

---

## 关键信息提取

### 技术要点
- (待补充)

### 代码示例
- (待补充)

### 相关链接
- (待补充)

---

## 抓取质量评估
- 完整性: 完整
- 可用性: 中
- 时效性: (待判定)
