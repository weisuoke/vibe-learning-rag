### Generate Text Embeddings with Inference API (Ruby)

Source: https://context7.com/alchaplinsky/hugging-face/llms.txt

Creates vector representations of text using the Hugging Face Inference API. This is useful for semantic search, clustering, or similarity comparisons. It defaults to the 'sentence-transformers/all-MiniLM-L6-v2' model and returns an array of embedding vectors.

```ruby
client = HuggingFace::InferenceApi.new(api_token: ENV['HUGGING_FACE_API_TOKEN'])

# Generate embeddings for text comparison
result = client.embedding(
  input: ['How to build a ruby gem?', 'How to install ruby gem?']
)

# Response example - array of embedding vectors:
# [
#   [0.0599, 0.1089, 0.0346, ...],  # 384-dimensional vector for first text
#   [0.0612, 0.1102, 0.0298, ...]   # 384-dimensional vector for second text
# ]

# Use for semantic similarity
require 'matrix'
v1 = Vector.elements(result[0])
v2 = Vector.elements(result[1])
similarity = v1.inner_product(v2) / (v1.magnitude * v2.magnitude)
```

--------------------------------

### Hugging Face Inference API

Source: https://github.com/alchaplinsky/hugging-face/blob/main/README.md

The Hugging Face Inference API provides free access to machine learning models for prototyping. This section covers its usage for various tasks like question answering, text generation, summarization, embedding, and sentiment analysis.

```APIDOC
## Hugging Face Inference API

### Description
This API allows for free machine learning inference for prototyping purposes. It supports various tasks including question answering, text generation, summarization, embedding, and sentiment analysis.

### Method
POST (Implicitly through client methods)

### Endpoint
Not directly exposed; methods call the appropriate Hugging Face Inference API endpoints.

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
Parameters vary based on the task:
- **question** (string) - Required for question answering.
- **context** (string) - Required for question answering.
- **input** (string or array of strings) - Required for text generation, summarization, embedding, and sentiment analysis.

### Request Example
```ruby
require "hugging_face"
client = HuggingFace::InferenceApi.new(api_token: ENV['HUGGING_FACE_API_TOKEN'])

# Question Answering
client.question_answering(question: 'What is my name?', context: 'I am the only child. My father named his son John.')

# Text Generation
client.text_generation(input: 'Can you please let us know more details about your ')

# Summarization
client.summarization(input: 'The tower is 324 metres (1,063 ft) tall...')

# Embedding
client.embedding(input: ['How to build a ruby gem?', 'How to install ruby gem?'])

# Sentiment Analysis
client.sentiment(input: ['My life sucks', 'Life is a miracle'])
```

### Response
#### Success Response (200)
- **Response structure varies based on the ML task.**

#### Response Example
```json
// Example for sentiment analysis
[
  { "label": "NEGATIVE", "score": 0.999 },
  { "label": "POSITIVE", "score": 0.998 }
]
```
```

### InferenceApi#embedding - Generate Text Embeddings

Source: https://context7.com/alchaplinsky/hugging-face/llms.txt

Creates vector representations of text for semantic search, clustering, or similarity comparisons. Uses sentence-transformers/all-MiniLM-L6-v2 by default.

--------------------------------

### Summary

Source: https://context7.com/alchaplinsky/hugging-face/llms.txt

The hugging-face gem is ideal for Ruby applications requiring machine learning capabilities without managing ML infrastructure. Common use cases include building chatbots with question answering, content summarization for news aggregators or document processing, sentiment analysis for social media monitoring or customer feedback analysis, semantic search using text embeddings, and text generation for content creation tools. The Inference API is perfect for prototyping and low-volume applications, while the Endpoints API serves production workloads requiring dedicated resources.

Integration follows a simple pattern: initialize a client with your API token, call the appropriate method with your input data, and process the JSON response. All methods support custom models, allowing you to use any compatible model from the Hugging Face Hub. The gem handles authentication, request formatting, response parsing, and retry logic automatically, making it straightforward to add ML capabilities to any Ruby application with just a few lines of code.

--------------------------------

### Usage > Inference API

Source: https://github.com/alchaplinsky/hugging-face/blob/main/README.md

The inference API is a free Machine Learning API from Hugging Face. It is meant for prototyping and not production use, see below for Inference Endpoints, the product for use with production LLMs.