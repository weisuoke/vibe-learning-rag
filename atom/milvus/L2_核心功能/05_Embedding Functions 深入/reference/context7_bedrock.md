### Configure Titan Text Embeddings V2 with Dimensions and Normalization (Java)

Source: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModelWithResponseStream_TitanTextEmbeddings_section

This Java code snippet shows how to invoke Amazon Titan Text Embeddings V2 with specific inference parameters: the number of dimensions for the output embeddings and a flag to enable normalization. It requires the AWS SDK for Java and Bedrock Runtime client. The function returns a JSONObject containing the embedding and input token count.

```java
import org.json.JSONObject;

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;
import software.amazon.awssdk.core.SdkBytes;


    /**
     * Invoke Amazon Titan Text Embeddings V2 with additional inference parameters.
     *
     * @param inputText  - The text to convert to an embedding.
     * @param dimensions - The number of dimensions the output embeddings should have.
     *                   Values accepted by the model: 256, 512, 1024.
     * @param normalize  - A flag indicating whether or not to normalize the output embeddings.
     * @return The {@link JSONObject} representing the model's response.
     */
    public static JSONObject invokeModel(String inputText, int dimensions, boolean normalize) {

        // Create a Bedrock Runtime client in the AWS Region of your choice.
        var client = BedrockRuntimeClient.builder()
                .region(Region.US_WEST_2)
                .build();

        // Set the model ID, e.g., Titan Embed Text v2.0.
        var modelId = "amazon.titan-embed-text-v2:0";

        // Create the request for the model.
        var nativeRequest = """
                {
                    \"inputText\": \"%s\",
                    \"dimensions\": %d,
                    \"normalize\": %b
                }
                """.formatted(inputText, dimensions, normalize);

        // Encode and send the request.
        var response = client.invokeModel(request -> {
            request.body(SdkBytes.fromUtf8String(nativeRequest));
            request.modelId(modelId);
        });

        // Decode the model's response.
        var modelResponse = new JSONObject(response.body().asUtf8String());

        // Extract and print the generated embedding and the input text token count.
        var embedding = modelResponse.getJSONArray("embedding");
        var inputTokenCount = modelResponse.getBigInteger("inputTextTokenCount");
        System.out.println("Embedding: " + embedding);
        System.out.println("\nInput token count: " + inputTokenCount);

        // Return the model's native response.
        return modelResponse;
    }


```

--------------------------------

### InvokeModel V2 Embeddings

Source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text

Generates embeddings using the V2 model. Supports optional normalization, dimensions, and embedding types.

```APIDOC
## POST /invoke/model (V2 Embeddings)

### Description
Generates an embedding vector for the provided input text using the V2 model. Supports optional parameters for normalization, embedding dimensions, and embedding types.

### Method
POST

### Endpoint
/invoke/model

### Parameters
#### Request Body
- **inputText** (string) - Required - The text to convert into an embedding.
- **normalize** (boolean) - Optional - Flag indicating whether or not to normalize the output embedding. Defaults to true.
- **dimensions** (int) - Optional - The number of dimensions the output embedding should have. Accepted values: 1024 (default), 512, 256.
- **embeddingTypes** (list) - Optional - Accepts a list containing "float", "binary", or both. Defaults to `"float"`.

### Request Example
```json
{
    "inputText": "Sample text for embedding",
    "dimensions": 512,
    "normalize": false,
    "embeddingTypes": ["float", "binary"]
}
```

### Response
#### Success Response (200)
- **embedding** (array of float) - An array representing the embedding vector of the input.
- **inputTextTokenCount** (int) - The number of tokens in the input.
- **embeddingsByType** (object) - A dictionary containing embeddings categorized by type (e.g., "float", "binary"). This field always appears.
  - **float** (array of float) - The float embedding vector.
  - **binary** (array of int) - The binary embedding vector (if requested).

#### Response Example
```json
{
    "embedding": [0.123, -0.456, ...],
    "inputTextTokenCount": 10,
    "embeddingsByType": {
        "float": [0.123, -0.456, ...],
        "binary": [101, 010, ...]
    }
}
```
```

--------------------------------

### POST /invokeModel with Parameters (Java SDK)

Source: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModelWithResponseStream_TitanTextEmbeddings_section

This Java code example demonstrates how to invoke the Titan Text Embeddings V2 model with additional inference parameters such as `dimensions` and `normalize`. It shows how to format a more complex request body and process the response.

```APIDOC
## POST /invokeModel with Parameters (Java SDK)

### Description
Generates an embedding for the provided input text, allowing configuration of the output embedding dimensions and whether to normalize the embeddings.

### Method
POST

### Endpoint
`/invokeModel` (within the Bedrock Runtime client)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **inputText** (string) - Required - The text to convert into an embedding.
- **dimensions** (integer) - Optional - The number of dimensions for the output embeddings (e.g., 256, 512, 1024).
- **normalize** (boolean) - Optional - A flag indicating whether to normalize the output embeddings.

### Request Example
```java
// ... (client setup and model ID definition)
var inputText = "Example text for embedding.";
int dimensions = 512;
boolean normalize = true;

var nativeRequest = "{ \"inputText\": \"%s\", \"dimensions\": %d, \"normalize\": %b }" \
    .formatted(inputText, dimensions, normalize);

client.invokeModel(request -> {
    request.body(SdkBytes.fromUtf8String(nativeRequest));
    request.modelId(modelId);
});
```

### Response
#### Success Response (200)
- **embedding** (array of numbers) - The generated embedding vector.
- **inputTextTokenCount** (integer) - The number of tokens in the input text.

#### Response Example
```json
{
  "embedding": [
    0.9876,
    0.1111,
    // ... more dimensions
  ],
  "inputTextTokenCount": 8
}
```
```

Source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text

Request and response: The inputText parameter is required. The normalize and dimensions parameters are optional.
  * inputText – Enter text to convert to an embedding.
  * normalize – (optional) Flag indicating whether or not to normalize the output embedding. Defaults to true.
  * dimensions – (optional) The number of dimensions the output embedding should have. The following values are accepted: 1024 (default), 512, 256.

--------------------------------

Source: https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-setup

Unable to save cookie preferences > Note: The number of dimensions in each vector must match the vector dimensions in the embeddings model. Refer to the following table to determine how many dimensions the vector should contain:
Model | Dimensions  
---|---  
Titan G1 Embeddings - Text | 1,536  
Titan V2 Embeddings - Text | 1,024, 512, and 256  
Cohere Embed English | 1,024  
Cohere Embed Multilingual | 1,024  
    5. Leave all other settings to their default and create the graph.
  2. Once the graph is created, click it to take note of the **Resource ARN** and **Vector dimensions** for when you create the knowledge base. When choosing the embeddings model in Amazon Bedrock, make sure that you choose a model with the same dimensions as the **Vector dimensions** you configured on your Neptune Analytics graph.