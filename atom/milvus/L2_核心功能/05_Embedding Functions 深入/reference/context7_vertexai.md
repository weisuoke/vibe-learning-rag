### Predict Text Embeddings with Vertex AI in Java

Source: https://cloud.google.com/vertex-ai/docs/samples/aiplatform-sdk-embedding

This Java code snippet demonstrates how to get text embeddings from a pretrained, foundational model in Vertex AI. It requires the Vertex AI Java client library and proper authentication setup. The function takes endpoint, project, model, texts, task type, and optional output dimensionality as input and returns a list of float embeddings.

```java
import static java.util.stream.Collectors.toList;

import com.google.cloud.aiplatform.v1.EndpointName;
import com.google.cloud.aiplatform.v1.PredictRequest;
import com.google.cloud.aiplatform.v1.PredictResponse;
import com.google.cloud.aiplatform.v1.PredictionServiceClient;
import com.google.cloud.aiplatform.v1.PredictionServiceSettings;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PredictTextEmbeddingsSample {
  public static void main(String[] args) throws IOException {
    // TODO(developer): Replace these variables before running the sample.
    // Details about text embedding request structure and supported models are available in:
    // https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings
    String endpoint = "us-central1-aiplatform.googleapis.com:443";
    String project = "YOUR_PROJECT_ID";
    String model = "gemini-embedding-001";
    predictTextEmbeddings(
        endpoint,
        project,
        model,
        List.of("banana bread?", "banana muffins?"),
        "QUESTION_ANSWERING",
        OptionalInt.of(3072));
  }

  // Gets text embeddings from a pretrained, foundational model.
  public static List<List<Float>> predictTextEmbeddings(
      String endpoint,
      String project,
      String model,
      List<String> texts,
      String task,
      OptionalInt outputDimensionality)
      throws IOException {
    PredictionServiceSettings settings = 
        PredictionServiceSettings.newBuilder().setEndpoint(endpoint).build();
    Matcher matcher = Pattern.compile("^(?<Location>\w+-\w+)").matcher(endpoint);
    String location = matcher.matches() ? matcher.group("Location") : "us-central1";
    EndpointName endpointName = 
        EndpointName.ofProjectLocationPublisherModelName(project, location, "google", model);

    List<List<Float>> floats = new ArrayList<>();
    // You can use this prediction service client for multiple requests.
    try (PredictionServiceClient client = PredictionServiceClient.create(settings)) {
      // gemini-embedding-001 takes one input at a time.
      for (int i = 0; i < texts.size(); i++) {
        PredictRequest.Builder request = 
            PredictRequest.newBuilder().setEndpoint(endpointName.toString());
        if (outputDimensionality.isPresent()) {
          request.setParameters(
              Value.newBuilder()
                  .setStructValue(
                      Struct.newBuilder()
                          .putFields(
                              "outputDimensionality", valueOf(outputDimensionality.getAsInt()))
                          .build()));
        }
        request.addInstances(
            Value.newBuilder()
                .setStructValue(
                    Struct.newBuilder()
                        .putFields("content", valueOf(texts.get(i)))
                        .putFields("task_type", valueOf(task))
                        .build()));
        PredictResponse response = client.predict(request.build());

        for (Value prediction : response.getPredictionsList()) {
          Value embeddings = prediction.getStructValue().getFieldsOrThrow("embeddings");
          Value values = embeddings.getStructValue().getFieldsOrThrow("values");
          floats.add(
              values.getListValue().getValuesList().stream()
                  .map(Value::getNumberValue)
                  .map(Double::floatValue)
                  .collect(toList()));
        }
      }
      return floats;
    }
  }

  private static Value valueOf(String s) {
    return Value.newBuilder().setStringValue(s).build();
  }

  private static Value valueOf(int n) {
    return Value.newBuilder().setNumberValue(n).build();
  }
}

```

--------------------------------

### Generate Text Embeddings with Vertex AI Node.js Client

Source: https://cloud.google.com/vertex-ai/docs/samples/aiplatform-sdk-embedding

This code snippet demonstrates how to generate embeddings for a given set of texts using the Vertex AI Node.js client library. It requires the `@google-cloud/aiplatform` package and proper authentication setup (Application Default Credentials). The function takes project ID, model name, texts, task type, dimensionality, and API endpoint as input, and outputs the generated embeddings.

```javascript
async function main(
  project,
  model = 'gemini-embedding-001',
  texts = 'banana bread?;banana muffins?',
  task = 'QUESTION_ANSWERING',
  dimensionality = 0,
  apiEndpoint = 'us-central1-aiplatform.googleapis.com'
) {
  const aiplatform = require('@google-cloud/aiplatform');
  const {PredictionServiceClient} = aiplatform;
  const {helpers} = aiplatform; // helps construct protobuf.Value objects.
  const clientOptions = {apiEndpoint: apiEndpoint};
  const location = 'us-central1';
  const endpoint = `projects/${project}/locations/${location}/publishers/google/models/${model}`;

  async function callPredict() {
    const instances = texts
      .split(';')
      .map(e => helpers.toValue({content: e, task_type: task}));

    const client = new PredictionServiceClient(clientOptions);
    const parameters = helpers.toValue(
      dimensionality > 0 ? {outputDimensionality: parseInt(dimensionality)} : {}
    );
    const allEmbeddings = []
    // gemini-embedding-001 takes one input at a time.
    for (const instance of instances) {
      const request = {endpoint, instances: [instance], parameters};
      const [response] = await client.predict(request);
      const predictions = response.predictions;

      const embeddings = predictions.map(p => {
        const embeddingsProto = p.structValue.fields.embeddings;
        const valuesProto = embeddingsProto.structValue.fields.values;
        return valuesProto.listValue.values.map(v => v.numberValue);
      });

      allEmbeddings.push(embeddings[0])
    }


    console.log('Got embeddings: \n' + JSON.stringify(allEmbeddings));
  }

  callPredict();
}

```

### Vertex AI > Generative AI > Getting Text Embeddings on Vertex AI

Source: https://cloud.google.com/vertex-ai/docs/tutorials/jupyter-notebooks

Learn how to get a text embedding given a text-embedding model and a text. Tutorial steps

--------------------------------

### Embeddings

Source: https://cloud.google.com/vertex-ai/docs/vector-search/try-it

**Text (question-answering):** Text semantic search on item names and descriptions, with improved search quality by task type QUESTION_ANSWERING. This is suited for Q&A types of applications, where the goal is to find answers to specific questions within the item data.

--------------------------------

### Deep dive resources > Related videos > New "task type" embedding from the DeepMind team improves RAG search quality

Source: https://cloud.google.com/vertex-ai/docs/vector-search/overview

The Google DeepMind team has developed new _task type_ embeddings that significantly improve the accuracy and relevance of Retrieval Augmented Generation (RAG) systems. This resource addresses common challenges in RAG search quality and explains how these task type embeddings effectively bridge the semantic gap between queries and answers, leading to better retrieval and enhanced RAG performance.