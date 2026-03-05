---
source: https://milvus.io/docs/query.md
title: Query | Milvus Documentation
fetched_at: 2026-02-21
---

# Query

In addition to ANN searches, Milvus also supports metadata filtering through queries. This page introduces how to use **Query**, **Get**, and **QueryIterator** to perform metadata filtering.

If you dynamically add new fields after the collection has been created, queries that include these fields will return the defined default values or **NULL** for entities that have not explicitly set values. For details, refer to [Add Fields to an Existing Collection](/docs/add-fields-to-an-existing-collection.md).

## Overview

A Collection can store various types of scalar fields. You can have Milvus filter Entities based on one or more scalar fields. Milvus offers three types of queries: **Query**, **Get**, and **QueryIterator**. The table below compares these three query types.

|          | Get                                      | Query                                      | QueryIterator                                      |
|----------|------------------------------------------|--------------------------------------------|---------------------------------------------------|
| Applicable scenarios | To find entities that hold the specified primary keys. | To find all or a specified number of entities that meet the custom filtering conditions | To find all entities that meet the custom filtering conditions in paginated queries. |
| Filtering method     | By primary keys                          | By filtering expressions.                  | By filtering expressions.                          |
| Mandatory parameters | Collection name<br>Primary keys          | Collection name<br>Filtering expressions   | Collection name<br>Filtering expressions<br>Number of entities to return per query |
| Optional parameters  | Partition name<br>Output fields          | Partition name<br>Number of entities to return<br>Output fields | Partition name<br>Number of entities to return in total<br>Output fields |
| Returns              | Returns entities that hold the specified primary keys in the specified collection or partition. | Returns all or a specified number of entities that meet the custom filtering conditions in the specified collection or partition. | Returns all entities that meet the custom filtering conditions in the specified collection or partition through paginated queries. |

For more on metadata filtering, refer to [Filtering Explained](/docs/boolean.md).

## Use Get

When you need to find entities by their primary keys, you can use the **Get** method. The following code examples assume that there are three fields named `id`, `vector`, and `color` in your collection.

Example data (JSON format):

```json
[
    {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "color": "pink_8682"},
    {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], "color": "red_7025"},
    {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592], "color": "orange_6781"},
    {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345], "color": "pink_9298"},
    {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184, 0.30337481143159106], "color": "red_4794"},
    {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383, -0.1446277761879955], "color": "yellow_4222"},
    {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192, -0.8984947637863987], "color": "red_9392"},
    {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709, 0.5378064918413052], "color": "grey_8510"},
    {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872, -0.6140360785406336], "color": "white_9381"},
    {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717, -0.6980531615588608], "color": "purple_4976"}
]
```

### Python

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

res = client.get(
    collection_name="my_collection",
    ids=[0, 1, 2],
    output_fields=["vector", "color"]
)

print(res)
```

### Java

```java
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.vector.request.GetReq;
import io.milvus.v2.service.vector.response.GetResp;
// ... other imports

MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
        .uri("http://localhost:19530")
        .token("root:Milvus")
        .build());

GetReq getReq = GetReq.builder()
        .collectionName("my_collection")
        .ids(Arrays.asList(0, 1, 2))
        .outputFields(Arrays.asList("vector", "color"))
        .build();

GetResp getResp = client.get(getReq);

List<QueryResp.QueryResult> results = getResp.getGetResults();
for (QueryResp.QueryResult result : results) {
    System.out.println(result.getEntity());
}
```

## Use Query

When you need to find entities by custom filtering conditions, use the **Query** method. The following code examples assume there are three fields named `id`, `vector`, and `color` and return the specified number of entities that hold a `color` value starting with `red`.

### Python Example

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

res = client.query(
    collection_name="my_collection",
    filter="color like \"red%\"",
    output_fields=["vector", "color"],
    limit=3
)

print(res)
```
