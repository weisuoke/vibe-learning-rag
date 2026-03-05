---
source: https://milvus.io/docs/insert-update-delete.md
title: Insert Entities | Milvus Documentation
fetched_at: 2026-02-21
---

# Insert Entities

Entities in a collection are data records that share the same set of fields. Field values in every data record form an entity. This page introduces how to insert entities into a collection.

If you dynamically add new fields after the collection has been created, and you do not specify values for these fields when inserting entities, Milvus automatically populates them with either their defined default values or **NULL** if defaults are not set. For details, refer to [Add Fields to an Existing Collection](/docs/manage-collections.md#add-fields-to-an-existing-collection).

**Fields added after collection creation**: If you add new fields to a collection after creation and don't specify values during insertion, Milvus automatically populates them with defined default values or **NULL** if no defaults are set. For details, refer to [Add Fields to an Existing Collection](/docs/manage-collections.md#add-fields-to-an-existing-collection).

**Duplicate handling**: The standard insert operation does not check for duplicate primary keys. Inserting data with an existing primary key creates a new entity with the same key, leading to data duplication and potential application issues. To update existing entities or avoid duplicates, use the **upsert** operation instead. For more information, refer to [Upsert Entities](/docs/upsert-entities.md).

## Overview

In Milvus, an **Entity** refers to data records in a **Collection** that share the same **Schema**, with the data in each field of a row constituting an Entity. Therefore, the Entities within the same Collection have the same attributes (such as field names, data types, and other constraints).

When inserting an Entity into a Collection, the Entity to be inserted can only be successfully added if it contains all the fields defined in the Schema. The inserted Entity will enter a Partition named **_default** in the order of insertion. Provided that a certain Partition exists, you can also insert Entities into that Partition by specifying the Partition name in the insertion request.

Milvus also supports dynamic fields to maintain the scalability of the Collection. When the dynamic field is enabled, you can insert fields that are not defined in the Schema into the Collection. These fields and values will be stored as key-value pairs in a reserved field named **$meta**. For more information about dynamic fields, please refer to [Dynamic Field](/docs/enable-dynamic-field.md).

## Insert Entities into a Collection

Before inserting data, you need to organize your data into a list of dictionaries according to the Schema, with each dictionary representing an Entity and containing all the fields defined in the Schema. If the Collection has the dynamic field enabled, each dictionary can also include fields that are not defined in the Schema.

In this section, you will insert entities into a Collection created in the quick-setup manner. A Collection created in this manner has only two fields, named **id** and **vector**. Additionally, this Collection has the dynamic field enabled, so the Entities in the example code include a field called **color** that is not defined in the Schema.

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

data=[
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

res = client.insert(
    collection_name="quick_setup",
    data=data
)

print(res)

# Output
# {'insert_count': 10, 'ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
```

```java
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.response.InsertResp;

import java.util.*;

MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
        .uri("http://localhost:19530")
        .token("root:Milvus")
        .build());

Gson gson = new Gson();
List<JsonObject> data = Arrays.asList(
        gson.fromJson("{\"id\": 0, \"vector\": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], \"color\": \"pink_8682\"}", JsonObject.class),
        gson.fromJson("{\"id\": 1, \"vector\": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], \"color\": \"red_7025\"}", JsonObject.class),
        gson.fromJson("{\"id\": 2, \"vector\": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592], \"color\": \"orange_6781\"}", JsonObject.class),
        gson.fromJson("{\"id\": 3, \"vector\": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345], \"color\": \"pink_9298\"}", JsonObject.class),
        gson.fromJson("{\"id\": 4, \"vector\": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184, 0.30337481143159106], \"color\": \"red_4794\"}", JsonObject.class),
        gson.fromJson("{\"id\": 5, \"vector\": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383, -0.1446277761879955], \"color\": \"yellow_4222\"}", JsonObject.class),
        gson.fromJson("{\"id\": 6, \"vector\": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192, -0.8984947637863987], \"color\": \"red_9392\"}", JsonObject.class),
        gson.fromJson("{\"id\": 7, \"vector\": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709, 0.5378064918413052], \"color\": \"grey_8510\"}", JsonObject.class),
        gson.fromJson("{\"id\": 8, \"vector\": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872, -0.6140360785406336], \"color\": \"white_9381\"}", JsonObject.class),
        gson.fromJson("{\"id\": 9, \"vector\": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717, -0.6980531615588608], \"color\": \"purple_4976\"}", JsonObject.class)
);

InsertReq insertReq = InsertReq.builder()
        .collectionName("quick_setup")
        .data(data)
        .build();

InsertResp insertResp = client.insert(insertReq);
System.out.println(insertResp);

// Output
// InsertResp(InsertCnt=10, primaryKeys=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```
