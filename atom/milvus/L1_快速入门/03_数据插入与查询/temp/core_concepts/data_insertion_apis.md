# Data Insertion APIs Documentation (Milvus 2.6)

---
source: https://milvus.io/docs/upsert-entities.md
title: Upsert Entities | Milvus Documentation
fetched_at: 2026-02-21
---

# Upsert Entities

The `upsert` operation provides a convenient way to insert or update entities in a collection.

## Overview

You can use `upsert` to either insert a new entity or update an existing one, depending on whether the primary key provided in the upsert request exists in the collection. If the primary key is not found, an insert operation occurs. Otherwise, an update operation will be performed.

An upsert in Milvus works in either **override** or **merge** mode.

### Upsert in override mode

An upsert request that works in override mode combines an insert and a delete. When an `upsert` request for an existing entity is received, Milvus inserts the data carried in the request payload and deletes the existing entity with the original primary key specified in the data at the same time.

If the target collection has `autoid` enabled on its primary field, Milvus will generate a new primary key for the data carried in the request payload before inserting it.

For fields with `nullable` enabled, you can omit them in the `upsert` request if they do not require any updates.

### Upsert in merge mode *Compatible with Milvus v2.6.2+*

You can also use the `partial_update` flag to make an upsert request work in merge mode. This allows you to include only the fields that need updating in the request payload.

To perform a merge, set `partial_update` to `True` in the `upsert` request along with the primary key and the fields to update with their new values.

Upon receiving such a request, Milvus performs a query with strong consistency to retrieve the entity, updates the field values based on the data in the request, inserts the modified data, and then deletes the existing entity with the original primary key carried in the request.

### Upsert behaviors: special notes

There are several special notes you should consider before using the merge feature. The following cases assume that you have a collection with two scalar fields named `title` and `issue`, along with a primary key `id` and a vector field called `vector`.

**Upsert fields with nullable enabled.**

Suppose that the issue field can be null. When you upsert these fields, note that:

- If you omit the issue field in the upsert request and disable partial_update, the issue field will be updated to null instead of retaining its original value.
- To preserve the original value of the issue field, you need either to enable partial_update and omit the issue field or include the issue field with its original value in the upsert request.

**Upsert keys in the dynamic field.**

Suppose that you have enabled the dynamic key in the example collection, and the key-value pairs in the dynamic field of an entity are similar to `{"author": "John", "year": 2020, "tags": ["fiction"]}`.

When you upsert the entity with keys, such as author, year, or tags, or add other keys, note that:

- If you upsert with partial_update disabled, the default behavior is to **override**. It means that the value of the dynamic field will be overridden by all non-schema-defined fields included in the request and their values. For example, if the data included in the request is `{"author": "Jane", "genre": "fantasy"}`, the key-value pairs in the dynamic field of the target entity will be updated to that.
- If you upsert with partial_update enabled, the default behavior is to **merge**. It means that the value of the dynamic field will merge with all non-schema-defined fields included in the request and their values. For example, if the data included in the request is `{"author": "John", "year": 2020, "tags": ["fiction"]}`, the key-value pairs in the dynamic field of the target entity will become `{"author": "John", "year": 2020, "tags": ["fiction"], "genre": "fantasy"}` after the upsert.

**Upsert a JSON field.**

Suppose that the example collection has a schema-defined JSON field named extras, and the key-value pairs in this JSON field of an entity are similar to `{"author": "John", "year": 2020, "tags": ["fiction"]}`.

When you upsert the extras field of an entity with modified JSON data, note that the JSON field is treated as a whole, and you cannot update individual keys selectively. In other words, the JSON field **DOES NOT** support upsert in **merge** mode.

### Limits & Restrictions

Based on the above content, there are several limits and restrictions to follow:

- The upsert request must always include the primary keys of the target entities.
- The target collection must be loaded and available for queries.
- All fields specified in the request must exist in the schema of the target collection.
- The values of all fields specified in the request must match the data types defined in the schema.
- For any field derived from another using functions, Milvus will remove the derived field during the upsert to allow recalculation.

## Upsert entities in a collection

In this section, we will upsert entities into a collection named `my_collection`. This collection has only two fields, named `id`, `vector`, `title`, and `issue`. The `id` field is the primary field, while the `title` and `issue` fields are scalar fields.

The three entities, if exists in the collection, will be overridden by those included the upsert request.

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

data=[
    {
        "id": 0,
        "vector": [-0.619954382375778, 0.4479436794798608, -0.17493894838751745, -0.4248030059917294, -0.8648452746018911],
        "title": "Artificial Intelligence in Real Life",
        "issue": "vol.12"
    }, {
        "id": 1,
        "vector": [0.4762662251462588, -0.6942502138717026, -0.4490002642657902, -0.628696575798281, 0.9660395877041965],
        "title": "Hollow Man",
        "issue": "vol.19"
    }, {
        "id": 2,
        "vector": [-0.8864122635045097, 0.9260170474445351, 0.801326976181461, 0.6383943392381306, 0.7563037341572827],
        "title": "Treasure Hunt in Missouri",
        "issue": "vol.12"
    }
]

res = client.upsert(
    collection_name='my_collection',
    data=data
)

print(res)

# Output
# {'upsert_count': 3}
```

## Key Points for RAG Development

1. **Upsert with Embedding Functions**: When using Embedding Functions in Milvus 2.6, you can upsert raw text directly without manually generating vectors. Milvus will automatically call the configured embedding provider to generate vectors.

2. **Partial Update**: The `partial_update` flag (Milvus 2.6.2+) allows you to update only specific fields without replacing the entire entity, which is useful for incremental updates in RAG systems.

3. **Dynamic Fields**: Upsert supports dynamic fields, allowing you to add metadata to entities without modifying the schema. This is particularly useful for RAG applications where document metadata may vary.

4. **Performance Considerations**:
   - Upsert operations are more expensive than pure inserts because they require a query to check if the entity exists
   - For bulk operations, consider batching upsert requests
   - Use partial_update when only updating specific fields to reduce overhead

5. **Use Cases in RAG**:
   - **Document Updates**: When a document in your knowledge base is updated, use upsert to replace the old version
   - **Incremental Indexing**: Add new documents or update existing ones without rebuilding the entire index
   - **Metadata Enrichment**: Update document metadata (e.g., tags, categories) without re-embedding the content
