# Insert Succeeds But Query Returns 0 (Milvus 2.6 + pymilvus)

## Symptom

In `01_test_connection.py`, we insert 10 rows successfully, but an immediate scalar `query()`
returns `0` results (even with a broad filter like `id >= 0`).

## Repro

1. Start Milvus standalone (Docker).
2. Run:

```bash
./.venv/bin/python examples/milvus/l1_quick_start/01_install/01_test_connection.py
```

Observed output:

- `✅ 插入 10 条测试数据`
- `✅ 查询成功,返回 0 条数据`

## Root Cause (What is actually happening)

Milvus write and read paths are asynchronous:

- `insert()` confirms the write request is accepted/processed on the write path.
- Newly inserted data typically lands in a *growing segment* and becomes visible to the
  query side after a propagation/visibility delay.
- With the default consistency (commonly `Bounded`), a query uses a "safe" timestamp
  (`guarantee_ts`) and may **not** read the very latest writes yet.

So "insert OK" does not imply "immediately queryable".

## Fix / Mitigations

### Option A (Most deterministic): flush + strong consistency

```python
ret = client.insert(collection_name=test_collection, data=test_data)
client.flush(collection_name=test_collection)

results = client.query(
    collection_name=test_collection,
    filter="id >= 0",
    output_fields=["id", "text"],
    limit=5,
    consistency_level="Strong",  # or "Session"
)
```

### Option B: small delay (less deterministic)

After `insert()`, wait a short time (e.g. `time.sleep(1)`) before querying.

## Notes

- This is not a bug in the filter expression; it's a read-your-writes visibility behavior
  governed by consistency and flush/segment state.
- If you still see `0` after `flush + Strong`, double-check: collection name, primary key
  field type, whether you are querying the expected field names, and whether partitions
  are involved.

