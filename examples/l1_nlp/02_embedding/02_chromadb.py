import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

collection.add(
    documents=["文档1内容", "文档2内容"],
    ids=["id1", "id2"]
)   

# 检索
results = collection.query(query_texts=["查询内容"], n_results=2)

print(results)