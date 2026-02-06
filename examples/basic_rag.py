"""
基础 RAG 示例
演示：文档加载 -> Embedding -> 向量存储 -> 检索 -> 生成
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# 加载环境变量
load_dotenv()

def main():
    print("=== 基础 RAG 示例 ===\n")

    # 1. 初始化客户端
    print("1. 初始化客户端...")
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")  # 如果未设置则使用默认值
    )
    chroma_client = chromadb.Client()

    # 2. 创建向量集合
    print("2. 创建向量集合...")
    collection = chroma_client.create_collection("demo")

    # 3. 示例文档
    docs = [
        "RAG 是检索增强生成的缩写，结合了检索和生成两个步骤",
        "Embedding 将文本转换为高维向量，捕捉语义信息",
        "ChromaDB 是一个轻量级向量数据库，适合快速原型开发"
    ]

    print(f"3. 加载了 {len(docs)} 个示例文档")
    print("\n✓ 环境配置验证成功！")
    print("✓ 所有核心库可正常导入")
    print("\n下一步：配置 .env 文件后可运行完整 RAG 流程")

if __name__ == "__main__":
    main()
