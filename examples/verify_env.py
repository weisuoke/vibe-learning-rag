"""
环境验证脚本
验证所有核心库是否正确安装
"""

print("=== Python 环境验证 ===\n")

# 检查 Python 版本
import sys
print(f"✓ Python 版本: {sys.version.split()[0]}")

# 检查核心库
libraries = [
    ("openai", "OpenAI API"),
    ("sentence_transformers", "Sentence Transformers"),
    ("chromadb", "ChromaDB"),
    ("langchain", "LangChain"),
    ("langchain_openai", "LangChain OpenAI"),
    ("pypdf", "PyPDF"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("dotenv", "Python Dotenv"),
]

print("\n核心库检查：")
all_ok = True
for module_name, display_name in libraries:
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
    except ImportError as e:
        print(f"✗ {display_name}: {e}")
        all_ok = False

if all_ok:
    print("\n🎉 所有核心库已正确安装！")
    print("\n下一步：")
    print("1. 复制 .env.example 为 .env")
    print("2. 在 .env 中配置你的 OPENAI_API_KEY")
    print("3. 运行 python examples/basic_rag.py")
else:
    print("\n⚠️  部分库安装失败，请运行 'uv sync' 重新安装")
