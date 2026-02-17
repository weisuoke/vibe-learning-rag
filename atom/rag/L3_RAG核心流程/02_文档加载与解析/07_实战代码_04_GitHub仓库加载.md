# 实战代码4：GitHub仓库加载

> 从GitHub仓库加载代码和文档的完整示例

---

## 场景描述

演示如何从GitHub仓库加载代码文件和文档，构建代码问答系统。

**适用场景：**
- 代码库文档化
- 技术知识库构建
- 代码搜索和推荐

---

## 依赖库

```bash
# 安装依赖
uv add gitpython requests langchain langchain-openai chromadb python-dotenv
```

---

## 完整代码

```python
"""
GitHub仓库加载示例
演示：从GitHub加载代码和文档并构建知识库

依赖库：
- gitpython: Git仓库操作
- requests: HTTP请求
- langchain: 文档处理框架

参考来源：
- GitPython Documentation (2025): https://gitpython.readthedocs.io/
- GitHub API Documentation (2025): https://docs.github.com/en/rest
- LangChain Git Loaders (2025): https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/git
"""

import os
import base64
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
import requests

load_dotenv()

# ===== 1. 使用GitHub API加载文件 =====
print("=== 1. 使用GitHub API加载文件 ===")

class GitHubLoader:
    """GitHub仓库加载器"""

    def __init__(self, repo: str, branch: str = "main", access_token: str = None):
        """
        初始化GitHub加载器

        Args:
            repo: 仓库名称，格式：owner/repo
            branch: 分支名称
            access_token: GitHub访问令牌（可选，但推荐）
        """
        self.repo = repo
        self.branch = branch
        self.access_token = access_token or os.getenv("GITHUB_TOKEN")

        self.api_url = f"https://api.github.com/repos/{repo}"
        self.headers = {}
        if self.access_token:
            self.headers["Authorization"] = f"token {self.access_token}"

    def load_file(self, file_path: str) -> Document:
        """加载单个文件"""
        print(f"加载文件: {file_path}")

        # 获取文件内容
        url = f"{self.api_url}/contents/{file_path}?ref={self.branch}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        file_data = response.json()

        # 解码内容（GitHub API返回base64编码）
        content = base64.b64decode(file_data["content"]).decode("utf-8")

        # 创建Document
        return Document(
            page_content=content,
            metadata={
                "source": f"github:{self.repo}/{file_path}",
                "repo": self.repo,
                "branch": self.branch,
                "file_path": file_path,
                "file_type": os.path.splitext(file_path)[1],
                "url": file_data["html_url"],
                "size": file_data["size"]
            }
        )

    def load_directory(self, directory: str = "", file_extensions: List[str] = None) -> List[Document]:
        """加载目录下的所有文件"""
        file_extensions = file_extensions or [".md", ".py", ".js", ".ts"]

        print(f"扫描目录: {directory or '根目录'}")

        # 获取仓库树
        tree_url = f"{self.api_url}/git/trees/{self.branch}?recursive=1"
        response = requests.get(tree_url, headers=self.headers)
        response.raise_for_status()

        tree = response.json()

        # 过滤文件
        files = [
            item for item in tree.get("tree", [])
            if item["type"] == "blob" and
            (not directory or item["path"].startswith(directory)) and
            any(item["path"].endswith(ext) for ext in file_extensions)
        ]

        print(f"找到 {len(files)} 个匹配的文件")

        # 加载文件（限制数量避免API限制）
        documents = []
        for file_info in files[:50]:  # 限制50个文件
            try:
                doc = self.load_file(file_info["path"])
                documents.append(doc)
                print(f"  ✅ {file_info['path']}")
            except Exception as e:
                print(f"  ❌ {file_info['path']}: {e}")

        return documents

# 测试GitHub加载器
# loader = GitHubLoader(
#     repo="langchain-ai/langchain",
#     branch="master",
#     access_token=os.getenv("GITHUB_TOKEN")
# )
# docs = loader.load_directory("docs", file_extensions=[".md"])
# print(f"\n总共加载 {len(docs)} 个文档")

# ===== 2. 使用LangChain的GitLoader =====
print("\n=== 2. 使用LangChain的GitLoader ===")

from langchain.document_loaders import GitLoader

def load_repo_with_langchain(
    clone_url: str,
    repo_path: str = "./temp_repo",
    branch: str = "main",
    file_filter: callable = None
):
    """使用LangChain加载Git仓库"""

    print(f"克隆仓库: {clone_url}")
    print(f"本地路径: {repo_path}")

    # 默认文件过滤器：只加载.md和.py文件
    if file_filter is None:
        file_filter = lambda file_path: file_path.endswith((".md", ".py"))

    try:
        loader = GitLoader(
            clone_url=clone_url,
            repo_path=repo_path,
            branch=branch,
            file_filter=file_filter
        )

        documents = loader.load()
        print(f"✅ 加载成功: {len(documents)} 个文档")

        # 显示前3个文档
        for i, doc in enumerate(documents[:3], 1):
            print(f"\n文档 {i}:")
            print(f"  来源: {doc.metadata['source']}")
            print(f"  文件: {doc.metadata['file_path']}")
            print(f"  内容: {doc.page_content[:100]}...")

        return documents

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return []

# 测试LangChain GitLoader
# docs = load_repo_with_langchain(
#     clone_url="https://github.com/langchain-ai/langchain",
#     repo_path="./temp/langchain",
#     branch="master"
# )

# ===== 3. 本地Git仓库加载 =====
print("\n=== 3. 本地Git仓库加载 ===")

import git

class LocalGitRepoLoader:
    """本地Git仓库加载器"""

    def __init__(
        self,
        repo_path: str,
        file_extensions: List[str] = None,
        exclude_dirs: List[str] = None
    ):
        self.repo_path = repo_path
        self.file_extensions = file_extensions or [".md", ".py", ".js", ".ts"]
        self.exclude_dirs = exclude_dirs or [
            ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"
        ]

        # 打开Git仓库
        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"{repo_path} 不是有效的Git仓库")

    def load(self) -> List[Document]:
        """加载仓库文档"""
        documents = []

        print(f"扫描仓库: {self.repo_path}")
        print(f"当前分支: {self.repo.active_branch.name}")

        # 遍历仓库文件
        for root, dirs, files in os.walk(self.repo_path):
            # 排除特定目录
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file in files:
                # 检查文件扩展名
                if not any(file.endswith(ext) for ext in self.file_extensions):
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)

                try:
                    doc = self._load_file(file_path, relative_path)
                    if doc:
                        documents.append(doc)
                        print(f"  ✅ {relative_path}")
                except Exception as e:
                    print(f"  ❌ {relative_path}: {e}")

        print(f"\n总共加载 {len(documents)} 个文档")
        return documents

    def _load_file(self, file_path: str, relative_path: str) -> Document:
        """加载单个文件"""
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 获取Git信息
        try:
            commits = list(self.repo.iter_commits(paths=relative_path, max_count=1))
            if commits:
                last_commit = commits[0]
                last_modified = last_commit.committed_datetime.isoformat()
                last_author = last_commit.author.name
            else:
                last_modified = None
                last_author = None
        except:
            last_modified = None
            last_author = None

        # 创建Document
        return Document(
            page_content=content,
            metadata={
                "source": file_path,
                "relative_path": relative_path,
                "file_type": os.path.splitext(file_path)[1],
                "repo_path": self.repo_path,
                "last_modified": last_modified,
                "last_author": last_author,
                "current_branch": self.repo.active_branch.name
            }
        )

# 测试本地仓库加载
# loader = LocalGitRepoLoader(
#     repo_path="/path/to/your/repo",
#     file_extensions=[".md", ".py"]
# )
# docs = loader.load()

# ===== 4. RAG应用：构建代码问答系统 =====
print("\n=== 4. RAG应用：构建代码问答系统 ===")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def build_code_qa_system(repo_path: str, output_dir: str = "./code_kb"):
    """从代码仓库构建问答系统"""

    print(f"=== 构建代码问答系统 ===")
    print(f"仓库路径: {repo_path}")
    print(f"输出目录: {output_dir}\n")

    # 1. 加载代码仓库
    print("步骤1: 加载代码仓库")
    loader = LocalGitRepoLoader(
        repo_path=repo_path,
        file_extensions=[".md", ".py", ".js", ".ts"]
    )
    documents = loader.load()

    if not documents:
        print("❌ 没有加载到任何文档")
        return None

    # 2. 文本分块
    print("\n步骤2: 文本分块")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ 分块完成: {len(chunks)} 个文本块")

    # 3. 向量化并存储
    print("\n步骤3: 向量化并存储")
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=output_dir
        )
        print(f"✅ 知识库构建成功")

        # 4. 构建QA链
        print("\n步骤4: 构建QA链")
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        # 5. 测试问答
        print("\n步骤5: 测试问答")
        test_queries = [
            "这个项目的主要功能是什么？",
            "如何安装和使用？",
            "有哪些核心的类和函数？"
        ]

        for query in test_queries:
            print(f"\n查询: {query}")
            result = qa_chain({"query": query})

            print(f"答案: {result['result']}\n")
            print("相关代码:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"  {i}. {doc.metadata['relative_path']}")
                if doc.metadata.get('last_author'):
                    print(f"     作者: {doc.metadata['last_author']}")

        return qa_chain

    except Exception as e:
        print(f"❌ 构建失败: {e}")
        return None

# 测试构建代码问答系统
# qa_chain = build_code_qa_system("/path/to/your/repo")

# ===== 5. 代码搜索功能 =====
print("\n=== 5. 代码搜索功能 ===")

def search_code(query: str, vectorstore, k: int = 5):
    """搜索相关代码"""
    print(f"搜索: {query}\n")

    # 检索相关文档
    results = vectorstore.similarity_search(query, k=k)

    print(f"找到 {len(results)} 个相关代码片段:\n")

    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata['relative_path']}")
        print(f"   文件类型: {doc.metadata.get('file_type', 'unknown')}")
        if doc.metadata.get('last_author'):
            print(f"   最后修改: {doc.metadata['last_author']}")
        print(f"   内容: {doc.page_content[:200]}...")
        print()

# 测试代码搜索
# search_code("数据库连接", vectorstore)

# ===== 6. 增量更新策略 =====
print("\n=== 6. 增量更新策略 ===")

class IncrementalGitLoader:
    """增量Git仓库加载器"""

    def __init__(self, repo_path: str, last_commit_sha: str = None):
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)
        self.last_commit_sha = last_commit_sha

    def load_changes(self) -> List[Document]:
        """只加载自上次以来的变更"""

        if not self.last_commit_sha:
            print("首次加载，加载所有文件")
            loader = LocalGitRepoLoader(self.repo_path)
            return loader.load()

        print(f"增量加载，从提交 {self.last_commit_sha[:7]} 开始")

        # 获取变更的文件
        changed_files = self._get_changed_files()
        print(f"发现 {len(changed_files)} 个变更文件")

        documents = []
        for file_path in changed_files:
            try:
                full_path = os.path.join(self.repo_path, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": full_path,
                            "relative_path": file_path,
                            "file_type": os.path.splitext(file_path)[1],
                            "change_type": "modified"
                        }
                    )
                    documents.append(doc)
                    print(f"  ✅ {file_path}")
            except Exception as e:
                print(f"  ❌ {file_path}: {e}")

        return documents

    def _get_changed_files(self) -> List[str]:
        """获取变更的文件列表"""
        # 获取两个提交之间的差异
        diff = self.repo.git.diff(
            self.last_commit_sha,
            self.repo.head.commit.hexsha,
            name_only=True
        )

        return diff.split('\n') if diff else []

    def get_current_commit(self) -> str:
        """获取当前提交SHA"""
        return self.repo.head.commit.hexsha

# 测试增量加载
# loader = IncrementalGitLoader(
#     repo_path="/path/to/repo",
#     last_commit_sha="abc123"
# )
# new_docs = loader.load_changes()
# current_commit = loader.get_current_commit()
# print(f"当前提交: {current_commit}")

# ===== 7. 文件过滤策略 =====
print("\n=== 7. 文件过滤策略 ===")

# 推荐的文件过滤配置
FILE_EXTENSIONS = {
    "文档": [".md", ".rst", ".txt"],
    "Python": [".py"],
    "JavaScript": [".js", ".jsx", ".ts", ".tsx"],
    "配置": [".json", ".yaml", ".yml", ".toml"],
    "其他": [".go", ".rs", ".java", ".cpp"]
}

EXCLUDE_DIRS = [
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    "target",
    "coverage"
]

EXCLUDE_FILES = [
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    ".DS_Store"
]

def should_include_file(file_path: str) -> bool:
    """判断是否应该包含文件"""
    # 检查文件扩展名
    all_extensions = []
    for exts in FILE_EXTENSIONS.values():
        all_extensions.extend(exts)

    if not any(file_path.endswith(ext) for ext in all_extensions):
        return False

    # 检查排除的文件
    file_name = os.path.basename(file_path)
    if file_name in EXCLUDE_FILES:
        return False

    # 检查排除的目录
    path_parts = file_path.split(os.sep)
    if any(excluded in path_parts for excluded in EXCLUDE_DIRS):
        return False

    return True

# 测试文件过滤
test_files = [
    "src/main.py",
    "node_modules/package/index.js",
    "docs/README.md",
    "package-lock.json",
    ".venv/lib/python.py"
]

print("文件过滤测试:")
for file_path in test_files:
    result = "✅ 包含" if should_include_file(file_path) else "❌ 排除"
    print(f"  {result}: {file_path}")

print("\n=== 示例代码执行完成 ===")
print("\n使用说明:")
print("1. 设置GITHUB_TOKEN环境变量（用于GitHub API）")
print("2. 准备要加载的仓库路径或URL")
print("3. 取消注释相应的测试代码")
print("4. 运行脚本查看效果")
print("\n功能:")
print("- 从GitHub加载代码和文档")
print("- 从本地Git仓库加载")
print("- 构建代码问答系统")
print("- 代码搜索和推荐")
print("- 增量更新支持")
```

---

## 运行输出示例

```
=== 1. 使用GitHub API加载文件 ===
扫描目录: docs
找到 25 个匹配的文件
  ✅ docs/README.md
  ✅ docs/getting_started.md
  ✅ docs/api_reference.md
  ...

总共加载 25 个文档

=== 3. 本地Git仓库加载 ===
扫描仓库: /path/to/repo
当前分支: main
  ✅ README.md
  ✅ src/main.py
  ✅ src/utils.py
  ...

总共加载 45 个文档

=== 4. RAG应用：构建代码问答系统 ===
步骤1: 加载代码仓库
步骤2: 文本分块
✅ 分块完成: 120 个文本块
步骤3: 向量化并存储
✅ 知识库构建成功

查询: 这个项目的主要功能是什么？
答案: 这个项目是一个RAG系统实现，主要功能包括文档加载、文本分块、向量化存储和智能检索...

相关代码:
  1. README.md
     作者: 张三
  2. src/main.py
     作者: 李四
```

---

## 关键要点

### 1. GitHub API限制

```python
# 未认证：60次/小时
# 已认证：5000次/小时

# 设置访问令牌
headers = {
    "Authorization": f"token {GITHUB_TOKEN}"
}
```

### 2. 文件过滤最佳实践

```python
# 推荐的文件扩展名
INCLUDE_EXTENSIONS = [".md", ".py", ".js", ".ts"]

# 必须排除的目录
EXCLUDE_DIRS = [
    ".git",           # Git元数据
    "node_modules",   # Node依赖
    "__pycache__",    # Python缓存
    ".venv", "venv"   # Python虚拟环境
]
```

### 3. 增量更新策略

```python
# 保存上次处理的提交SHA
last_commit = loader.get_current_commit()
save_to_config(last_commit)

# 下次只加载变更
loader = IncrementalGitLoader(repo_path, last_commit)
new_docs = loader.load_changes()
```

---

## 常见问题

### Q1: 如何处理大型仓库？

```python
# 1. 限制文件数量
files = files[:100]  # 只加载前100个文件

# 2. 按目录分批加载
directories = ["docs", "src", "tests"]
for directory in directories:
    docs = loader.load_directory(directory)

# 3. 使用浅克隆
git clone --depth 1 <repo_url>
```

### Q2: 如何处理私有仓库？

```python
# 需要GitHub访问令牌
# 1. 生成Personal Access Token
# 2. 设置环境变量
export GITHUB_TOKEN=your_token_here

# 3. 在代码中使用
loader = GitHubLoader(
    repo="owner/private-repo",
    access_token=os.getenv("GITHUB_TOKEN")
)
```

### Q3: 如何提取代码结构？

```python
import ast

def extract_python_functions(code: str):
    """提取Python函数定义"""
    tree = ast.parse(code)
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "docstring": ast.get_docstring(node)
            })

    return functions
```

---

## 扩展阅读

- [GitPython Documentation](https://gitpython.readthedocs.io/) (2025)
- [GitHub API Documentation](https://docs.github.com/en/rest) (2025)
- [LangChain Git Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/git) (2025)

---

**版本：** v1.0
**最后更新：** 2026-02-15
**下一步：** 阅读 [07_实战代码_05_多格式统一管道.md](./07_实战代码_05_多格式统一管道.md)
