import tiktoken  # OpenAI 的 tokenizer 库: 用于把文本编码/解码为 token 序列

#
def split_by_tokens(text: str, max_tokens: int = 500, model: str = "gpt-5-2") -> list[str]:  # 按 token 上限把文本切块
    """按Token数量切分文本"""  # 文档字符串: 简要说明该函数用途
    encoder = tiktoken.encoding_for_model(model)  # 根据模型名选择对应的编码器
    tokens = encoder.encode(text)  # 把原始文本编码为 token id 列表

    # 预分配结果列表: 每个元素是一个不超过 max_tokens 的文本块
    chunks = []  # 存放切分后的文本块
    for i in range(0, len(tokens), max_tokens):  # 以 max_tokens 为步长遍历 token 序列的起始位置
        chunk_tokens = tokens[i:i + max_tokens]  # 取当前窗口的一段 token
        chunk_text = encoder.decode(chunk_tokens)  # 将 token 段解码回文本
        chunks.append(chunk_text)  # 追加到 chunks 结果中

    # 返回所有切分得到的文本块
    return chunks  # list[str]，每个字符串对应一个 chunk

# ------------------------------------------------------------
long_text = "这是一段很长的文本..." * 100  # 构造一段较长文本用于演示切分效果
chunks = split_by_tokens(long_text, max_tokens=100)  # 按 100 tokens 为上限切分
print(f"切分成 {len(chunks)} 个块")  # 打印最终切分得到的块数量
