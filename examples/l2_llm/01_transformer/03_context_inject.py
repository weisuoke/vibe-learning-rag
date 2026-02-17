"""
RAG 场景：观察注入的上下文是否被“问题词”关注
说明：这里用 BERT 的 sentence-pair 输入 (question, context)
"""

# PyTorch：提供张量、推理模式（no_grad）等能力
import torch

# HuggingFace Transformers：加载分词器与预训练模型
from transformers import AutoTokenizer, AutoModel

# 使用英文模型（更好的演示效果）
model_name = "bert-base-uncased"

# 加载分词器：负责把文本 -> token ids（以及 token_type_ids 等）
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型：output_attentions=True 让 forward 返回每一层的注意力矩阵
model = AutoModel.from_pretrained(model_name, output_attentions=True)

# eval()：关闭 dropout 等训练期随机性，让结果更稳定
model.eval()

def analyze_rag_attention(question, context):
    """分析 RAG 场景下的注意力（question -> context）"""
    # BERT pair: [CLS] question [SEP] context [SEP]
    inputs = tokenizer(
        question,  # 第 1 段文本（token_type_id = 0）
        context,   # 第 2 段文本（token_type_id = 1）
        return_tensors="pt",          # 返回 PyTorch 张量（而不是 Python list）
        max_length=512,               # BERT 最大序列长度一般为 512
        truncation=True,              # 超长则截断（通常优先截断第 2 段）
        return_token_type_ids=True,   # 返回 token_type_ids，用于区分两段文本
    )

    # inputs["input_ids"] 形状通常是 (batch, seq_len)；这里取 batch=0
    input_ids = inputs["input_ids"][0]

    # 把 token ids 转回 token 字符串，便于打印/分析
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # token_type_ids: 0 表示 question 段，1 表示 context 段
    token_type_ids = inputs["token_type_ids"][0].tolist()

    # no_grad：关闭梯度计算，推理更快、更省内存
    with torch.no_grad():
        # **inputs 会展开为 input_ids / attention_mask / token_type_ids 等
        outputs = model(**inputs)

    # 取最后一层的平均注意力 (seq_len, seq_len)
    # 行表示 query token，列表示 key token
    attention = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()

    return tokens, token_type_ids, attention

# ===== RAG 示例 =====
print("=== RAG 注意力分析 ===\n")

question = "What color is the apple?"
context = "The apple on the table is red and fresh."

tokens, token_type_ids, attention = analyze_rag_attention(question, context)

print(f"问题: {question}")
print(f"上下文: {context}")
print(f"\n分词结果: {tokens}")

# 指定我们关心的“问题关键词”（会在 question 段里寻找对应 token）
question_keywords = ["color", "apple"]

# question_positions / context_positions：用于限制“搜索范围”
# - 只在 question 段里找 query token
# - 只在 context 段里挑 key token（top-k 只从这里取）
question_positions = [
    i for i, (tok, tt) in enumerate(zip(tokens, token_type_ids))
    if tt == 0 and tok not in ("[CLS]", "[SEP]")
]
context_positions = [
    i for i, (tok, tt) in enumerate(zip(tokens, token_type_ids))
    if tt == 1 and tok not in ("[CLS]", "[SEP]")
]

print("\n=== 问题词对上下文的注意力（只看 context 段） ===")
for q_word in question_keywords:
    # 找到该关键词在 question 段对应的 token 位置（可能多个：wordpiece 会拆词）
    q_positions = [
        i for i in question_positions
        if q_word in tokens[i].lower().replace("##", "")
    ]
    if not q_positions:
        continue

    # 如果一个词被拆成多个 wordpiece，这里对它们的 attention 做平均
    # attention[q_positions]：取这些 query token 的“整行”注意力分布
    att_scores = attention[q_positions].mean(axis=0)

    # 只在 context_positions 里选 top-5（避免把 question 段 token 也混进来）
    top_context = sorted(
        [(i, att_scores[i]) for i in context_positions],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    print(f"\n'{q_word}' 关注的上下文词:")
    for idx, score in top_context:
        print(f"  → '{tokens[idx]}': {score:.3f}")

print("\n=== 提示 ===")
print("注意力只是模型内部信号，不等价于可解释性；不同层/不同 head 的模式也会不同。")
