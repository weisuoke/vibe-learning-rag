"""
RAG 场景：观察检索内容如何影响注意力
演示：模型如何 "关注" 注入的上下文
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# 使用英文模型 （更好的演示效果）
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

def analyze_rag_attention(question, context):
    """分析 RAG 场景下的注意力 """
    # 构造 RAG 风格的输入
    text = f"Context: {context} Question: {question}"

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    #取最后一层的平均注意力
    attention = outputs.attentions[-1][0].mean(dim=0).numpy()

    return tokens, attention

# ===== RAG 示例 =====
print("=== RAG 注意力分析 ===\n")

question = "What color is the apple?"
context = "The apple on the table is red and fresh."

tokens, attention = analyze_rag_attention(question, context)

print(f"问题: {question}")
print(f"上下文: {context}")
print(f"\n分词结果: {tokens}")