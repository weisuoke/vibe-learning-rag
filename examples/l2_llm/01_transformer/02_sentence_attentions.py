"""
对比不同句子的注意力模式
演示：词序变化如何影响注意力
"""

import torch
from transformers import AutoTokenizer, AutoModel

# 加载模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

def get_attention_summary(text):
    """获取文本的注意力摘要"""
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # 取最后一层的平均注意力
    avg_attention = outputs.attentions[-1][0].mean(dim=0).numpy()

    return tokens, avg_attention

# ===== 对比两个句子 =====
print("=== 对比词序变化的影响 ===\n")

sentences = [
    "狗咬人",
    "人咬狗"
]

for sent in sentences:
    tokens, attention = get_attention_summary(sent)
    print(f"句子: {sent}")
    print(f"分词: {tokens}")

    # 显示每个词的注意力分布
    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]"]:
            continue
        scores = attention[i][1:-1] # 去掉 CLS 和 SEP
        content_tokens = tokens[1:-1]
        print(f" '{token}' 的注意力: ", end="")
        for t, s in zip(content_tokens, scores):
            print(f"{t}:{s:.2f} ", end="")
        print()
    print()