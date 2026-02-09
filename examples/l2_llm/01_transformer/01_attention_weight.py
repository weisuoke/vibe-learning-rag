import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 1. 加载模型和分词器 =====
print("=== 加载模型 ===")
model_name = "bert-base-chinese" # 使用中文 BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True) # 输出注意力权重

print(f"模型: {model_name}")
print(f"注意力头数: {model.config.num_attention_heads}")
print(f"隐藏层数: {model.config.num_hidden_layers}")

#  ===== 2. 准备输入文本 =====
text = "小猫坐在垫子上"
print(f"\n=== 输入文本 ===")
print(f"原文: {text}")

# 分词
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"分词结果: {tokens}")

# ===== 3. 获取注意力权重 =====
print("\n=== 获取注意力权重 ===")
with torch.no_grad():
    outputs = model(**inputs)

# attentions 是一个元组，每层一个张量
# 形状: (batch_size, num_heads, seq_len, seq_len)
attentions = outputs.attentions
print(f"层数: {len(attentions)}")
print(f"每层形状: {attentions[0].shape}")

# ===== 4. 可视化第一层的注意力 =====
def plot_attention(attention, tokens, layer=0, head=0):
    """绘制注意力热力图"""
    att = attention[layer][0, head].numpy()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        att,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
    )
    plt.title(f"注意力热力图 (Layer {layer}, Head {head})")
    plt.xlabel("被关注的词 (Key)")
    plt.ylabel("关注者 (Query)")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.show()
    print(f"图片已保存: attention_heatmap.png")

# 绘制第一层第一个头的注意力
plot_attention(attentions, tokens, layer=0, head=0)

# ===== 5. 分析注意力模式 =====
print("\n=== 注意力分析 ===")
# 取第一层所有头的平均
avg_attention = attentions[0][0].mean(dim=0).numpy()

# 找出每个词最关注的词
for i, token in enumerate(tokens):
    if token in ["[CLS]", "[SEP]"]:
        continue
    top_idx = avg_attention[i].argsort()[-3:][::-1] # 前3个
    top_tokens = [tokens[j] for j in top_idx]
    top_scores = [avg_attention[i][j] for j in top_idx]
    print(f"'{token}' 最关注: {list(zip(top_tokens, [f'{s:.2f}' for s in top_scores]))}")