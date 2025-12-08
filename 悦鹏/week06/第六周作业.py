import torch
import torch.nn as nn
import math
import numpy as np
from transformers import BertModel


class SingleLayerTransformerTorch(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072):
        super(SingleLayerTransformerTorch, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # 自注意力层的线性变换
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # 注意力输出层
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_layer_norm = nn.LayerNorm(hidden_size)

        # 前馈网络
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)

        # 激活函数
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        """多头机制 - 将隐藏状态拆分为多个注意力头"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_size]

    def forward(self, hidden_states, attention_mask=None):
        # 自注意力计算
        # 线性变换得到Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 多头拆分
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算注意力分数 [batch, heads, seq_len, seq_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Softmax归一化
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 注意力加权求和
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 注意力输出层 + 残差连接 + LayerNorm
        attention_output = self.attention_output(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.attention_layer_norm(hidden_states + attention_output)

        # 前馈网络
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.gelu(intermediate_output)

        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)

        # 前馈网络输出 + 残差连接 + LayerNorm
        layer_output = self.output_layer_norm(attention_output + layer_output)

        return layer_output


class SimpleTorchBERT(nn.Module):
    """简化的BERT模型，只包含单层Transformer"""

    def __init__(self, vocab_size=30522, hidden_size=768, num_attention_heads=12):
        super(SimpleTorchBERT, self).__init__()
        self.hidden_size = hidden_size

        # 词嵌入层
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)  # 最大512位置
        self.token_type_embeddings = nn.Embedding(2, hidden_size)  # 句子A和B

        # LayerNorm和Dropout
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

        # 单层Transformer
        self.transformer_layer = SingleLayerTransformerTorch(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads
        )

        # 池化层
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # 嵌入层
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # 词嵌入
        words_embeddings = self.word_embeddings(input_ids)

        # 位置嵌入
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        position_embeddings = self.position_embeddings(position_ids)

        # 句子类型嵌入
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 三个嵌入相加
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # 单层Transformer
        sequence_output = self.transformer_layer(embeddings, attention_mask)

        # 池化输出（取第一个token [CLS]）
        pooled_output = self.pooler(sequence_output[:, 0])
        pooled_output = self.tanh(pooled_output)

        return sequence_output, pooled_output


# 测试代码
if __name__ == "__main__":
    x = torch.LongTensor([[2450, 15486, 102, 2110]])  # 形状: [1, 4]

    # 初始化模型
    model = SimpleTorchBERT(vocab_size=21128, hidden_size=768, num_attention_heads=12)
    model.eval()

    # 前向传播
    with torch.no_grad():
        sequence_output, pooled_output = model(x)

    print(f"Sequence output shape: {sequence_output.shape}")  # 应该是 [1, 4, 768]
    print(f"Pooled output shape: {pooled_output.shape}")  # 应该是 [1, 768]

