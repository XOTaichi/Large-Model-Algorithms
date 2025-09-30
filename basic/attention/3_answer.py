import torch
import math

class MultiHeadAttention():
    def __init__(self, W_Q, W_K, W_V, num_heads):
        self.W_Q = W_Q  # shape: (d_model, d_model)
        self.W_K = W_K
        self.W_V = W_V
        self.num_heads = num_heads

    def attention(self, query, key, value):
        batch_size, seq_len, d_model = query.size()
        d_k = d_model // self.num_heads

        # 1. 线性变换
        Q = torch.matmul(query, self.W_Q)  # (batch, seq_len, d_model)
        K = torch.matmul(key, self.W_K)
        V = torch.matmul(value, self.W_V)

        # 2. 分头
        # (batch, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)  
        K = K.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(p_attn, V)  # (batch, num_heads, seq_len, d_k)

        # 4. 拼回原维度
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return out

'''
假设 d_model = 512，num_heads = 8，那么每个头的维度 d_k = d_model / num_heads = 64。
view(batch_size, seq_len, num_heads, d_k) 把原始的 d_model 拆分成 num_heads 个 d_k 维度，并保留 batch_size 和 seq_len。
结果的形状是 (batch_size, seq_len, num_heads, d_k)。

transpose(1, 2) 是把第二维和第三维交换位置。即将 seq_len 和 num_heads 互换，使得 num_heads 成为维度 1。
经过转置后，Q 的形状从 (batch_size, seq_len, num_heads, d_k) 变为 (batch_size, num_heads, seq_len, d_k)。
这样，num_heads 就变成了我们可以并行处理的维度。
'''