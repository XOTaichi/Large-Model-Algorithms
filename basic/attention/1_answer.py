import torch
import math
class SelfAttention():
    def __init__(self, W_Q, W_K, W_V):
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V

    def attention(self, x):
        query = self.W_Q * x
        key = self.W_K * x
        value = self.W_V * x
        d_k = query.size(-1)
        key_trans = key.transpose(-2, -1) 
        scores = torch.matmul(query, key_trans) / math.sqrt(d_k)
        atten = scores.softmax(dim=-1)
        return torch.matmul(atten, value)
'''
Self-Attention = “我自己问自己，看看序列中的每个词应该关注序列中哪些词，得到融合后的表示”
'''