import torch
import math
class MultiHeadAttention():
    def __init__(self, W_Q, W_K, W_V, num_heads):
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V

    def attention(self, query, answer, value):
       
        pass