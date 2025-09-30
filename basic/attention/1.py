import torch
import math
class SelfAttention():
    def __init__(self, W_Q, W_K, W_V):
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V

    def attention(self, x):
        # TODO
        pass