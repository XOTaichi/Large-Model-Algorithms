import torch
import math
class MaskAttention():
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def attention(self, query, key, value):
        mask = torch.full((query.size(0), self.max_seq_len, self.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        key_trans = key.transpose(-2, -1)
        scores = torch.matmul(query, key_trans) / math.sqrt(query.size(-1))
        seq_len = query.size(1)
        mask = mask[:, :seq_len, :seq_len]
        scores_mask = scores + mask
        p_attn = scores_mask.softmax(dim=-1)
        output = torch.matmul(p_attn, value)
        return output

