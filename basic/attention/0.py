"Simple Attention Mechanism"
import torch

class SimpleAttention():
    def __init__(self):
        pass

    def attention(self, query, key, value):
        # TODO 计算注意力
        pass

def main():
    # 0. 初始化attention计算器
    attention_component = SimpleAttention()

    # 1. 定义 query、key、value
    # 假设：
    # batch_size = 2
    # seq_len = 2
    # d_k = 3
    # value 的维度 d_v = 2
    # batch_size=2, seq_len=2, d_k=3
    query = torch.tensor([
        [[1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]],

        [[0.5, 1.0, 0.5],
        [1.0, 0.0, 1.0]]
    ])  # shape: (2, 2, 3)

    key = torch.tensor([
        [[1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]],

        [[0.5, 1.0, 0.5],
        [1.0, 0.0, 1.0]]
    ])  # shape: (2, 2, 3)

    value = torch.tensor([
        [[1.0, 2.0],
        [3.0, 4.0]],

        [[2.0, 1.0],
        [0.0, 1.0]]
    ])  # shape: (2, 2, 2)

    # attention计算：每个词在考虑上下文后得到的加权表示
    result = attention_component.attention(query=query, key=key, value=value)

    print(result)