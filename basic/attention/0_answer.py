import torch
from math import sqrt
class SimpleAttention():
    def __init__(self):
        pass

    def attention(self, query, key, value):
        # TODO 计算注意力
        d_k = query.size(-1) 
        key_transpose = key.transpose(-2, -1) 
        scores = torch.matmul(query, key_transpose) / sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        return torch.matmul(p_attn, value)

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

    result = attention_component.attention(query=query, key=key, value=value)

    print(result)

if __name__ == "__main__":
    main()
'''
1. key.transpose(-2, -1) 是干什么的？

torch.transpose() 是用来交换张量的两个维度。在这个函数中，-2 和 -1 是表示从 最后两个维度 的位置索引。
在 注意力机制 中，query 是形状 (batch_size, seq_len, d_k)，key 也是类似的形状 (batch_size, seq_len, d_k)。
为了计算它们的点积，query 的 最后一个维度 （d_k）需要与 key 的 倒数第二个维度（seq_len）对齐。


2. softmax(dim=-1) 是什么意思？
dim=-1 表示在 最后一个维度 上应用 softmax。这通常适用于对**每一行**的元素进行归一化（标准化），尤其是在处理多维张量时。
这里的 “最后一个维度” 对应的是 注意力得分 的每一行，即每个查询的权重分布。
输出 p_attn 的形状仍然是 (batch_size, seq_len, seq_len)，但是每一行的元素（即某个查询的权重分布）都会经过 softmax 转换
'''