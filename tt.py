import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 512
        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def self_attention(self, query, key, value,
                       mask=None, hidden_size=None,
                       project_size=None, dropout=None):
        assert hidden_size is not None
        assert project_size is not None
        # 对于Q, K, V 的 shape 都是 (batch, seq_len, hidden_dim)
        # todo:: 了解 hidden_size, and project_size
        key, query, value = self.get_key_query_embedding(
            key, query, value,
            hidden_size=hidden_size,
            project_size=project_size)
        # Compute 'Scaled Dot Product Attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 把要 mask，也就是说 mask 值为 0 的，变成负无穷
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # return torch.matmul(p_attn, value), p_attn
        return torch.matmul(p_attn, value)

    def get_key_query_embedding(self, key, query, value, hidden_size, project_size):
        # 对 key, query, value 三个指进行映射
        key = self.key_layer(key)
        query = self.query_layer(query)
        value = self.value_layer(value)
        return key, query, value


def main():
    a = SelfAttention()
    q = torch.rand(1, 14, 512)
    r = a.self_attention(q, q, q, hidden_size=512, project_size=256)
    print(r.shape)
    print(q.shape)
    print(r == q)


label2idx = {'C_-4': 0, 'C_0': 1, 'C_2': 2, 'C_-3': 3, 'O': 4,
     'C_1': 5, 'C_-2': 6, 'C_4': 7, 'C_3': 8, 'C_-1': 9}

label2idx = {
    'O': 0,
    'C_0': 1,
    'C_1': 2,
    'C_-1': 3,
    'C_2': 4,
    'C_-2': 5,
    'C_3': 6,
    'C_-3': 7,
    'C_4': 8,
    'C_-4': 9,
    'C_5': 10,
    'C_-5': 11,
    'C_6': 12,
    'C_-6': 13
}

if __name__ == '__main__':
    main()
