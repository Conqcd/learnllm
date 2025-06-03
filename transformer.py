import torch
from torch import nn
from torch import functional

class FFN(nn.Module):
    def __init__ (self, vector_size):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(vector_size, vector_size)
        self.w2 = nn.Linear(vector_size, vector_size)
        self.w3 = nn.Linear(vector_size, vector_size)

    def forward(self,x):
        x = functional.F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * norm


class MultiHeadAttention(nn.Module):
    def __init__(self,n_head,kv_head,vector_size):
        super(MultiHeadAttention, self).__init__()
        self.q_m = nn.Linear(vector_size, vector_size)
        group_size = vector_size * kv_head / n_head
        self.k_m = nn.Linear(vector_size, group_size)
        self.v_m = nn.Linear(vector_size, group_size)
        self.o_m = nn.Linear(group_size, vector_size)
        self.n_head = n_head
        self.kv_head = kv_head

        #rope
        self.phi = 1 / ( 5000 ** torch.as_tensor(range(group_size)).float() / group_size)
    def rotate_q_k(self, q, k):
        length_size = torch.as_tensor(range(q.shape[1])).float() / q.shape[1]
        matrix = torch.outer(self.phi, length_size)
        q_complex = torch.view_as_complex(q.view(-1, 2))
        k_complex = torch.view_as_complex(k.view(-1, 2))
        q_rotated = q_complex * matrix
        k_rotated = k_complex * matrix
        q_rotated = torch.view_as_real(q_rotated).view(q.shape)
        k_rotated = torch.view_as_real(k_rotated).view(k.shape)
        return q_rotated, k_rotated

    def forward(self, x, y):
        q = self.q_m(x).view(-1, self.n_head)
        k = self.k_m(y).view(-1, self.kv_head)
        v = self.v_m(y).view(-1, self.kv_head)
        module = self.n_head / self.kv_head
        attention_arr = []
        for head in range(self.n_head):
            q_head = q[:, head]
            k_head = k[:, head // module]
            v_head = v[:, head // module]
            # Apply RoPE
            q_head, k_head = self.rotate_q_k(q_head, k_head)

            attn_weights = torch.matmul(q_head, k_head.transpose(-1, -2)) / (k_head.size(-1) ** 0.5)
            attn_weights = functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_head)
            attention_arr.append(attn_output)
        attention = torch.stack(attention_arr, dim=1)
        output = self.o_m(attention)
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_head, kv_head, vector_size):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(n_head, kv_head, vector_size)
        self.ffn = FFN(vector_size)
        self.rms_norm1 = RMSNorm(vector_size)
        self.rms_norm2 = RMSNorm(vector_size)

    def forward(self, x):
        x = self.rms_norm1(x)
        x = self.attention(x, x) + x
        x = self.rms_norm2(x)
        x = self.ffn(x) + x
        return x
class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_head, kv_head, vector_size):
        super(TransformerDecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(n_head, kv_head, vector_size)
        self.cross_attention = MultiHeadAttention(n_head, kv_head, vector_size)
        self.ffn = FFN(vector_size)
        self.rms_norm1 = RMSNorm(vector_size)
        self.rms_norm2 = RMSNorm(vector_size)
        self.rms_norm3 = RMSNorm(vector_size)

    def forward(self, x, y):
        x = self.rms_norm1(x)
        x = self.attention(x, x) + x
        x = self.rms_norm2(x)
        x = self.cross_attention(x, y) + x
        x = self.rms_norm3(x)
        x = self.ffn(x) + x
        return x
class TransformerEncoder(nn.Module):
    def __init__(self,n_block,n_head,kv_head,vector_size,token_size):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(token_size, vector_size)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(n_head=n_head, kv_head=kv_head, vector_size=vector_size) for _ in range(n_block)
        ])

    def forward(self,x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,n_block,n_head,kv_head,vector_size,token_size):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(token_size, vector_size)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(n_head=n_head, kv_head=kv_head, vector_size=vector_size) for _ in range(n_block)
        ])
        self.lm_head = nn.Linear(vector_size, token_size)
    def forward(self,x,y):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, y)
        x = self.lm_head(x)
        return x

class Transformer(nn.Module):
    def __init__(self, n_block,n_head,kv_head, vector_size=732, token_size=1000):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(n_block,n_head,kv_head, vector_size, token_size)
        self.decoder = TransformerDecoder(n_block,n_head,kv_head, vector_size, token_size)

    def forward(self, x, y):
        x = self.encoder(x)
        y = self.decoder(y, x)
        return y