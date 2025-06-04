import torch
from torch import nn
from torch import functional

class FFN(nn.Module):
    def __init__ (self, vector_size, hidden_size):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(vector_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, vector_size)
        self.w3 = nn.Linear(vector_size, hidden_size)

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
        group_size = int(vector_size * kv_head / n_head)
        self.k_m = nn.Linear(vector_size, group_size)
        self.v_m = nn.Linear(vector_size, group_size)
        self.o_m = nn.Linear(vector_size, vector_size)
        self.n_head = n_head
        self.kv_head = kv_head

        group_vector_size = int(vector_size / n_head / 2)
        #rope
        self.phi = 1 / (5000 ** (torch.as_tensor(range(group_vector_size)).float() / group_vector_size))
    def rotate_q_k(self, q, k):
        length_size = torch.arange(q.shape[1])
        matrix = torch.outer(self.phi, length_size)
        matrix = torch.polar(torch.ones_like(matrix), matrix)
        q_complex = torch.view_as_complex(q.view(q.shape[0],q.shape[1],-1, 2))
        k_complex = torch.view_as_complex(k.view(k.shape[0],k.shape[1],-1, 2))
        q_rotated = q_complex * matrix.T
        k_rotated = k_complex * matrix.T
        q_rotated = torch.view_as_real(q_rotated).view(q.shape)
        k_rotated = torch.view_as_real(k_rotated).view(k.shape)
        return q_rotated, k_rotated

    def forward(self, x, y):
        q = self.q_m(x).view(x.shape[0],x.shape[1], self.n_head,-1)
        k = self.k_m(y).view(y.shape[0],y.shape[1], self.kv_head,-1)
        v = self.v_m(y).view(x.shape[0],x.shape[1], self.kv_head,-1)
        module = int(self.n_head / self.kv_head)
        attention_arr = []
        for head in range(self.n_head):
            q_head = q[:,:, head]
            k_head = k[:,:, head // module]
            v_head = v[:,:, head // module]
            # Apply RoPE
            q_head, k_head = self.rotate_q_k(q_head, k_head)

            # mask
            attn_weights = torch.matmul(q_head, k_head.transpose(-1, -2)) / k_head.shape[-1] ** 0.5
            mask_matrix = torch.full(attn_weights.shape, float("-inf"))
            mask_matrix = torch.triu(mask_matrix, diagonal=1)
            attn_weights = attn_weights + mask_matrix

            attn_weights = functional.F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_head)
            attention_arr.append(attn_output)
        attention = torch.cat(attention_arr, dim=-1)
        output = self.o_m(attention)
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_head, kv_head, vector_size, hidden_size):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(n_head, kv_head, vector_size)
        self.ffn = FFN(vector_size,hidden_size)
        self.rms_norm1 = RMSNorm(vector_size)
        self.rms_norm2 = RMSNorm(vector_size)

    def forward(self, x):
        x = self.rms_norm1(x)
        x = self.attention(x, x) + x
        x = self.rms_norm2(x)
        x = self.ffn(x) + x
        return x
class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_head, kv_head, vector_size, hidden_size):
        super(TransformerDecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(n_head, kv_head, vector_size)
        self.cross_attention = MultiHeadAttention(n_head, kv_head, vector_size)
        self.ffn = FFN(vector_size, hidden_size)
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
    def __init__(self,n_block,n_head,kv_head,vector_size,token_size, hidden_size):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(token_size, vector_size)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(n_head=n_head, kv_head=kv_head, vector_size=vector_size, hidden_size=hidden_size) for _ in range(n_block)
        ])

    def forward(self,x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,n_block,n_head,kv_head,vector_size,token_size, hidden_size):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(token_size, vector_size)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(n_head=n_head, kv_head=kv_head, vector_size=vector_size, hidden_size=hidden_size) for _ in range(n_block)
        ])
        self.rms_norm = RMSNorm(vector_size)
        self.lm_head = nn.Linear(vector_size, token_size)
    def forward(self,x,y):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, y)
        x = self.rms_norm(x)
        x = self.lm_head(x)
        return x

class Transformer(nn.Module):
    def __init__(self, n_block,n_head,kv_head, hidden_size, vector_size=732, token_size=1000):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(n_block,n_head,kv_head, vector_size, token_size, hidden_size)
        self.decoder = TransformerDecoder(n_block,n_head,kv_head, vector_size, token_size, hidden_size)

    def forward(self, x, y):
        x = self.encoder(x)
        y = self.decoder(y, x)
        return y


if __name__ == "__main__":
    # Example usage
    n_block = 6
    n_head = 16
    kv_head = 8
    hidden_size = 1024
    vector_size = 736
    token_size = 10000

    model = Transformer(n_block, n_head, kv_head, hidden_size, vector_size, token_size)

    # Dummy input
    x = torch.randint(0, token_size, (10, 20))  # Batch of 10 sequences of length 20
    y = torch.randint(0, token_size, (10, 20))  # Batch of 10 sequences of length 20

    output = model(x, y)
    print(output.shape)  # Should be (10, 20, token_size)