import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):

        super().__init()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)    # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -math.log(10000.0) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)        # (1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6) -> None:            # eps is for numerical stability and avoid div by 0
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))        # multiplied
        self.bias = nn.Parameter(torch.zeros(1))        # added         # Q where to specify adam?

    def forward(self, x):
        mean = x.mean(dim = -1)
        std = x.std(dim = -1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)                # W_1 and b_1
        self.linear_2 = nn.Linear(d_ff, d_model)                # W_2 and b_2
        self.dropout = nn.Dropout(dropout)            

    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, seq _len, d_ff) --> (Batch, seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model / h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)              

        attention_scores = attention_scores.softmax(dim = -1)      # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores                 # second element is for visualization

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch , seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) 

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_model)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout), ResidualConnection(dropout)])


    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))  
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x))
        return x


class Encoder(nn.Module):
     
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)])

        def forward(self, x, encoder_output, enc_mask, dec_mask):        # src : encoder , target : decoder
            x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, dec_mask))
            x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, 
                                                                                     enc_mask))
            x = self.residual_connections[2](x, lambda x: self.feed_forward_block(x))
            return x
        
class Decoder(nn.Module):
    
    def __init(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, enc_mask, dec_mask):

        for layer in self.layers:
            x = layer(x, encoder_output, enc_mask, dec_mask)
        return self.norm(x)

