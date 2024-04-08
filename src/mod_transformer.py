import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional encoding definition

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.transpose(0, 1)
    def forward(self, x):
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, seq_len, d_k) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d_k = d_k

        self.W_q = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_k = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_v = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_o = nn.Linear(num_heads * num_hidden, num_hidden)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.mask = self.get_mask(self.seq_len)
    
    def get_mask(self, size):
        device = next(self.parameters()).device
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=0)  
        return mask.unsqueeze(0).unsqueeze(0)  

    def forward(self, query, key, values, dropout=0.1, mask=None):
        # Reshaping expanded to n_heads
        seq_len, num_hidden = query.shape[1], query.shape[2]
        query = self.W_q(query).view(-1, self.num_heads, seq_len, num_hidden)
        key = self.W_k(key).view(-1, self.num_heads, seq_len, num_hidden)
        values = self.W_v(values).view(-1, self.num_heads, seq_len, num_hidden)

        # Q * K_T
        QK_T = torch.matmul(query,  key.mT)

        # QK_T / sqrt(dk)
        QK_T = QK_T / math.sqrt(self.d_k)

        if mask:
            self.mask = self.mask.to(query.device)
            QK_T = QK_T.masked_fill(self.mask == 1, float('-inf'))

        # softmax(QK_T / sqrt(d_k)
        attention_scores = self.softmax(QK_T)
        
        #dropout
        attention_scores = self.dropout(attention_scores)
        output = torch.matmul(attention_scores, values)  
        # Reshape and apply output linear layer  
        output = output.transpose(1, 2).contiguous().view(-1, seq_len, self.num_heads * num_hidden)  
        output = self.W_o(output)  
          
        return output  

# Feed forward definition
    
class FeedForward(nn.Module):
    def __init__(self, num_hidden, num_ffn_hidden) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_ffn_hidden = num_ffn_hidden

        self.W_1 = nn.Linear(num_hidden, num_ffn_hidden)
        self.W_2 = nn.Linear(num_ffn_hidden, num_hidden)

    def forward(self, x):
        return self.W_2(F.relu(self.W_1(x)))

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, n_heads, seq_len, num_hidden) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.decoders = nn.ModuleList([TransformerDecoderLayer(num_hidden, n_heads, seq_len) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.decoders:
            x = layer(x)
        return x
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, num_hidden, num_heads, seq_len) -> None:
        super().__init__()
        topk_seq_len = seq_len // 2
        self.multihead_attention_masked = MultiHeadAttention(num_hidden=num_hidden, num_heads=num_heads, seq_len=topk_seq_len, d_k=1)
        self.multihead_attention = MultiHeadAttention(num_hidden=num_hidden, num_heads=num_heads, seq_len=topk_seq_len, d_k=1)
        
        self.feed_forward = FeedForward(num_hidden=num_hidden, num_ffn_hidden= 2 * num_hidden)
        self.layer_norm1 = nn.LayerNorm(num_hidden)
        self.layer_norm2 = nn.LayerNorm(num_hidden)
        self.layer_norm3 = nn.LayerNorm(num_hidden)
        self.router = Router(top_k=seq_len//2, num_hidden=num_hidden)

    
    def forward(self, x_in):
        # basically route some tokens in or around the att + lin + norm + feedforward block

        #get topk 
        x_topk, x_topk_indices = self.router(x_in)

        # masked attention
        x_topk_att = self.multihead_attention_masked(x_topk, x_topk, x_topk, mask=True)
        #add and norm
        x_topk = x_topk + x_topk_att
        x_topk = self.layer_norm1(x_topk)

        # attention
        x_topk_att = self.multihead_attention(x_topk, x_topk, x_topk)

        #add and norm
        x_topk = x_topk + x_topk_att
        x_topk = self.layer_norm2(x_topk)

        #feed forward
        x_topk_forward = self.feed_forward(x_topk)

        #add and norm
        x_topk = x_topk_forward + x_topk

        #now you want to insert the tokens back into the original sequence, index by x_topk_indices 
        x = x_in.scatter_(1, x_topk_indices.expand(-1, -1, x_in.size(-1)), x_topk)

        x = self.layer_norm3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, decoder_layers_num, num_hidden, num_heads, seq_len, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.decoder = TransformerDecoder(decoder_layers_num, num_heads, seq_len, num_hidden)
        self.pos_enc = PositionalEncoding(embedding_dim, max_len=seq_len)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #embeddings
        x = self.embedding(x)

        #pos encodings
        x = self.pos_enc(x)

        #forward pass
        dec_output = self.decoder(x)
        output = self.linear(dec_output)

        return output



class Router(nn.Module):
    def __init__(self, top_k, num_hidden):
        super().__init__()
        self.linear = nn.Linear(num_hidden, 1)
        self.top_k = top_k

    def forward(self, x):
        #get scores for each token
        scores = self.linear(x)
        #get top k tokens 
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=1)
        #get topk from x
        x_top_k = x.gather(1, top_k_indices.expand(-1, -1, x.size(-1)))

        return x_top_k, top_k_indices