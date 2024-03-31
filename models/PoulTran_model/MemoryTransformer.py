import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)

def create_padding_mask(seq):
    seq = (seq == 0)
    return seq.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # Upper triangular matrix of 0s and 1s

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(scaled_attention)
        return output, attention_weights

# Memory Multi-Head Attention
class MemoryMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_memory):
        super(MemoryMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.k_memory = nn.Parameter(torch.randn(1, num_memory, d_model))
        self.v_memory = nn.Parameter(torch.randn(1, num_memory, d_model))
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        k_memory = self.k_w(self.k_memory)
        v_memory = self.v_w(self.v_memory)
        k = torch.cat([k, k_memory.expand(k.size(0), -1, -1)], dim=1)
        v = torch.cat([v, v_memory.expand(v.size(0), -1, -1)], dim=1)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(scaled_attention)
        return output, attention_weights

class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MemoryMultiHeadAttention(d_model, num_heads, num_memory = 30)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output):
        attn1, _ = self.mha1(x, x, x, None)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)
        attn2, _ = self.mha2(enc_output, enc_output, out1, None)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3

# Memory Transformer
class MemoryTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, encoder_dim=2048, dropout=0.5):
        super(MemoryTransformer, self).__init__()
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.final_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(encoded_captions)

        decode_lengths = [c - 1 for c in caption_lengths]
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)
        for i in range(self.num_layers):
            embeddings = self.decoder_layers[i](embeddings, encoder_out)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            preds = self.final_layer(self.dropout(embeddings[:batch_size_t, t, :]))
            # preds = self.final_layer(self.dropout(embeddings[:, t, :]))
            # print("preds.shape: ", preds.shape)
            # print(len(predictions[:batch_size_t, t, :]))
            predictions[:batch_size_t, t, :] = preds
            # alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths

if __name__ == '__main__':

    # 模拟输入数据
    encoder_out = torch.randn(2, 1024)
    encoded_captions = torch.randint(1, 1000, (2, 13))  # 假设词汇表大小为1000
    caption_lengths = [13, 13, 10, 10, 10, 3, 3, 3]

    # 实例化模型
    num_layers = 2
    d_model = 1024
    num_heads = 8
    dff = 512
    vocab_size = 1000  # 假设词汇表大小为1000
    dropout = 0.1

    model = MemoryTransformer(num_layers, d_model, num_heads, dff, vocab_size, dropout=dropout)

    # 运行模型
    with torch.no_grad():
        model.eval()
        predictions, _, _ = model(encoder_out, encoded_captions, caption_lengths)

    # 打印输出
    print("Predictions shape:", predictions.shape)