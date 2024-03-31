# SwinTransformer模型结构，用于Encoder的部分
import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.softmax = nn.Softmax(dim = 1)
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)
        # alpha = self.softmax(attn)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)

class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        # mlp_head做了部分修改，以适应新的任务
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, 2048)
        )

    def forward(self, img):
        x = self.stage1(img)

        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])

        return self.mlp_head(x)
if __name__ == '__main__':

    # 测试案例
    net = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=3,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
    )

    dummy_x = torch.randn(1, 3, 224, 224)
    logits = net(dummy_x)

    print("模型输出形状:", logits.shape)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Attention(nn.Module):
#     def __init__(self, encoder_dim, decoder_dim, attention_dim):
#         super(Attention, self).__init__()
#         self.encoder_att = nn.Linear(encoder_dim, attention_dim)
#         self.decoder_att = nn.Linear(decoder_dim, attention_dim)
#         self.full_att = nn.Linear(attention_dim, 1)
#
#     def forward(self, encoder_out, decoder_hidden):
#         att1 = self.encoder_att(encoder_out)
#         att2 = self.decoder_att(decoder_hidden)
#         att = self.full_att(torch.tanh(att1 + att2.unsqueeze(1))).squeeze(2)
#         alpha = F.softmax(att, dim=1)
#         attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
#         return attention_weighted_encoding, alpha
#
#
# class TransformerDecoderLayer(nn.Module):
#     def __init__(self, embed_dim, decoder_dim, attention_dim, dropout=0.5):
#         super(TransformerDecoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=dropout)
#         self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=dropout)
#         self.linear1 = nn.Linear(embed_dim, decoder_dim)
#         self.linear2 = nn.Linear(decoder_dim, embed_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.norm3 = nn.LayerNorm(embed_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#
#     def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
#         tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#         tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#         tgt2 = self.linear2(F.relu(self.linear1(tgt)))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt
#
#
# class TransformerDecoder(nn.Module):
#     def __init__(self, vocab_size, embed_dim, decoder_dim, attention_dim, num_layers=6, dropout=0.5):
#         super(TransformerDecoder, self).__init__()
#         self.embed_dim = embed_dim
#         self.decoder_dim = decoder_dim
#         self.vocab_size = vocab_size
#         self.num_layers = num_layers
#
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.layers = nn.ModuleList([TransformerDecoderLayer(embed_dim, decoder_dim, attention_dim, dropout) for _ in range(num_layers)])
#         self.fc = nn.Linear(embed_dim, vocab_size)
#
#     def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
#         tgt = self.embedding(tgt) * math.sqrt(self.embed_dim)
#         tgt = self.dropout(tgt)
#
#         for layer in self.layers:
#             tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
#
#         tgt = self.fc(tgt)
#         return tgt
#
#
# class Transformer(nn.Module):
#     def __init__(self, encoder_dim, decoder_dim, embed_dim, vocab_size, attention_dim, num_layers=6, dropout=0.5):
#         super(Transformer, self).__init__()
#         self.encoder_dim = encoder_dim
#         self.decoder_dim = decoder_dim
#         self.embed_dim = embed_dim
#         self.vocab_size = vocab_size
#         self.attention_dim = attention_dim
#         self.num_layers = num_layers
#
#         self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
#         self.decoder = TransformerDecoder(vocab_size, embed_dim, decoder_dim, attention_dim, num_layers, dropout)
#
#     def forward(self, encoder_out, encoded_captions, caption_lengths):
#         batch_size = encoder_out.size(0)
#         encoder_dim = encoder_out.size(-1)
#         vocab_size = self.vocab_size
#
#         encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
#         num_pixels = encoder_out.size(1)
#
#         embeddings = self.embedding(encoded_captions)
#
#         h, c = self.init_hidden_state(encoder_out)
#
#         decode_lengths = [c - 1 for c in caption_lengths]
#
#         predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
#         alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
#
#         for t in range(max(decode_lengths)):
#             batch_size_t = sum([l > t for l in decode_lengths])
#             attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
#             gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
#             attention_weighted_encoding = gate * attention_weighted_encoding
#             h, c = self.decode_step(
#                 torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
#                 (h[:batch_size_t], c[:batch_size_t]))
#             preds = self.fc(self.dropout_layer(h))
#             predictions[:batch_size_t, t, :] = preds
#             alphas[:batch_size_t, t, :] = alpha
#
#         return predictions, encoded_captions, decode_lengths, alphas
#
#
#
# class TextDecoder(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(TextDecoder, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, vocab_size)
#
#     def forward(self, features, captions):
#         captions = captions[:, :-1]  # remove <end> token
#         embeddings = self.embed(captions)
#         print(features.unsqueeze(1).shape)
#         print(embeddings.shape)
#         inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
#         hiddens, _ = self.lstm(inputs)
#         outputs = self.linear(hiddens)
#         return outputs
#
#
# # 示例用法
# embed_size = 256
# hidden_size = 512
# vocab_size = 1000
# num_layers = 1
#
# decoder = TextDecoder(embed_size, hidden_size, vocab_size, num_layers)
#
# # 假设图像编码器输出和captions
# image_features = torch.randn(2, 14, 14, 2048)  # 假设的图像编码器输出
# captions = torch.randint(0, vocab_size, (4, 13))  # 假设的captions
#
# outputs = decoder(image_features, captions)
# print(outputs.size())  # 应为 [2, 13, vocab_size]，表示两个样本的13个词的概率分布

