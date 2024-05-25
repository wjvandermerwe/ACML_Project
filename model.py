import math
from torch import nn
import torch

class Embedding(nn.Module):
    # input and output tokens to vectors of dimension dmodel
    def __init__(self, size: int, vocab_size: int) -> object:
        super().__init__()
        self.size = size
        self.vocab_size = vocab_size
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embeddings = nn.Embedding(vocab_size, size)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.size)

class PositionalEncoding(nn.Module):
    def __init__(self, size: int, seq_len: int, dropout: float):
        super().__init__()
        self.size = size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pos_enc = torch.zeros(seq_len, size)

        # Create a tensor of shape (max_len, 1) for the positions
        # https://pytorch.org/docs/stable/generated/torch.arange.html#torch-arange
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # common denominator in the sinusoidal functions
        div_term = torch.exp(torch.arange(0, size, 2).float() * (-math.log(10000.0) / size))

        # odds and evens -> start even go forward 2, start odd go forward 2
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension and register the positional encoding matrix as a buffer
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        # for persistant storage -> saved as seperate state file
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encodings to the input embeddings
        return x + self.pos_enc[:x.size(0), :]

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=features, eps=eps)
    def forward(self, x):
        return self.layer_norm(x)

class FeedForwardComponent(nn.Module):
    def __init__(self, size: int, expansion: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(size, expansion)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expansion, size)
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, size: int, heads: int, dropout_rate: float):
        super().__init__()
        self.size = size
        self.heads = heads
        assert size % heads == 0, "vector sequences is divisble by the amount of heads"

        self.vec = size // heads
        self.query_weights = nn.Linear(size, size)
        self.key_weights = nn.Linear(size, size)
        self.value_weights = nn.Linear(size, size)

        self.output_weights = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask):

        # apply weights
        queries = self.query_weights(query)
        keys = self.key_weights(key)
        values = self.value_weights(value)

        queries = queries.view(queries.shape[0], queries.shape[1], self.heads, self.vec).transpose(1, 2)
        keys = keys.view(keys.shape[0], keys.shape[1], self.heads, self.vec).transpose(1, 2)
        values = values.view(values.shape[0], values.shape[1], self.heads, self.vec).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(queries, keys, values, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.vec)

        return self.output_weights(x)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        vec = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(vec)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, value), attention_scores


class SkipConnection(nn.Module):
    def __init__(self, features_size:int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features=features_size)
    def forward(self, x, connector):
        # apply the layer (norm and add)
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        return x + self.dropout(connector(self.norm(x)))

class MappingLayer(nn.Module):
    def __init__(self, size, vocab_size):
        super().__init__()
        self.proj = nn.Linear(size, vocab_size)
    def forward(self, x) -> None:
        return self.proj(x)

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardComponent, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.attention_skip = SkipConnection(features, dropout)
        self.feed_forward_skip = SkipConnection(features, dropout)
    def forward(self, x, src_mask):
        x = self.attention_skip(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.feed_forward_skip(x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, features_size: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features_size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features: int,
                 self_attention: MultiHeadAttention,
                 cross_attention: MultiHeadAttention,
                 feed_forward: FeedForwardComponent,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.self_attention_skip = SkipConnection(features, dropout)
        self.cross_attention_skip = SkipConnection(features, dropout)
        self.feed_forward_skip = SkipConnection(features, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.self_attention_skip(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.cross_attention_skip(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.feed_forward_skip(x, self.feed_forward)
        return x

class Decoder(nn.Module):

    def __init__(self, features_size: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features_size)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    # transformer as a torch module, for ease of use
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: Embedding, tgt_embed: Embedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: MappingLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, size: int = 512,
                      N: int = 6, heads: int = 8, dropout: float = 0.1, expanded: int = 2048) -> Transformer:

    # input and output embedding vectors with appended positional encodings
    src_embed = Embedding(size, src_vocab_size)
    tgt_embed = Embedding(size, tgt_vocab_size)
    src_pos = PositionalEncoding(size, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(size, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(size, heads, dropout)
        feed_forward_block = FeedForwardComponent(size, expanded, dropout)
        encoder_block = EncoderBlock(size, self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(size, heads, dropout)
        cross_attention = MultiHeadAttention(size, heads, dropout)
        feed_forward = FeedForwardComponent(size, expanded, dropout)
        decoder_block = DecoderBlock(size, self_attention, cross_attention,
                                     feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(size, nn.ModuleList(encoder_blocks))
    decoder = Decoder(size, nn.ModuleList(decoder_blocks))

    projection_layer = MappingLayer(size, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)


    for p in transformer.parameters():
        if p.dim() > 1:
            # https://pytorch.org/docs/stable/nn.init.html
            nn.init.xavier_uniform_(p)

    return transformer