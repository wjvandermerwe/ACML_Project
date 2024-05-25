import torch
import torch.nn as nn
from model import Transformer, build_transformer
from config import get_config, get_weights_file_path
from train import load_and_preprocess
import altair as alt
import pandas as pd
import numpy as np
import warnings

from validation import greedy_decode
warnings.filterwarnings("ignore")


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = load_and_preprocess(config)
model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], size=config['d_model']).to(device)

# Load the pretrained weights
model_filename = get_weights_file_path(config, f"29")
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])


def load_next_batch():
    # Load a sample batch from the validation set
    batch = next(iter(val_dataloader))
    encoder_input = batch["encoder_input"].to(device)
    # encoder_mask = batch["encoder_mask"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    # decoder_mask = batch["decoder_mask"].to(device)

    encoder_input_tokens = [tokenizer_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [tokenizer_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    # check that the batch size is 1
    assert encoder_input.size(
        0) == 1, "Batch size must be 1 for validation"

    # model_out = greedy_decode(
    #     model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
    
    return batch, encoder_input_tokens, decoder_input_tokens

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def attn_map(layer, head, row_tokens, col_tokens, max_sentence_len):
    df = mtx2df(
        model.decoder.layers[layer].cross_attention_block.attention_scores[0, head].data,
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        #.title(f"Layer {layer} Head {head}")
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )

def get_all_attention_maps(layers: list[int], heads: list[int], row_tokens: list, col_tokens, max_sentence_len: int):
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(attn_map(layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)

batch, encoder_input_tokens, decoder_input_tokens = load_next_batch()
print(f'Source: {batch["src_text"][0]}')
print(f'Target: {batch["tgt_text"][0]}')
sentence_len = encoder_input_tokens.index("[PAD]")

layers = [0, 1, 2]
heads = [0, 1, 2, 3, 4, 5, 6, 7]

# Encoder Self-Attention
get_all_attention_maps(layers, heads, encoder_input_tokens, encoder_input_tokens, min(20, sentence_len))
