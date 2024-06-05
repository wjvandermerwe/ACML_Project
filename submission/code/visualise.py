import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import torch
from model import build_transformer
from config import get_config, get_weights_file_path
from train import load_and_preprocess
import altair as alt
import pandas as pd
from validation import greedy_decode

def load_model_from_weights(epoch = "15"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    _, val_dataloader, tokenizer_src, tokenizer_tgt = load_and_preprocess(config)
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], size=config['d_model']).to(device)
    model_filename = get_weights_file_path(config, epoch)
    state = torch.load(model_filename, map_location=device) # this is for loading the state files even on different devices than the trianing
    model.load_state_dict(state['model_state_dict'], strict=False)
    return val_dataloader, tokenizer_src, tokenizer_tgt, model, device, config


def load_next_batch(model, val_dataloader, tokenizer_src, tokenizer_tgt, device, config):

    batch = next(iter(val_dataloader))
    encoder_input = batch["encoder_input"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_mask = batch["decoder_mask"].to(device)

    encoder_input_tokens = [tokenizer_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [tokenizer_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    assert encoder_input.size(
        0) == 1, "Batch size must be 1 for validation"

    # populates attention scores
    model_out = greedy_decode(
        model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
    
    return batch, encoder_input_tokens, decoder_input_tokens, model_out

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

def attn_map(model, layer, head, row_tokens, col_tokens, max_sentence_len):

    df = mtx2df(
        model.decoder.layers[layer].self_attention.attention_scores[0, head].data,
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

def get_all_attention_maps(model, layers: list[int], heads: list[int], row_tokens: list, col_tokens, max_sentence_len: int):
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(attn_map(model, layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)


def run_attention_visual():
    val_dataloader, tokenizer_src, tokenizer_tgt, model, device, config= load_model_from_weights();
    batch, encoder_input_tokens, decoder_input_tokens = load_next_batch(model,val_dataloader, tokenizer_src, tokenizer_tgt, device,config)
    print(f'Source: {batch["src_text"][0]}')
    print(f'Target: {batch["tgt_text"][0]}')
    
    sentence_len = encoder_input_tokens.index("[PAD]")

    layers = [0, 1, 2]
    heads = [0, 1, 2, 3, 4, 5, 6, 7]
    charts = get_all_attention_maps(model, layers, heads, encoder_input_tokens, decoder_input_tokens, min(20, sentence_len))
    charts.save('vis/attention_maps.html')
    print("Saved attention maps to attention_maps.html")

def load_tensorboard_runs():
    log_path = 'runs/tmodel/tmodel'
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()
    return ea

def smooth_data(data, window_size=10):
    return data.rolling(window=window_size, min_periods=1).mean()

def export_tensorboard_metrics():
    ea = load_tensorboard_runs();
    for tag in ea.Tags()['scalars']:
        metrics = ea.Scalars(tag)
        df = pd.DataFrame([(x.step, x.value) for x in metrics], columns=['Step', 'Value'])
        # Apply smoothing
        df['Smoothed Value'] = smooth_data(df['Value'])

        # Plot the metric
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x='Step', y='Value')
        plt.title(f'{tag} Over Training Steps')
        plt.xlabel('Training Steps')
        plt.ylabel('Value')
        plt.grid(True)
        
        # Save the plot as an image
        image_filename = f'vis/{tag.replace("/", "_")}.png'
        plt.savefig(image_filename)
        plt.close()
        print(f"Saved BLEU score distribution plot as {image_filename}")


    
