from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# https://huggingface.co/docs/tokenizers/quicktour
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

# import torch
#
# def torch_get_or_build_tokenizer(config, ds, lang):
#     tokenizer_path = Path(config['tokenizer_file'].format(lang))
#     if not tokenizer_path.exists():
#         tokenizer = get_tokenizer('basic_english')
#         def yield_tokens(data_iter):
#             for _, text in data_iter:
#                 yield tokenizer(text)
#         vocab = build_vocab_from_iterator(yield_tokens(get_all_sentences(ds, lang)), specials=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
#         vocab.set_default_index(vocab["[UNK]"])
#         torch.save(vocab, tokenizer_path)
#     else:
#         vocab = torch.load(tokenizer_path)
#         tokenizer = get_tokenizer('basic_english')  # Ensure this matches the saved tokenizer type
#     return tokenizer, vocab


