import datasets as ds
from tokenizers import normalizers, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.processors import TemplateProcessing
from collections import defaultdict

# Inspect and load dataset for training
# Not using this data set for the initial training but will revert to thsi for future versions
# ds_builder = ds.load_dataset_builder("wikimedia/wikipedia", "20231101.en")
# dataset = ds.load_dataset("wikimedia/wikipedia", "20231101.en")
# Init normalizer, convert to lower case and strip all accents
normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
# Init pre_tokenizer remove whitespace only
pre_tokenizer = Whitespace()
# Init tokenizer, using WordPiece
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
# Init WordPiece trainer, include vocab size
trainer = WordPieceTrainer(
    vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
files = [
    f"./data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]
]
tokenizer.train(files, trainer)
tokenizer.save("data/token-wiki.json")

# tokenizer(dataset["train"][-1]["text"])
# def tokenization(example):
#     return tokenizer(example["text"])
# dataset = dataset.map(tokenization, batched=True)

# dataset.set_format(
#     type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
# )


tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

word_freqs = defaultdict(int)
for text in dataset:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text
    )
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs
