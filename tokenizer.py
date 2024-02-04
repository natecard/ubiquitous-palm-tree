# import datasets as ds
import os
from collections import defaultdict
from tokenizers import normalizers, Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.processors import TemplateProcessing

# Inspect and load dataset for training
# Not using this data set for the initial training but will revert to this for future versions
# ds_builder = ds.load_dataset_builder("wikimedia/wikipedia", "20231101.en")
# dataset = ds.load_dataset("wikimedia/wikipedia", "20231101.en")
# Init tokenizer, using BPE
tokenizer = Tokenizer(BPE())
# Init normalizer, convert to lower case and strip all accents
normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
# Init pre_tokenizer remove whitespace only
pre_tokenizer = Whitespace()
# Init BPE trainer, include vocab size
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# dataset.set_format(
#     type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
# )


def file_exists(filepath):
    return os.path.isfile(filepath)


# Usage
print(file_exists("./training_data/wiki.test.raw"))  # Replace with your file path

if file_exists("./tokenizer-data/token-wiki.json"):
    tokenizer = Tokenizer.from_file("tokenizer-data/token-wiki.json")
    print("Tokenizer loaded")
else:
    files = [
        os.path.abspath(f"./training_data/wiki.{split}.raw")
        for split in ["test", "train", "valid"]
    ]
    # files = [f"../training_data/wiki.{split}.raw" for split in ["test", "train", "valid"]]
    tokenizer.train(files=files, trainer=trainer)
    # Save tokenizer
    tokenizer.save("tokenizer-data/token-wiki.json")

output = tokenizer.encode(sequence="Hello, y'all! How are you?")
print(output.tokens)
print(output.ids)

tokenizer.token_to_id("[SEP]")
# def tokenization(example):
#     return tokenizer(example["text"])
# dataset = dataset.map(tokenization, batched=True)

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

output2 = tokenizer.encode(sequence="Hello, y'all! How are you?")
print(f"Second: {output2.tokens}")
# word_freqs = defaultdict(int)
# for text in files:
#     words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
#         text
#     )
#     new_words = [word for word, offset in words_with_offsets]
#     for word in new_words:
#         word_freqs[word] += 1

# word_freqs

# alphabet = []
# for word in word_freqs.keys():
#     if word[0] not in alphabet:
#         alphabet.append(word[0])
#     for letter in word[1:]:
#         if f"##{letter}" not in alphabet:
#             alphabet.append(f"##{letter}")

# alphabet.sort()
# alphabet

# vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

# splits = {
#     word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
#     for word in word_freqs.keys()
# }


# def compute_pair_scores(splits):
#     letter_freqs = defaultdict(int)
#     pair_freqs = defaultdict(int)
#     for word, freq in word_freqs.items():
#         split = splits[word]
#         if len(split) == 1:
#             letter_freqs[split[0]] += freq
#             continue
#         for i in range(len(split) - 1):
#             pair = (split[i], split[i + 1])
#             letter_freqs[split[i]] += freq
#             pair_freqs[pair] += freq
#         letter_freqs[split[-1]] += freq

#     scores = {
#         pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
#         for pair, freq in pair_freqs.items()
#     }
#     return scores


# pair_scores = compute_pair_scores(splits)
# for i, key in enumerate(pair_scores.keys()):
#     print(f"{key}: {pair_scores[key]}")
#     if i >= 5:
#         break
# best_pair = ""
# max_score = None
# for pair, score in pair_scores.items():
#     if max_score is None or max_score < score:
#         best_pair = pair
#         max_score = score

# print(best_pair, max_score)


# def merge_pair(a, b, splits):
#     for word in word_freqs:
#         split = splits[word]
#         if len(split) == 1:
#             continue
#         i = 0
#         while i < len(split) - 1:
#             if split[i] == a and split[i + 1] == b:
#                 merge = a + b[2:] if b.startswith("##") else a + b
#                 split = split[:i] + [merge] + split[i + 2 :]
#             else:
#                 i += 1
#         splits[word] = split
#     return splits


# vocab_size = 90
# while len(vocab) < vocab_size:
#     scores = compute_pair_scores(splits)
#     best_pair, max_score = "", None
#     for pair, score in scores.items():
#         if max_score is None or max_score < score:
#             best_pair = pair
#             max_score = score
#     splits = merge_pair(*best_pair, splits)
#     new_token = (
#         best_pair[0] + best_pair[1][2:]
#         if best_pair[1].startswith("##")
#         else best_pair[0] + best_pair[1]
#     )
#     vocab.append(new_token)
