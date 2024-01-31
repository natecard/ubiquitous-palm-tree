import datasets as ds
from transformers import AutoTokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from collections import defaultdict

ds_builder = ds.load_dataset_builder('wikimedia/wikipedia', '20231101.simple')
dataset = ds.load_dataset('wikimedia/wikipedia', '20231101.simple')

normalizer = normalizers.Sequence([NFD(), StripAccents()])

tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
tokenizer(dataset['train'][-1]['text'])    

def tokenization(example):
    return tokenizer(example["text"])

dataset = dataset.map(tokenization, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])

word_freqs = defaultdict(int)
for text in dataset:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

word_freqs