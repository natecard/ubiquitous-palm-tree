import os.path as path
from tokenizers import normalizers, Tokenizer, BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.processors import TemplateProcessing

# Inspect and load dataset for training
# Not using this data set for the initial training but will revert to this for future versions
# ds_builder = ds.load_dataset_builder("wikimedia/wikipedia", "20231101.en")
# dataset = ds.load_dataset("wikimedia/wikipedia", "20231101.en")
# Init tokenizer
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
    return path.isfile(filepath)


# Usage
print(file_exists("./training_data/wiki.test.raw"))  # Replace with your file path

if file_exists("./tokenizer-data/token-wiki.json"):
    tokenizer = Tokenizer.from_file("tokenizer-data/token-wiki.json")
    print("Tokenizer loaded")
else:
    files = [
        path.abspath(f"./training_data/wiki.{split}.raw")
        for split in ["test", "train", "valid"]
    ]
    # files = [f"../training_data/wiki.{split}.raw" for split in ["test", "train", "valid"]]
    tokenizer.train(files=files, trainer=trainer)
    # Save tokenizer
    tokenizer.save("tokenizer-data/token-wiki.json")

output = tokenizer.encode(sequence="Hello, y'all! How are you?")
print(output.tokens)
print(output.ids)


output = tokenizer.encode(text="Hello, y'all! How are you?")
print(output)

tokenizer.post_processor = TemplateProcessing(
    # single string template
    single="[CLS] $A [SEP]",
    # sentence pair template
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)
