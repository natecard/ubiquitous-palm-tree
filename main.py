"""
Testing through using either deepseek-coder the llama 2 7b 
or mistral 7b model 
Description: This is a simple example of how to use the DeepSeek AI
API to generate code using the DeepSeek AI API.
"""

from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    # choosing either this model, llama 2 7b, or mistral 7b
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    trust_remote_code=True,
)
# choosing either this model, llama 2 7b, or mistral 7b
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    trust_remote_code=True,
    device_map="auto",
)
# load_in_4bit=True,
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    trust_remote_code=True,
)

prompt = "Write python code to reverse a string"

sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    temperature=0.8,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
