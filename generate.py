import torch

from model import GPT
from config import Config
from tokenizer import Tokenizer

config = Config()

tokenizer = Tokenizer()
tokenizer.load_tokenizer("tokenizer.json")

config.vocab_size = tokenizer.vocab_size

model = GPT(config)

model.load_state_dict(torch.load("gpt_model.pth"))

# Generate answer to prompt
response_length = 100
prompt = "hello my name is"
tokenized_prompt = tokenizer.encode(prompt)
answer = model.generate(tokenized_prompt, response_length)

print(tokenizer.decode(answer))
