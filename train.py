import torch
import torch.nn as nn
import os
import requests
import pandas as pd


from config import Config
from model import GPT
from tokenizer import Tokenizer

config = Config()

input_file_path = os.path.join(os.path.dirname(__file__), 'data/training_data/shakespeare.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)
data = data[int(n*0.75):]

# Train test split
n = len(data)
train_data = data[:int(n*0.9)]
test_data = data[int(n*0.9):]

# Creating objects
tokenizer = Tokenizer()
tokenizer.load_tokenizer("tokenizer.json")

config.vocab_size = tokenizer.vocab_size

model = GPT(config)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()

# Encoding training and test data
enc_train_data = torch.tensor(tokenizer.encode(train_data))
enc_test_data = torch.tensor(tokenizer.encode(test_data))
train_data_len = len(enc_train_data)
test_data_len = len(enc_test_data)

# Training loop
losses = []

log = {
    "step": [],
    "train_loss_list": [],
    "test_loss_list": []
}

for step in range(config.steps):
    batch_tokens_x, batch_tokens_y = model.get_batch(
        train_data_len,
        config.batch_size,
        config.block_size,
        enc_train_data
    )

    logits = model.forward(batch_tokens_x)
    B,T,N = logits.shape
    logits = logits.reshape(B*T, -1)

    B, T = batch_tokens_y.shape
    targets = batch_tokens_y.reshape(B*T,)

    optimizer.zero_grad()

    loss = criterion(logits, targets)
    loss.backward()

    optimizer.step()
    losses.append(loss.item())

    if step % config.average_loss_every == 0:
        avg_loss = sum(losses) / len(losses)
        losses = []

        model.eval()
        test_batch_losses = []

        with torch.no_grad():
            for _ in range(5):
                test_tokens_x, test_tokens_y = model.get_batch(
                    test_data_len,
                    config.batch_size,
                    config.block_size,
                    enc_test_data
                )

                test_logits = model.forward(test_tokens_x)

                B, T, C = test_logits.shape
                test_logits = test_logits.reshape(B * T, C)

                test_targets = test_tokens_y.reshape(B * T)

                test_loss = criterion(test_logits, test_targets)
                test_batch_losses.append(test_loss.item())

        model.train()
        test_loss_avg = sum(test_batch_losses) / len(test_batch_losses)

        log["step"].append(step)
        log["train_loss_list"].append(avg_loss)
        log["test_loss_list"].append(test_loss_avg)


        print(f"steps: {step} train loss: {avg_loss:.4f} test loss: {test_loss_avg:.4f}")

# Saving model
torch.save(model.state_dict(), "gpt_model.pth")

df = pd.DataFrame(log)
df.to_csv("norm.csv")
























