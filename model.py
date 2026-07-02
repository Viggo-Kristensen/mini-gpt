import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

config = Config()

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embd_dim = config.embd_dim
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.w_q = nn.Linear(self.embd_dim, self.head_size * self.num_heads, bias=False)
        self.w_k = nn.Linear(self.embd_dim, self.head_size * self.num_heads, bias=False)
        self.w_v = nn.Linear(self.embd_dim, self.head_size * self.num_heads, bias=False)
        self.proj = nn.Linear(self.head_size * self.num_heads, self.embd_dim)

        self.ln1 = nn.LayerNorm(self.embd_dim)
        self.ln2 = nn.LayerNorm(self.embd_dim)

        self.mlp1 = nn.Sequential(nn.Linear(self.embd_dim, 4 * self.embd_dim),
                                  nn.GELU(),
                                  nn.Linear(4 * self.embd_dim, self.embd_dim)
                                  )

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.mlp_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        h = self.ln1(x)

        q = self.w_q(h)
        q = q.view(B, T, self.num_heads, self.head_size)
        q = q.transpose(-2, -3)

        k = self.w_k(h)
        k = k.view(B, T, self.num_heads, self.head_size)
        k = k.transpose(-2, -3)

        wei = q @ k.transpose(-1, -2)
        wei /= (self.head_size ** 0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )
        wei = F.softmax(wei, dim=-1)

        wei = self.attn_dropout(wei)

        v = self.w_v(h)
        v = v.view(B, T, self.num_heads, self.head_size)
        v = v.transpose(-2, -3)

        out = wei @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, -1)
        out = self.proj(out)
        out = self.resid_dropout(out)
        x = x + out

        h = self.ln2(x)

        mlp_out = self.mlp1(h)
        mlp_out = self.mlp_dropout(mlp_out)
        return x + mlp_out


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_table = nn.Embedding(config.block_size, config.embd_dim)
        self.token_table = nn.Embedding(config.vocab_size, config.embd_dim)
        self.blocks = nn.Sequential(
            TransformerBlock(config),
            )
        self.linear_out = nn.Linear(config.embd_dim, config.vocab_size)


    def get_batch(self,data_len, batch_size, block_size, token_data):
        starts= torch.randint(low=0, high=data_len-block_size-1, size=(batch_size,))
        batch_indices_x = starts[:, None] + torch.arange(block_size) # (B, 1) + (block_size)

        batch_indices_y = batch_indices_x + 1
        batch_tokens_x = token_data[batch_indices_x]
        batch_tokens_y = token_data[batch_indices_y]

        return batch_tokens_x, batch_tokens_y

    def forward(self, batch_tokens_x):
        # batch_tokens_x (B, T)
        token_x = self.token_table(batch_tokens_x)

        B, T = batch_tokens_x.shape

        pos = torch.arange(T)
        pos = pos.unsqueeze(0).expand(B, T)

        pos_x = self.pos_table(pos)

        x = token_x + pos_x
        x = self.blocks(x)
        logits = self.linear_out(x)
        return logits

    def generate(self, tokenized_prompt, response_length):
        answer = []
        for i in range(response_length):
            prompt_token_array = torch.tensor(tokenized_prompt).unsqueeze(0)
            prompt_pos_array = torch.arange(len(tokenized_prompt)).unsqueeze(0)

            token_x = self.token_table(prompt_token_array)
            pos_x = self.pos_table(prompt_pos_array)

            x = token_x + pos_x

            x = self.blocks(x)

            logits = self.linear_out(x)
            logits_last = logits[0, -1]
            probs = F.softmax(logits_last, dim=-1)

            generated_token = torch.multinomial(probs, num_samples=1).item()
            answer.append(generated_token)
            tokenized_prompt.append(generated_token)
            tokenized_prompt = tokenized_prompt[1:]

        return answer
