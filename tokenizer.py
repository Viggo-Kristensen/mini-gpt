import json
from config import Config

# Load the shakespeare dataset
with open("data/training_data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

class Tokenizer:
    def __init__(self, data=None):
        self.vocab = []
        self.vocab_size = 0
        self.enc_dict = {}
        self.dec_dict = {}

        if data is not None:
            self.vocab = self.init_vocab(data)
            self.vocab_size = len(self.vocab)
            self.enc_dict = self.init_enc_dict()
            self.dec_dict = self.init_dec_dict()

    def init_vocab(self, data):
        vocab = sorted(list(set(data)))
        return vocab

    def init_enc_dict(self):
        enc_dict = {}
        for i, c in enumerate(self.vocab):
            enc_dict[c] = i
        return enc_dict

    def init_dec_dict(self):
        dec_dict = {}
        for i, c in enumerate(self.vocab):
            dec_dict[i] = c
        return dec_dict

    def encode(self, text):
        return [self.enc_dict[c] for c in text]

    def decode(self, tokens):
        return ''.join([self.dec_dict[token] for token in tokens])

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                "vocab": self.vocab,
                "vocab_size": self.vocab_size
            }, f)

    def load_tokenizer(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.vocab_size = data["vocab_size"]
        self.enc_dict = self.init_enc_dict()
        self.dec_dict = self.init_dec_dict()

config = Config()

tokenizer = Tokenizer(text)
tokenizer.save("tokenizer.json")

config.vocab_size = tokenizer.vocab_size
