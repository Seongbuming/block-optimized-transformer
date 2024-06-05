import argparse
import gzip
import random
import torch.ao.quantization
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.quantization import default_qconfig, get_default_qconfig, prepare, convert
from torch.quantization.quantize_fx import prepare_fx, convert_fx

from accelerate import Accelerator
import bot.block_optimized as bot, bot.block_state as bst

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 250
GENERATE_LENGTH = 2048
SEQ_LEN = 2048
SAVE_MODEL_EVERY = 1000

# Argument parser
parser = argparse.ArgumentParser(description='Train a Transformer model.')
parser.add_argument('--model_name', type=str, default='bot', choices=['bot', 'bst'], help='The base name for the saved model files.')
args = parser.parse_args()

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# accelerator

accelerator = Accelerator()

device = accelerator.device
acc_print = accelerator.print

# instantiate palm

if args.model_name == 'bst':
    model = bst.BlockStateTransformer(
        num_tokens = 256,
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        max_seq_len = 512,
        block_width = 512,
        num_state_vectors = 512,
        recurrent_layers = (4,),
        use_flash_attn = True,
        s4_n_ssm=64,
    )

    train_wrapper = bst.RecurrentTrainerWrapper(
        model,
        xl_memories_dropout = 0.1,
        state_dropout = 0.1,
    )

    model.to(device)
elif args.model_name == 'bot':
    model = bot.BlockOptimizedTransformer(
        num_tokens = 256,
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        max_seq_len = 512,
        block_width = 512,
        num_state_vectors = 512,
        recurrent_layers = (4,),
        use_flash_attn = True,
    )

    train_wrapper = bot.RecurrentTrainerWrapper(
        model,
        xl_memories_dropout = 0.1,
        state_dropout = 0.1,
    )

    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    embedding_layer = model.token_emb
    model.token_emb = torch.nn.Identity()
    prepare(model, inplace=True)

    model.token_emb = embedding_layer
    convert(model, inplace=True)

    model.to(device)

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = train_wrapper(next(train_loader))
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    acc_print(f"training loss: {loss.item()}")
    accelerator.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = train_wrapper(next(val_loader))
            acc_print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        acc_print(f" prime: {prime}\n\n{'*' * 100}")

        sample = train_wrapper.generate(inp[None, ...], length = GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        acc_print(f" output_str: {output_str}\n\n")

    if i % SAVE_MODEL_EVERY == 0:
        torch.save(model.state_dict(), f"checkpoints/{args.model_name}_{i}.pt")
        acc_print(f"Model saved at iteration {i} as {args.model_name}_{i}.pt")

convert(model, inplace=True)
