import argparse
import gzip
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from bot.block_state import BlockStateTransformer, RecurrentTrainerWrapper

# Argument parser
parser = argparse.ArgumentParser(description='Evaluate a Transformer model.')
parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the model checkpoint to be evaluated.')
args = parser.parse_args()

# accelerator
accelerator = Accelerator()
device = accelerator.device
acc_print = accelerator.print

# prepare enwik8 data
with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    _, np_valid = np.split(data, [int(90e6)])
    data_val = torch.from_numpy(np_valid)

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

SEQ_LEN = 2048
BATCH_SIZE = 4

val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load model
model = BlockStateTransformer(
    num_tokens=256,
    dim=512,
    depth=6,
    dim_head=64,
    heads=8,
    max_seq_len=512,
    block_width=512,
    num_state_vectors=512,
    recurrent_layers=(4,),
    use_flash_attn=True,
    s4_n_ssm=64,
)

model.load_state_dict(torch.load(args.model_checkpoint))
model.to(device)

train_wrapper = RecurrentTrainerWrapper(
    model,
    xl_memories_dropout=0.1,
    state_dropout=0.1,
)

model.eval()

# Evaluation function
def evaluate(model, data_loader, device):
    total_loss = 0.0
    total_tokens = 0
    total_bits = 0.0

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[:, :-1], batch[:, 1:]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item() * labels.size(0)
            total_tokens += labels.size(0)
            
            # Calculate Bits per Character (BPC)
            log_probs = F.log_softmax(outputs, dim=-1)
            bits = -log_probs.gather(2, labels.unsqueeze(-1)).sum()
            total_bits += bits.item()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    bpc = total_bits / total_tokens / np.log(2)
    
    return avg_loss, perplexity, bpc

# Perform evaluation
avg_loss, perplexity, bpc = evaluate(train_wrapper, val_loader, device)
acc_print(f"Validation Loss: {avg_loss}")
acc_print(f"Perplexity: {perplexity}")
acc_print(f"Bits per Character (BPC): {bpc}")
