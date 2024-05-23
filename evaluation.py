import argparse
import gzip
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from bot.block_state import BlockStateTransformer, RecurrentTrainerWrapper

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

def calculate_perplexity(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            inputs, labels = batch[:, :-1], batch[:, 1:]
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs.transpose(1, 2), labels, reduction='sum')
            total_loss += loss.item()
            total_count += labels.numel()
    return np.exp(total_loss / total_count)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = BlockStateTransformer(
        num_tokens=50257,  # Assuming GPT-2 tokenizer
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

    # Load datasets
    datasets = {
        "PG19": "./data/pg19.gz",
        "arXiv": "./data/arxiv.gz",
        "GitHub": "./data/github.gz",
    }

    results = {}
    for name, path in datasets.items():
        with gzip.open(path, 'rb') as file:
            data = np.frombuffer(file.read(), dtype=np.uint8).copy()
        dataset = TextDataset(torch.from_numpy(data), args.seq_len)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Calculate perplexity
        perplexity = calculate_perplexity(model, data_loader, device)
        results[name] = perplexity
        print(f"{name} Perplexity: {perplexity}")

    # Print and save results
    for dataset, perplexity in results.items():
        print(f"{dataset} Perplexity: {perplexity}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Transformer model on various datasets.')
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length for evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    args = parser.parse_args()
    main(args)
