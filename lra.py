import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from bot.block_state import BlockStateTransformer, RecurrentTrainerWrapper
from datasets import load_dataset

# Argument parser
parser = argparse.ArgumentParser(description='Evaluate a Transformer model on LRA benchmark.')
parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the model checkpoint to be evaluated.')
parser.add_argument('--task', type=str, required=True, choices=['listops', 'text', 'retrieval', 'image', 'pathfinder', 'pathx'], help='The LRA task to evaluate on.')
args = parser.parse_args()

# accelerator
accelerator = Accelerator()
device = accelerator.device
acc_print = accelerator.print

# Task-specific dataset preparation
def prepare_dataset(task):
    if task == 'listops':
        dataset = load_dataset('lhoestq/lra', 'listops')
    elif task == 'text':
        dataset = load_dataset('lhoestq/lra', 'imdb_reviews')
    elif task == 'retrieval':
        dataset = load_dataset('lhoestq/lra', 'retrieval')
    elif task == 'image':
        dataset = load_dataset('lhoestq/lra', 'cifar10')
    elif task == 'pathfinder':
        dataset = load_dataset('lhoestq/lra', 'pathfinder')
    elif task == 'pathx':
        dataset = load_dataset('lhoestq/lra', 'pathx')
    return dataset

# Tokenizer and padding for text and listops tasks
def tokenize_function(example):
    return {'input_ids': list(map(ord, example['text']))[:2048]}  # truncate to 2048 tokens

def pad_to_max_length(input_ids, max_length=2048):
    return input_ids + [0] * (max_length - len(input_ids))

class LRADataset(Dataset):
    def __init__(self, dataset, task):
        self.dataset = dataset
        self.task = task
        self.max_length = 2048

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.task in ['listops', 'text']:
            input_ids = pad_to_max_length(item['input_ids'], self.max_length)
        else:
            input_ids = item['image'].flatten().tolist()[:self.max_length]
            input_ids = pad_to_max_length(input_ids, self.max_length)
        label = item['label']
        return torch.tensor(input_ids), torch.tensor(label)

    def __len__(self):
        return len(self.dataset)

dataset = prepare_dataset(args.task)
train_dataset = dataset['train']
test_dataset = dataset['test']

if args.task in ['listops', 'text']:
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = LRADataset(train_dataset, args.task)
test_dataset = LRADataset(test_dataset, args.task)

BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

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
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    return accuracy

# Perform evaluation
test_accuracy = evaluate(train_wrapper, test_loader, device)
acc_print(f"Test Accuracy on {args.task}: {test_accuracy}")
