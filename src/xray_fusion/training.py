# Training, evaluation-loop, optimizer, and batching helpers.

from __future__ import annotations

import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm


# Seed Python, numpy, and torch for reproducible experiment runs.
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Create a seeded torch Generator for deterministic DataLoader shuffling.
def make_torch_generator(seed: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


# Create AdamW parameter groups with separate text and image learning rates.
def configure_optimizer(model, lr_text: float, lr_image: float, weight_decay: float):
    bert_params = []
    image_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bert" in name:
            bert_params.append(param)
        elif "image_encoder" in name or "backbone" in name:
            image_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if bert_params:
        groups.append({"params": bert_params, "lr": lr_text})
    if image_params:
        groups.append({"params": image_params, "lr": lr_image})
    if other_params:
        groups.append({"params": other_params, "lr": lr_image})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)

# Build the configured learning-rate scheduler.
def build_scheduler(optimizer, scheduler_name: str = "constant"):
    if scheduler_name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


# Train one epoch for the tuple-based dual-image baseline.
def train_dual_image_epoch(model, loader, criterion, optimizer, device, desc: str = "Train"):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for frontal, lateral, labels in tqdm(loader, desc=desc, leave=False):
        frontal = frontal.to(device)
        lateral = lateral.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(frontal, lateral)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count

# Train one epoch for the simple vocabulary fusion baseline.
def train_simple_fusion_epoch(model, loader, criterion, optimizer, device, desc: str = "Train Fusion"):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for frontal, lateral, text_ids, labels in tqdm(loader, desc=desc, leave=False):
        frontal = frontal.to(device)
        lateral = lateral.to(device)
        text_ids = text_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(frontal, lateral, text_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count

@torch.no_grad()
# Evaluate the tuple-based dual-image baseline.
def evaluate_dual_image(model, loader, criterion, device, desc: str = "Eval"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_true = []
    all_pred = []
    for frontal, lateral, labels in tqdm(loader, desc=desc, leave=False):
        frontal = frontal.to(device)
        lateral = lateral.to(device)
        labels = labels.to(device)
        logits = model(frontal, lateral)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
    return total_loss / total_count, total_correct / total_count, np.array(all_true), np.array(all_pred)

@torch.no_grad()
# Evaluate the simple vocabulary fusion baseline.
def evaluate_simple_fusion(model, loader, criterion, device, desc: str = "Eval Fusion"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_true = []
    all_pred = []
    for frontal, lateral, text_ids, labels in tqdm(loader, desc=desc, leave=False):
        frontal = frontal.to(device)
        lateral = lateral.to(device)
        text_ids = text_ids.to(device)
        labels = labels.to(device)
        logits = model(frontal, lateral, text_ids)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
    return total_loss / total_count, total_correct / total_count, np.array(all_true), np.array(all_pred)


# Move tensor-like values in a batch dict onto a torch device.
def move_batch_to_device(batch: dict, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}

# Select and move only the tensors needed for a task type.
def move_task_batch_to_device(batch: dict, task_type: str, device):
    keys_by_task = {
        "image_only": ["frontal", "lateral", "labels"],
        "text_only": ["input_ids", "attention_mask", "labels"],
        "fusion": ["frontal", "lateral", "input_ids", "attention_mask", "labels"],
    }
    return {key: batch[key].to(device) for key in keys_by_task[task_type]}

# Dispatch a model forward pass based on the configured task type.
def forward_task(model, batch: dict, task_type: str):
    if task_type == "image_only":
        return model(batch["frontal"], batch["lateral"])
    if task_type == "text_only":
        return model(batch["input_ids"], batch["attention_mask"])
    if task_type == "fusion":
        return model(batch["frontal"], batch["lateral"], batch["input_ids"], batch["attention_mask"])
    raise ValueError(f"Unknown task_type: {task_type}")


# Train one epoch using dict batches for image, text, or fusion tasks.
def train_one_epoch(model, loader, criterion, optimizer, device, task_type: str = "fusion", desc: str = "Train"):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        batch = move_task_batch_to_device(batch, task_type, device)
        labels = batch["labels"]
        optimizer.zero_grad()
        logits = forward_task(model, batch, task_type)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size
    return total_loss / total_count, total_correct / total_count

@torch.no_grad()
# Evaluate a dict-batch model and return labels, predictions, and probabilities.
def evaluate_model(model, loader, criterion, device, task_type: str = "fusion", desc: str = "Eval"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true = []
    y_pred = []
    y_prob = []
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        batch = move_task_batch_to_device(batch, task_type, device)
        labels = batch["labels"]
        logits = forward_task(model, batch, task_type)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())
    return total_loss / total_count, total_correct / total_count, np.array(y_true), np.array(y_pred), np.array(y_prob)

@torch.no_grad()
# Evaluate gated fusion and collect average gate weights for interpretation.
def evaluate_gated_fusion(model, loader, criterion, device, desc: str = "Eval Fusion"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true, y_pred, gate_weight_batches = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        batch = move_task_batch_to_device(batch, "fusion", device)
        labels = batch["labels"]
        logits, aux = model(batch["frontal"], batch["lateral"], batch["input_ids"], batch["attention_mask"], return_aux=True)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        gate_weight_batches.append(aux["gate_weights"].cpu().numpy())
    return total_loss / total_count, total_correct / total_count, np.array(y_true), np.array(y_pred), np.concatenate(gate_weight_batches, axis=0)
