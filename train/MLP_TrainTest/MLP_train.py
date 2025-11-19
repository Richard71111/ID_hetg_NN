import torch.nn as nn
from train.Early_Stop import EarlyStopping
from torch.utils.data import DataLoader
import torch
from typing import Optional
import tqdm

def train_MLP_epoch(model: nn.Module,
                    data_loader: DataLoader,
                    loss_fn,
                    opt,
                    device: torch.device,
                    grad_clip=None):
    model.train()
    total_loss = 0.0
    for inputs, targets in data_loader:
        if inputs.device != device or targets.device != device:
            inputs, targets = inputs.to(device), targets.to(device)
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

def validate_MLP_epoch(model: nn.Module,
                       data_loader: DataLoader,
                       loss_fn,
                       device: torch.device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            if inputs.device != device or targets.device != device:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

def train_MLP(model: nn.Module, 
                      train_dataloader, 
                      val_dataloader: DataLoader, 
                      optimizer, 
                      device,  
                      epochs: int, 
                      scheduler,
                      loss_fn,
                      grad_clip: Optional[float]=None,
                      patience: int=5, 
                      min_delta: int=0,
                      best_model_state_name: Optional[str]=None):
    model.to(device)

    best_model_state_name = best_model_state_name or "best_mlp_model.pth"
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, model='MLP',
                                  file_name=best_model_state_name)
    
    for epoch in tqdm.tqdm(range(epochs), desc="Training MLP"):
        train_loss = train_MLP_epoch(model, train_dataloader, loss_fn, optimizer, device, grad_clip)
        val_loss = validate_MLP_epoch(model, val_dataloader, loss_fn, device)
        
        if scheduler:
            scheduler.step(val_loss)
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        tqdm.tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return early_stopping.best_model_state if early_stopping.best_model_state is not None else {}