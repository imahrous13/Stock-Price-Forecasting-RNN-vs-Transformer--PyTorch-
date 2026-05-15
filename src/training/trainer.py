import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm
import os

from .utils import EarlyStopping, set_seed


class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 0.0,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5
    ):
     
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta
        )
        
        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
     
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            
            # Handle different output shapes
            if outputs.shape != batch_y.shape:
                if outputs.dim() == 2 and batch_y.dim() == 2:
                    if outputs.shape[1] == 1 and batch_y.shape[1] > 1:
                        # Expand output to match target
                        outputs = outputs.repeat(1, batch_y.shape[1])
                    elif outputs.shape[1] > 1 and batch_y.shape[1] == 1:
                        # Take first output
                        outputs = outputs[:, 0:1]
            
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader: DataLoader) -> float:
      
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                
                # Handle different output shapes
                if outputs.shape != batch_y.shape:
                    if outputs.dim() == 2 and batch_y.dim() == 2:
                        if outputs.shape[1] == 1 and batch_y.shape[1] > 1:
                            outputs = outputs.repeat(1, batch_y.shape[1])
                        elif outputs.shape[1] > 1 and batch_y.shape[1] == 1:
                            outputs = outputs[:, 0:1]
                
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
     
        start_time = time.time()
        
        if verbose:
            print(f"Training on device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        prev_lr = self.optimizer.param_groups[0]['lr']
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check if learning rate changed
            if verbose and current_lr != prev_lr:
                print(f"Learning rate reduced from {prev_lr:.6f} to {current_lr:.6f}")
            prev_lr = current_lr
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                if verbose:
                    print(f"Training stopped early at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Best validation loss: {self.early_stopping.best_loss:.6f}")
        
        self.history['training_time'] = training_time
        
        return self.history
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
     
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': self.model.__dict__
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Model loaded from {filepath}")

