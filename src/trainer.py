"""
Training module for text generation models.
Located in: src/trainer.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle

class TextDataset(Dataset):
    """Dataset class for text generation training."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ModelTrainer:
    """Comprehensive trainer for text generation models."""
    
    def __init__(self, model, vocab_size: int, device: str = 'auto'):
        self.model = model
        self.vocab_size = vocab_size
        
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("🚀 Using Apple Silicon GPU (MPS)")
            else:
                self.device = torch.device('cpu')
                print("🚀 Using CPU")
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def create_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                           batch_size: int = 64, validation_split: float = 0.1):
        """Create training and validation data loaders."""
        
        if X_val is None or y_val is None:
            # Split training data for validation
            split_idx = int(len(X_train) * (1 - validation_split))
            X_train, X_val = X_train[:split_idx], X_train[split_idx:]
            y_train, y_val = y_train[:split_idx], y_train[split_idx:]
        
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Validation samples: {len(X_val):,}")
        
        # Create datasets
        train_dataset = TextDataset(X_train, y_train)
        val_dataset = TextDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        return train_loader, val_loader
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 50, batch_size: int = 64, learning_rate: float = 0.001,
              validation_split: float = 0.1, save_dir: str = "models",
              model_name: str = "text_generator", patience: int = 5,
              grad_clip: float = 1.0):
        """Train the text generation model."""
        
        print("🎯 STARTING MODEL TRAINING")
        print("=" * 50)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, batch_size=batch_size, validation_split=validation_split
        )
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3
        )
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        best_epoch = 0
        patience_counter = 0
        
        print(f"🔧 Training Configuration:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Device: {self.device}")
        print(f"   - Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader, criterion, optimizer, grad_clip)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader, criterion)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                self._save_model(save_path / f"{model_name}_best.pth")
                
            else:
                patience_counter += 1
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏰ Early stopping triggered after {patience} epochs without improvement")
                print(f"🏆 Best validation loss: {self.best_val_loss:.4f} at epoch {best_epoch + 1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ Training completed! Loaded best model from epoch {best_epoch + 1}")
        
        # Save final model and training history
        self._save_model(save_path / f"{model_name}_final.pth")
        self._save_training_history(save_path / f"{model_name}_history.json")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': best_epoch
        }
    
    def _train_epoch(self, train_loader, criterion, optimizer, grad_clip):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'init_hidden'):
                # LSTM model
                output, _ = self.model(data)
            else:
                # Transformer model
                output = self.model(data)
            
            # Calculate loss (use last time step for next token prediction)
            if len(output.shape) == 3:  # (batch, seq, vocab)
                output = output[:, -1, :]  # Get last time step
            
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'init_hidden'):
                    # LSTM model
                    output, _ = self.model(data)
                else:
                    # Transformer model
                    output = self.model(data)
                
                # Calculate loss
                if len(output.shape) == 3:  # (batch, seq, vocab)
                    output = output[:, -1, :]  # Get last time step
                
                loss = criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _save_model(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.vocab_size,
            'model_type': type(self.model).__name__,
        }
        torch.save(checkpoint, filepath)
    
    def _save_training_history(self, filepath):
        """Save training history."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 64):
        """Evaluate model on test data."""
        print("📊 EVALUATING MODEL")
        print("-" * 30)
        
        test_dataset = TextDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'init_hidden'):
                    output, _ = self.model(data)
                else:
                    output = self.model(data)
                
                if len(output.shape) == 3:
                    output = output[:, -1, :]
                
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = torch.argmax(output, dim=1)
                correct_predictions += (predicted == target).sum().item()
                total_predictions += target.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"📊 Test Results:")
        print(f"   - Loss: {avg_loss:.4f}")
        print(f"   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   - Perplexity: {perplexity:.2f}")
        
        return {
            'test_loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity
        }

def load_model(filepath: str, device: str = 'auto'):
    """Load a saved model."""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    print(f"✅ Loaded model: {checkpoint['model_type']} with {checkpoint['vocab_size']} vocab size")
    
    return checkpoint

# Example usage
if __name__ == "__main__":
    print("🧪 Testing ModelTrainer...")
    
    # Mock test data
    vocab_size = 1000
    seq_len = 50
    num_samples = 1000
    
    X_dummy = np.random.randint(0, vocab_size, (num_samples, seq_len))
    y_dummy = np.random.randint(0, vocab_size, (num_samples,))
    
    print(f"✅ Created dummy data: X{X_dummy.shape}, y{y_dummy.shape}")
    print("🎉 ModelTrainer module ready!")
