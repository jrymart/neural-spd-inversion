from landlab_torch_tools import build_datasets_from_db
import torch
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import time
import os

class PecletModelTrainer:
    """
    A class to train a model on Peclet number data.
    """

    def __init__(self, db_path, dataset_dir, model, label_query="SELECT log_peclet FROM model_run_outputs",
                 filter_query="", split_by="model_param.seed", train_fraction=.8, trim=5,
                 batch_size=64, epochs=5, learning_rate=0.001, train_transform=None, test_transform=None, **kwargs):
        """
        Initialize the trainer with a path to the SQLite database.
        
        Args:
            train_transform: Transform(s) to apply to training data (single transform, Compose, or list)
            test_transform: Transform(s) to apply to test data (single transform, Compose, or list)
        """
        self.db_path = db_path
        self.dataset_dir = dataset_dir
        self.label_query = label_query
        self.filter_query = filter_query
        self.split_by = split_by
        self.train_fraction = train_fraction
        self.trim = trim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = model
        num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
        num_workers=max(1, num_cpus - 1)
        self.train_ds, self.test_ds = build_datasets_from_db(
            db_path,
            dataset_dir,
            label_query,
            filter_query,
            split_by,
            train_fraction,
            trim,
            train_transform=train_transform,
            test_transform=test_transform,
            **kwargs
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Initialize training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'learning_rate': [],
            'batch_losses': [],
            'training_time': 0
        }

    def train(self, epochs=None, learning_rate=None, verbose=True, validate_every=1, checkpoint_every=10, checkpoint_path="/tmp/nn-spd_checkpoint.pt", reload_from_checkpoint=False):
        """
        Train the model with comprehensive metric tracking.
        
        Args:
            epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            verbose: Whether to print training progress
            validate_every: How often to run validation (every N epochs)
        """
        if epochs is None:
            epochs = self.epochs
        if learning_rate is None:
            learning_rate = self.learning_rate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}", flush=True)
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        start_epoch = 0
        
        if reload_from_checkpoint and Path(checkpoint_path).exists():
            if verbose:
                print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.training_history = checkpoint.get('training_history', self.training_history)
        start_time = time.time()
        if verbose:
            print(f"Starting training for {epochs} epochs with learning rate {learning_rate}...", flush=True)
        for epoch in range(start_epoch, epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            batch_losses = []
            for i, batch_content in enumerate(self.train_loader):
                inputs, labels = batch_content
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                epoch_train_loss += batch_loss
            
            # Calculate average training loss for epoch
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            
            # Validation phase (if enabled for this epoch)
            val_loss = None
            if epoch % validate_every == 0 or epoch == epochs - 1:
                val_loss = self._compute_validation_loss(criterion)
            
            # Record metrics
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(learning_rate)
            self.training_history['batch_losses'].extend(batch_losses)
            
            if verbose:
                val_str = f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}{val_str}", flush=True)

            if checkpoint_every and epoch % checkpoint_every == 0:
                tmp_path = f"{checkpoint_path}.tmp"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_history': self.training_history
                }, tmp_path)
                os.replace(tmp_path, checkpoint_path)
        # Record total training time
        self.training_history['training_time'] = time.time() - start_time
        
        if verbose:
            print(f"Training completed in {self.training_history['training_time']:.2f} seconds")
    
    def _compute_validation_loss(self, criterion):
        """Compute validation loss without updating gradients."""
        self.model.eval()
        device = next(self.model.parameters()).device
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = self.model(data)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
        return val_loss / len(self.test_loader)

    def save_weights(self, path):
        """
        Save the model weights to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        """
        Load the model weights from a file.
        """
        self.model.load_state_dict(torch.load(path))

    def evaluate(self):
        """
        Evaluate the model on the test set.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}", flush=True)
        self.model.to(device)
        self.model.eval()
        total_loss = 0
        criterion = torch.nn.MSELoss()
        predictions = []
        true_labels = []
        #names = []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader,0):
                data, labels = data
                data, labels = data.to(device), labels.to(device)
                outputs = self.model(data)
                loss = criterion(outputs, labels.unsqueeze(1))
                total_loss += loss.item()
                predictions += outputs
                true_labels += labels
                #names += names.tolist()
        if self.test_loader.dataset.normalize:
            predictions = [p*self.test_loader.dataset.labels_std + self.test_loader.dataset.labels_mean for p in predictions]
            true_labels = [l*self.test_loader.dataset.labels_std + self.test_loader.dataset.labels_mean for l in true_labels]
        average_loss = total_loss / len(self.test_loader)
        self.test_df = pd.DataFrame({'predictions': [float(p) for p in predictions],
                                     'true_labels': [float(p) for p in true_labels]})
         #                            'names': names})
        print(f"Test Loss: {average_loss}")
        return average_loss
    
    def save_training_history(self, path):
        """Save training history to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        history_copy = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                history_copy[key] = [float(x) if x is not None else None for x in value]
            else:
                history_copy[key] = float(value) if value is not None else None
        
        with open(path, 'w') as f:
            json.dump(history_copy, f, indent=2)
        print(f"Training history saved to {path}")
    
    def load_training_history(self, path):
        """Load training history from JSON file."""
        with open(path, 'r') as f:
            self.training_history = json.load(f)
        print(f"Training history loaded from {path}")
    
    def plot_loss_curves(self, save_path=None, show_validation=True, show_batch_loss=False):
        """
        Plot training and validation loss curves.
        
        Args:
            save_path: Path to save the plot (if None, just display)
            show_validation: Whether to show validation loss
            show_batch_loss: Whether to show individual batch losses
        """
        if not self.training_history['epochs']:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2 if show_batch_loss else 1, figsize=(12 if show_batch_loss else 8, 5))
        
        if not show_batch_loss:
            axes = [axes]  # Make it a list for consistent indexing
        
        # Plot epoch-wise losses
        epochs = self.training_history['epochs']
        train_losses = self.training_history['train_loss']
        
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if show_validation and any(x is not None for x in self.training_history['val_loss']):
            val_losses = self.training_history['val_loss']
            val_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
            val_losses_clean = [v for v in val_losses if v is not None]
            axes[0].plot(val_epochs, val_losses_clean, 'r-', label='Validation Loss', linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot batch-wise losses if requested
        if show_batch_loss and self.training_history['batch_losses']:
            batch_losses = self.training_history['batch_losses']
            batch_indices = range(len(batch_losses))
            axes[1].plot(batch_indices, batch_losses, 'g-', alpha=0.7, linewidth=0.5)
            axes[1].set_xlabel('Batch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Batch-wise Training Loss')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curves saved to {save_path}")
        
        plt.show()
    
    def get_training_summary(self):
        """Get a summary of training metrics."""
        if not self.training_history['epochs']:
            return "No training history available."
        
        summary = {
            'total_epochs': len(self.training_history['epochs']),
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'][-1] is not None else 'N/A',
            'min_train_loss': min(self.training_history['train_loss']),
            'min_val_loss': min([x for x in self.training_history['val_loss'] if x is not None]) if any(x is not None for x in self.training_history['val_loss']) else 'N/A',
            'training_time': self.training_history['training_time'],
            'total_batches': len(self.training_history['batch_losses'])
        }
        
        return summary
    
    def print_training_summary(self):
        """Print a formatted training summary."""
        summary = self.get_training_summary()
        if isinstance(summary, str):
            print(summary)
            return
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Total Batches: {summary['total_batches']}")
        print(f"Training Time: {summary['training_time']:.2f} seconds")
        print(f"Final Training Loss: {summary['final_train_loss']:.6f}")
        print(f"Final Validation Loss: {summary['final_val_loss']}")
        print(f"Best Training Loss: {summary['min_train_loss']:.6f}")
        print(f"Best Validation Loss: {summary['min_val_loss']}")
        print("="*50)
