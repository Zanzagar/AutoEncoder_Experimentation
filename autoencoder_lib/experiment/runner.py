"""
ExperimentRunner Class for Autoencoder Training Management
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from ..visualization.latent_viz import visualize_latent_space_2d
from ..visualization.reconstruction_viz import visualize_reconstructions


class ExperimentRunner:
    """
    Manages training and evaluation of autoencoder experiments.
    Provides systematic exploration capabilities and comprehensive result tracking.
    """
    
    def __init__(self, device=None, output_dir=None, random_seed=42):
        """
        Initialize the ExperimentRunner.
        
        Args:
            device: torch.device for training ('cpu' or 'cuda')
            output_dir: Directory to save experiment results
            random_seed: Random seed for reproducibility
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir if output_dir is not None else "experiment_results"
        self.random_seed = random_seed
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        print(f"ExperimentRunner initialized - Device: {self.device}, Output: {self.output_dir}")
    
    def memory_efficient_evaluation(self, model, test_data, loss_func, batch_size=64):
        """
        Evaluate model on test data in a memory-efficient way.
        
        Args:
            model: The model to evaluate
            test_data: Test data tensor
            loss_func: Loss function
            batch_size: Batch size for evaluation
            
        Returns:
            Average loss across all test samples
        """
        model.eval()
        n_samples = test_data.size(0)
        total_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = test_data[i:i+batch_size].to(self.device)
                encoded, decoded = model(batch)
                loss = loss_func(decoded, batch)
                total_loss += loss.item() * batch.size(0)
        
        avg_loss = total_loss / n_samples
        return avg_loss
    
    def select_visualization_samples(self, test_data, test_labels, class_names=None, samples_per_class=2):
        """
        Select representative samples from each class for visualization.
        Uses deterministic selection to ensure the same images are shown every time.
        
        Args:
            test_data: Test data tensor
            test_labels: Test labels tensor
            class_names: List of class names
            samples_per_class: Number of samples to show per class
            
        Returns:
            Tuple of (view_data, view_labels, view_indices)
        """
        if class_names is not None:
            # Find examples of each class in the test data (deterministic selection)
            class_samples = {}
            
            # First pass: collect all samples for each class
            for i in range(len(test_labels)):
                label = test_labels[i].item()
                if label not in class_samples:
                    class_samples[label] = []
                class_samples[label].append(i)
            
            # Second pass: deterministically select first N samples for each class
            view_indices = []
            for label in sorted(class_samples.keys()):
                # Sort indices to ensure deterministic selection
                available_indices = sorted(class_samples[label])
                # Take the first samples_per_class indices for consistency
                selected_indices = available_indices[:min(samples_per_class, len(available_indices))]
                view_indices.extend(selected_indices)
                
                if len(available_indices) < samples_per_class:
                    print(f"Warning: Only {len(available_indices)} samples available for class {label}, requested {samples_per_class}")
            
            # Sort view_indices to ensure consistent ordering
            view_indices = sorted(view_indices)
            view_data = test_data[view_indices].to(self.device)
            view_labels = test_labels[view_indices]
            
            print(f"Selected visualization samples: {len(view_indices)} total ({samples_per_class} per class)")
            for label in sorted(class_samples.keys()):
                label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
                selected_for_label = [i for i in view_indices if test_labels[i].item() == label]
                print(f"  {label_name}: indices {selected_for_label}")
        else:
            # Use first n samples if no class information (deterministic)
            n_test_img = min(10, len(test_data))
            view_indices = list(range(n_test_img))
            view_data = test_data[:n_test_img].to(self.device)
            view_labels = test_labels[:n_test_img]
            print(f"Selected first {n_test_img} samples for visualization (no class info)")
        
        return view_data, view_labels, view_indices
    
    def collect_training_data_sample(self, train_loader, max_samples=500):
        """
        Collect a sample of training data for visualization and analysis.
        Uses deterministic selection for consistency.
        
        Args:
            train_loader: DataLoader for training data
            max_samples: Maximum number of training samples to collect
            
        Returns:
            Tuple of (train_data_tensor, train_labels_tensor)
        """
        train_data_tensor = None
        train_labels_tensor = None
        
        for batch in train_loader:
            if len(batch) == 3:
                x, _, labels = batch
            elif len(batch) == 2:
                x, _ = batch
                labels = None
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            
            # Initialize or concatenate tensors
            if train_data_tensor is None:
                train_data_tensor = x.clone()
                train_labels_tensor = labels.clone() if labels is not None else None
            else:
                train_data_tensor = torch.cat((train_data_tensor, x), 0)
                if labels is not None and train_labels_tensor is not None:
                    train_labels_tensor = torch.cat((train_labels_tensor, labels), 0)
            
            # Limit to max_samples to prevent memory issues
            if train_data_tensor.size(0) >= max_samples:
                train_data_tensor = train_data_tensor[:max_samples]
                if train_labels_tensor is not None:
                    train_labels_tensor = train_labels_tensor[:max_samples]
                break
        
        return train_data_tensor, train_labels_tensor
    
    def select_training_visualization_samples(self, train_data_tensor, train_labels_tensor, 
                                            class_names=None, samples_per_class=2):
        """
        Select representative training samples for visualization.
        Uses deterministic selection to ensure the same images are shown every time.
        
        Args:
            train_data_tensor: Training data tensor
            train_labels_tensor: Training labels tensor
            class_names: List of class names
            samples_per_class: Number of samples to show per class
            
        Returns:
            Tuple of (train_view_data, train_view_labels, train_view_indices)
        """
        if train_labels_tensor is not None and class_names is not None:
            # Find examples of each class in the training data (deterministic selection)
            class_samples = {}
            
            # First pass: collect all samples for each class
            for i in range(len(train_labels_tensor)):
                label = train_labels_tensor[i].item()
                if label not in class_samples:
                    class_samples[label] = []
                class_samples[label].append(i)
            
            # Second pass: deterministically select first N samples for each class
            view_indices = []
            for label in sorted(class_samples.keys()):
                # Sort indices to ensure deterministic selection
                available_indices = sorted(class_samples[label])
                # Take the first samples_per_class indices for consistency
                selected_indices = available_indices[:min(samples_per_class, len(available_indices))]
                view_indices.extend(selected_indices)
            
            # Sort view_indices to ensure consistent ordering
            view_indices = sorted(view_indices)
            train_view_data = train_data_tensor[view_indices].to(self.device)
            train_view_labels = train_labels_tensor[view_indices]
            
            print(f"Selected training visualization samples: {len(view_indices)} total ({samples_per_class} per class)")
            for label in sorted(class_samples.keys()):
                label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
                selected_for_label = [i for i in view_indices if train_labels_tensor[i].item() == label]
                print(f"  Train {label_name}: indices {selected_for_label}")
        else:
            # Use first n samples if no class information (deterministic)
            n_train_img = min(len(train_data_tensor), 10)
            view_indices = list(range(n_train_img))
            train_view_data = train_data_tensor[:n_train_img].to(self.device)
            train_view_labels = train_labels_tensor[:n_train_img] if train_labels_tensor is not None else None
            print(f"Selected first {n_train_img} training samples for visualization (no class info)")
        
        return train_view_data, train_view_labels, view_indices
    
    def calculate_visualization_epochs(self, epochs, num_visualizations=5):
        """
        Calculate epochs where visualizations should be performed.
        
        Args:
            epochs: Total number of training epochs
            num_visualizations: Number of visualizations to show
            
        Returns:
            List of epoch indices for visualization
        """
        num_visualizations = min(num_visualizations, epochs)
        key_epochs = []
        
        if num_visualizations == 1:
            key_epochs = [epochs - 1]
        elif num_visualizations == 2:
            key_epochs = [0, epochs - 1]
        else:
            # Calculate step size for equally spaced epochs
            step = (epochs - 1) / (num_visualizations - 1)
            for i in range(num_visualizations):
                key_epoch = min(round(i * step), epochs - 1)
                if key_epoch not in key_epochs:
                    key_epochs.append(key_epoch)
        
        return key_epochs
    
    def train_autoencoder(self, model, train_loader, test_data, test_labels, 
                          epochs=10, learning_rate=0.001, loss_func=None,
                          class_names=None, visualization_interval=500,
                          num_visualizations=5, save_model=True,
                          experiment_name=None, calculate_train_silhouette=True,
                          calculate_test_silhouette=True):
        """
        Train an autoencoder model with comprehensive tracking and visualization.
        
        Args:
            model: The autoencoder model to train
            train_loader: DataLoader for training data
            test_data: Tensor of test data
            test_labels: Tensor of test labels
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            loss_func: Loss function (default: MSELoss)
            class_names: List of class names for visualization
            visualization_interval: Steps between visualizations
            num_visualizations: Number of visualizations during training
            save_model: Whether to save the model after training
            experiment_name: Name for this experiment
            calculate_train_silhouette: Whether to calculate train silhouette scores
            calculate_test_silhouette: Whether to calculate test silhouette scores
            
        Returns:
            Tuple of (model, history) containing the trained model and training history
        """
        # Set default loss function
        if loss_func is None:
            loss_func = nn.MSELoss()
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{model.__class__.__name__}_{timestamp}"
        
        print(f"Training {model.__class__.__name__} for {epochs} epochs on {self.device}")
        print(f"Learning rate: {learning_rate}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer with weight decay for regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Select visualization samples
        view_data, view_labels, view_indices = self.select_visualization_samples(
            test_data, test_labels, class_names
        )
        
        # Collect training data sample
        train_data_tensor, train_labels_tensor = self.collect_training_data_sample(train_loader)
        train_view_data, train_view_labels, train_view_indices = self.select_training_visualization_samples(
            train_data_tensor, train_labels_tensor, class_names
        )
        
        # Initialize training history
        history = {
            'train_loss': [],
            'test_loss': [],
            'epochs': epochs,
            'learning_rate': learning_rate,
            'model_name': model.__class__.__name__,
            'experiment_name': experiment_name,
            'train_silhouette_scores': [],
            'test_silhouette_scores': [],
            'view_indices': view_indices
        }
        
        # Calculate visualization epochs
        key_epochs = self.calculate_visualization_epochs(epochs, num_visualizations)
        print(f"Visualizing at epochs: {key_epochs}")
        
        # Training loop
        start_time = time.time()
        final_visualization_done = False
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            batches = 0
            
            for step, batch in enumerate(train_loader):
                # Unpack batch
                if len(batch) == 3:
                    x, y, label = batch
                elif len(batch) == 2:
                    x, y = batch
                    label = None
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                
                b_x = x.float().to(self.device)
                
                # Forward pass
                encoded, decoded = model(b_x)
                loss = loss_func(decoded, b_x)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                batches += 1
                
                # Check if visualization should be performed
                is_final_step = (epoch == epochs - 1) and (step == len(train_loader) - 1)
                should_visualize = (step % visualization_interval == 0 and epoch in key_epochs) or is_final_step
                
                if should_visualize:
                    if is_final_step:
                        print(f"Performing final visualization at epoch {epoch+1}/{epochs}, step {step}")
                        final_visualization_done = True
                    
                    model.eval()
                    test_loss = self.memory_efficient_evaluation(model, test_data, loss_func)
                    print(f'Epoch: {epoch+1}/{epochs}, Step: {step} | train loss: {loss.item():.4f} | test loss: {test_loss:.4f}')
                    
                    # Record metrics
                    history['train_loss'].append(loss.item())
                    history['test_loss'].append(test_loss)
                    
                    # Visualizations during training (except final step to avoid duplication)
                    if not is_final_step:
                        self._perform_training_visualizations(
                            model, train_view_data, train_view_labels, view_data, view_labels,
                            train_data_tensor, train_labels_tensor, test_data, test_labels,
                            class_names, epoch+1, step, history, calculate_train_silhouette, 
                            calculate_test_silhouette
                        )
                    
                    model.train()
            
            # End of epoch evaluation
            model.eval()
            test_loss = self.memory_efficient_evaluation(model, test_data, loss_func)
            avg_train_loss = epoch_loss / batches
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(test_loss)
            
            # Update learning rate scheduler
            scheduler.step(test_loss)
            
            print(f'Epoch {epoch+1}/{epochs} completed | '
                  f'Train loss: {avg_train_loss:.4f} | '
                  f'Test loss: {test_loss:.4f}')
        
        # Final evaluation and metrics
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Calculate final losses
        model.eval()
        with torch.no_grad():
            if train_data_tensor is not None:
                train_data_tensor = train_data_tensor.to(self.device)
                _, decoded_train = model(train_data_tensor)
                final_train_loss = loss_func(decoded_train, train_data_tensor).item()
            else:
                final_train_loss = None
            
            final_test_loss = self.memory_efficient_evaluation(model, test_data, loss_func)
        
        # Final visualization and silhouette calculation
        final_train_silhouette, final_test_silhouette = self._perform_final_evaluation(
            model, train_data_tensor, train_labels_tensor, test_data, test_labels,
            class_names, calculate_train_silhouette, calculate_test_silhouette,
            final_visualization_done, train_view_data, train_view_labels, view_data, view_labels
        )
        
        # CONSOLIDATED FINAL RESULTS SUMMARY
        print("\n" + "="*70)
        print("🎯 FINAL EXPERIMENT RESULTS")
        print("="*70)
        print(f"📊 Training Summary:")
        print(f"   • Model: {model.__class__.__name__}")
        print(f"   • Training Time: {train_time:.2f} seconds")
        print(f"   • Total Epochs: {epochs}")
        print(f"   • Learning Rate: {learning_rate}")
        print(f"\n📈 Final Loss Metrics:")
        if final_train_loss is not None:
            print(f"   • Final Train Loss: {final_train_loss:.6f}")
        print(f"   • Final Test Loss: {final_test_loss:.6f}")
        print(f"\n🎯 Final Silhouette Scores (Latent Space Quality):")
        if final_train_silhouette is not None:
            print(f"   • Training Data: {final_train_silhouette:.4f}")
        else:
            print(f"   • Training Data: N/A (insufficient classes)")
        if final_test_silhouette is not None:
            print(f"   • Test Data: {final_test_silhouette:.4f}")
        else:
            print(f"   • Test Data: N/A (insufficient classes)")
        print("="*70)
        print("✅ Experiment Complete!")
        print("="*70 + "\n")
        
        # Complete history
        history.update({
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'training_time': train_time,
            'final_train_silhouette': final_train_silhouette,
            'final_test_silhouette': final_test_silhouette,
            'final_silhouette': final_test_silhouette  # For backward compatibility
        })
        
        # Store latent dimension
        self._extract_latent_dimension(model, history)
        
        # Save model if requested
        if save_model:
            self._save_model(model, optimizer, history, experiment_name)
        
        return model, history
    
    def _perform_training_visualizations(self, model, train_view_data, train_view_labels,
                                       view_data, view_labels, train_data_tensor, train_labels_tensor,
                                       test_data, test_labels, class_names, epoch, step, history,
                                       calculate_train_silhouette, calculate_test_silhouette):
        """Perform visualizations during training."""
        with torch.no_grad():
            # Get reconstructions for both train and test data
            _, train_decoded = model(train_view_data)
            _, test_decoded = model(view_data)
            
            # Show training reconstructions
            self._show_reconstructions(
                train_view_data, train_decoded, train_view_labels, class_names,
                f'Training Progress - Epoch {epoch}, Step {step} - Train: Original vs. Reconstructed'
            )
            
            # Show test reconstructions
            self._show_reconstructions(
                view_data, test_decoded, view_labels, class_names,
                f'Training Progress - Epoch {epoch}, Step {step} - Test: Original vs. Reconstructed'
            )
            
            # Visualize latent space and calculate silhouette scores
            # Extract latent representations
            encoded_train, _ = model(train_data_tensor.to(self.device))
            encoded_test, _ = model(test_data.to(self.device))
            
            # Flatten latent vectors for visualization
            train_latent = encoded_train.view(encoded_train.size(0), -1).cpu()
            test_latent = encoded_test.view(encoded_test.size(0), -1).cpu()
            
            # Visualize both train and test latent spaces side by side
            try:
                train_silhouette, test_silhouette = self.visualize_latent_space_side_by_side(
                    train_latent, train_labels_tensor, test_latent, test_labels,
                    class_names, f'Training Progress - Epoch {epoch}, Step {step}', 'tsne'
                )
                if calculate_train_silhouette and train_silhouette is not None:
                    history['train_silhouette_scores'].append(train_silhouette)
                if calculate_test_silhouette and test_silhouette is not None:
                    history['test_silhouette_scores'].append(test_silhouette)
            except Exception as e:
                print(f"Warning: Could not visualize latent spaces: {e}")
    
    def _perform_final_evaluation(self, model, train_data_tensor, train_labels_tensor,
                                test_data, test_labels, class_names, calculate_train_silhouette,
                                calculate_test_silhouette, final_visualization_done,
                                train_view_data, train_view_labels, view_data, view_labels):
        """Perform final evaluation and visualization."""
        print("Generating final latent space visualization and metrics")
        
        # Always perform comprehensive final visualization
        final_train_silhouette, final_test_silhouette = self.comprehensive_final_visualization(
            model, train_view_data, train_view_labels, view_data, view_labels,
            train_data_tensor, train_labels_tensor, test_data, test_labels, class_names
        )
        
        return final_train_silhouette, final_test_silhouette
    
    def _show_reconstructions(self, original_data, reconstructed_data, labels, class_names, title):
        """Show reconstruction visualizations."""
        n_images = len(original_data)
        f, a = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        plt.suptitle(title, fontsize=14)
        
        # Original images on top row
        for i in range(n_images):
            a[0][i].imshow(original_data[i, 0].cpu().numpy(), cmap='gray')
            a[0][i].set_xticks(())
            a[0][i].set_yticks(())
            
            if class_names is not None and labels is not None and i < len(labels):
                class_idx = labels[i].item()
                if i == 0:
                    a[0][i].set_ylabel("Original", fontsize=10)
                a[0][i].set_title(class_names[class_idx], fontsize=8)
        
        # Reconstructed images on bottom row
        for i in range(n_images):
            a[1][i].imshow(reconstructed_data[i, 0].cpu().numpy(), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
            if i == 0:
                a[1][i].set_ylabel("Reconstructed", fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def _extract_latent_dimension(self, model, history):
        """Extract and store the latent dimension from the model."""
        if hasattr(model, 'fc_encoder'):
            if isinstance(model.fc_encoder, nn.Sequential):
                # For models with sequential fc_encoder, get last layer's out_features
                history['latent_dim'] = model.fc_encoder[-1].out_features
            else:
                # For models with single fc_encoder layer
                history['latent_dim'] = model.fc_encoder.out_features
        elif hasattr(model.encoder[-1], 'out_features'):
            history['latent_dim'] = model.encoder[-1].out_features
    
    def _save_model(self, model, optimizer, history, experiment_name):
        """Save the trained model and training history."""
        save_path = os.path.join(self.output_dir, f"{experiment_name}_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def run_experiment(self, model_class, model_kwargs, train_loader, test_data, test_labels,
                       training_kwargs=None, experiment_name=None):
        """
        Run a complete autoencoder experiment.
        
        Args:
            model_class: Class of the model to create
            model_kwargs: Keyword arguments for model initialization
            train_loader: DataLoader for training data
            test_data: Test data tensor
            test_labels: Test labels tensor
            training_kwargs: Keyword arguments for training
            experiment_name: Name for this experiment
            
        Returns:
            Tuple of (model, history)
        """
        # Create model
        model = model_class(**model_kwargs)
        print(f"Created model: {model.__class__.__name__}")
        
        # Set default training parameters
        if training_kwargs is None:
            training_kwargs = {}
        
        # Train the model
        return self.train_autoencoder(
            model=model,
            train_loader=train_loader,
            test_data=test_data,
            test_labels=test_labels,
            experiment_name=experiment_name,
            **training_kwargs
        )
    
    def visualize_latent_space_side_by_side(self, train_latent, train_labels, test_latent, test_labels, 
                                           class_names=None, epoch_info="", method='tsne'):
        """
        Visualize train and test latent spaces side by side with silhouette scores in titles.
        
        Args:
            train_latent: Training latent representations
            train_labels: Training labels
            test_latent: Test latent representations  
            test_labels: Test labels
            class_names: List of class names
            epoch_info: Information about current epoch for title
            method: Dimensionality reduction method ('tsne' or 'pca')
            
        Returns:
            Tuple of (train_silhouette, test_silhouette)
        """
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create side-by-side subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            train_silhouette = None
            test_silhouette = None
            
            # Process training data
            if len(np.unique(train_labels)) > 1:  # Need at least 2 classes for silhouette
                if method == 'tsne':
                    if train_latent.shape[1] > 2:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(train_latent)-1))
                        train_2d = tsne.fit_transform(train_latent)
                    else:
                        train_2d = train_latent.numpy() if hasattr(train_latent, 'numpy') else train_latent
                else:  # PCA
                    pca = PCA(n_components=2, random_state=42)
                    train_2d = pca.fit_transform(train_latent)
                
                # Calculate silhouette score
                train_silhouette = silhouette_score(train_latent, train_labels)
                
                # Plot training data
                unique_labels = np.unique(train_labels)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = train_labels == label
                    label_name = class_names[label] if class_names and label < len(class_names) else f'Class {label}'
                    ax1.scatter(train_2d[mask, 0], train_2d[mask, 1], 
                              c=[colors[i]], label=label_name, alpha=0.7, s=50)
                
                ax1.set_title(f'{epoch_info} - Train Latent Space ({method.upper()})\nSilhouette Score: {train_silhouette:.4f}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Insufficient classes\nfor visualization', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f'{epoch_info} - Train Latent Space ({method.upper()})\nSilhouette Score: N/A')
            
            # Process test data
            if len(np.unique(test_labels)) > 1:  # Need at least 2 classes for silhouette
                if method == 'tsne':
                    if test_latent.shape[1] > 2:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_latent)-1))
                        test_2d = tsne.fit_transform(test_latent)
                    else:
                        test_2d = test_latent.numpy() if hasattr(test_latent, 'numpy') else test_latent
                else:  # PCA
                    pca = PCA(n_components=2, random_state=42)
                    test_2d = pca.fit_transform(test_latent)
                
                # Calculate silhouette score
                test_silhouette = silhouette_score(test_latent, test_labels)
                
                # Plot test data
                unique_labels = np.unique(test_labels)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = test_labels == label
                    label_name = class_names[label] if class_names and label < len(class_names) else f'Class {label}'
                    ax2.scatter(test_2d[mask, 0], test_2d[mask, 1], 
                              c=[colors[i]], label=label_name, alpha=0.7, s=50)
                
                ax2.set_title(f'{epoch_info} - Test Latent Space ({method.upper()})\nSilhouette Score: {test_silhouette:.4f}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Insufficient classes\nfor visualization', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'{epoch_info} - Test Latent Space ({method.upper()})\nSilhouette Score: N/A')
            
            plt.tight_layout()
            plt.show()
            
            return train_silhouette, test_silhouette
            
        except Exception as e:
            print(f"Warning: Could not create side-by-side latent space visualization: {e}")
            return None, None
    
    def comprehensive_final_visualization(self, model, train_view_data, train_view_labels, 
                                        view_data, view_labels, train_data_tensor, train_labels_tensor,
                                        test_data, test_labels, class_names):
        """
        Perform comprehensive final visualization showing both reconstructions and latent spaces.
        
        Args:
            model: Trained model
            train_view_data: Selected training data for visualization
            train_view_labels: Labels for training visualization data
            view_data: Selected test data for visualization
            view_labels: Labels for test visualization data
            train_data_tensor: Full training data tensor
            train_labels_tensor: Full training labels tensor
            test_data: Full test data tensor
            test_labels: Full test labels tensor
            class_names: List of class names
            
        Returns:
            Tuple of (final_train_silhouette, final_test_silhouette)
        """
        print("\n" + "="*60)
        print("📊 FINAL VISUALIZATION GENERATION")
        print("="*60)
        
        with torch.no_grad():
            # 1. Show final reconstructions
            print("Generating final reconstruction visualizations...")
            _, train_decoded = model(train_view_data)
            _, test_decoded = model(view_data)
            
            self._show_reconstructions(
                train_view_data, train_decoded, train_view_labels, class_names,
                'Final Results - Training: Original vs. Reconstructed'
            )
            
            self._show_reconstructions(
                view_data, test_decoded, view_labels, class_names,
                'Final Results - Test: Original vs. Reconstructed'
            )
            
            # 2. Show final latent space visualization
            print("Generating final latent space analysis...")
            # Extract latent representations for final visualization
            encoded_train, _ = model(train_data_tensor.to(self.device))
            encoded_test, _ = model(test_data.to(self.device))
            
            # Flatten latent vectors
            train_latent = encoded_train.view(encoded_train.size(0), -1).cpu()
            test_latent = encoded_test.view(encoded_test.size(0), -1).cpu()
            
            # Final latent space visualization (both train and test side by side)
            try:
                final_train_silhouette, final_test_silhouette = self.visualize_latent_space_side_by_side(
                    train_latent, train_labels_tensor, test_latent, test_labels,
                    class_names, 'Final Results', 'tsne'
                )
                
            except Exception as e:
                print(f"Warning: Could not create final latent space visualization: {e}")
                final_train_silhouette = None
                final_test_silhouette = None
        
        print("="*60)
        print("✅ FINAL VISUALIZATION COMPLETE")
        print("="*60)
        
        return final_train_silhouette, final_test_silhouette 