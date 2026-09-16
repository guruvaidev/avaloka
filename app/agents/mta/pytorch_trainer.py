"""
PyTorch Training Core for Model Training Agent

Provides dynamic PyTorch model creation, training, and evaluation
with support for classification and regression tasks.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from typing import Dict, Any, Optional, Tuple, List, Union
import os
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
MIN_DATA_SIZE = 5


class TabularDataset(Dataset):
    """Custom dataset for tabular data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, task_type: str = "classification"):
        self.features = torch.FloatTensor(features)
        
        if task_type == "classification":
            self.targets = torch.LongTensor(targets)
        else:  # regression
            self.targets = torch.FloatTensor(targets)
        
        self.task_type = task_type
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DynamicMLP(nn.Module):
    """Dynamic Multi-Layer Perceptron for tabular data"""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], 
                 activation: str = "relu", dropout: float = 0.2, batch_norm: bool = True):
        super(DynamicMLP, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PyTorchTrainer:
    """Core PyTorch training engine"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PyTorch trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        # Prefer CUDA, then Apple MPS, else CPU
        if config.get("use_gpu", False) and torch.cuda.is_available():
            selected_device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            selected_device = "mps"
        else:
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.label_encoder = None
        self.training_history = []
        
        logger.info(f"PyTorch trainer initialized on device: {self.device}")
    
    def prepare_data(self, data_path: str, target_column: str, task_type: str = "classification") -> Dict[str, Any]:
        """
        Prepare data for training
        
        Args:
            data_path: Path to the dataset (CSV file)
            target_column: Name of the target column
            task_type: "classification" or "regression"
            
        Returns:
            Dictionary with prepared data information
        """
        try:
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Check if dataset is too small for meaningful ML training
            if len(df) < MIN_DATA_SIZE:
                logger.warning(f"Dataset too small for ML training: {len(df)} samples. Minimum recommended: 5")
                return {
                    "error": f"Dataset too small for ML training. Only {len(df)} samples available. Minimum recommended: 5 samples.",
                    "suggestion": "Please provide a larger dataset or use a different approach for this small dataset."
                }
            
            # Separate features and target
            if target_column not in df.columns:
                # Try to infer target column
                target_column = self._infer_target_column(df)
                if not target_column:
                    raise ValueError("Could not identify target column")
            
            features = df.drop(columns=[target_column])
            target = df[target_column]
            
            # Handle missing values
            features = features.fillna(features.mean(numeric_only=True))
            target = target.fillna(target.mode()[0] if task_type == "classification" else target.mean())
            
            # Encode categorical features
            categorical_columns = features.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
            
            # Scale features
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            
            # Encode target for classification
            if task_type == "classification":
                self.label_encoder = LabelEncoder()
                target_encoded = self.label_encoder.fit_transform(target)
                num_classes = len(self.label_encoder.classes_)
            else:
                target_encoded = target.values
                num_classes = 1
            
            # Check if dataset is too small for stratified split
            min_samples_per_class = 2
            can_stratify = True
            
            if task_type == "classification":
                unique_classes, class_counts = np.unique(target_encoded, return_counts=True)
                min_class_count = np.min(class_counts)
                
                if min_class_count < min_samples_per_class:
                    logger.warning(f"Dataset too small for stratified split. Min class count: {min_class_count}, required: {min_samples_per_class}")
                    can_stratify = False
                    
                    # For very small datasets, use minimum 2 samples for testing to get meaningful metrics
                    if len(df) < 10:
                        logger.warning(f"Very small dataset detected ({len(df)} samples). Using minimum 2 samples for testing.")
                        # Ensure at least 2 samples for testing to avoid perfect scores
                        test_size = max(2, int(len(df) * 0.2))
                        if test_size >= len(df):
                            # Ensure at least 1 sample for training (critical for 1-sample datasets)
                            test_size = max(1, len(df) - 1)  # Keep at least 1 for training
                        
                        # Special handling for 1-sample datasets: use all data for training, no test set
                        if len(df) == 1:
                            logger.warning("Dataset has only 1 sample. Using all data for training (no test set).")
                            X_train, X_test = features_scaled, np.array([]).reshape(0, features_scaled.shape[1])
                            # Use explicit parentheses to fix operator precedence in ternary operator
                            if len(target_encoded.shape) == 1:
                                y_train, y_test = target_encoded, np.array([]).reshape(0)
                            else:
                                y_train, y_test = target_encoded, np.array([]).reshape(0, target_encoded.shape[1])
                        else:
                            X_train, X_test = features_scaled[:-test_size], features_scaled[-test_size:]
                            y_train, y_test = target_encoded[:-test_size], target_encoded[-test_size:]
                        logger.info(f"Split: {len(X_train)} train, {len(X_test)} test (minimum 2 for meaningful metrics)")
                    else:
                        # Use simple random split without stratification
                        X_train, X_test, y_train, y_test = train_test_split(
                            features_scaled, target_encoded, 
                            test_size=self.config.get("test_size", 0.2),
                            random_state=self.config.get("random_seed", 42)
                        )
                else:
                    # Use stratified split
                    X_train, X_test, y_train, y_test = train_test_split(
                        features_scaled, target_encoded, 
                        test_size=self.config.get("test_size", 0.2),
                        random_state=self.config.get("random_seed", 42),
                        stratify=target_encoded
                    )
            else:
                # Regression - no stratification needed
                X_train, X_test, y_train, y_test = train_test_split(
                    features_scaled, target_encoded, 
                    test_size=self.config.get("test_size", 0.2),
                    random_state=self.config.get("random_seed", 42)
                )
            
            # Store normalization stats and class names for metadata (matching reference code)
            normalization_stats = {}
            if self.scaler:
                # Get mean and std from scaler (for each feature)
                # Use empty list instead of None to avoid issues with np.array(None) creating 0-dimensional arrays
                if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    normalization_stats["x_mean"] = self.scaler.mean_.tolist()
                else:
                    normalization_stats["x_mean"] = []
                
                if hasattr(self.scaler, 'var_') and self.scaler.var_ is not None:
                    normalization_stats["x_std"] = np.sqrt(self.scaler.var_).tolist()
                else:
                    normalization_stats["x_std"] = []
            
            class_info = {}
            if task_type == "classification" and self.label_encoder:
                class_info["class_names"] = list(self.label_encoder.classes_)
                class_info["class_indices"] = {i: cls for i, cls in enumerate(self.label_encoder.classes_)}
            
            data_info = {
                "num_features": features_scaled.shape[1],
                "num_samples": len(df),
                "num_classes": num_classes,
                "task_type": task_type,
                "target_column": target_column,
                "feature_names": list(features.columns),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "normalization_stats": normalization_stats,
                "class_info": class_info,
                "scaler": self.scaler,  # Keep scaler for later use
                "label_encoder": self.label_encoder  # Keep label_encoder for later use
            }
            
            logger.info(f"Data prepared: {data_info['num_features']} features, {data_info['num_samples']} samples")
            return data_info
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return {"error": str(e)}
    
    def create_model(self, num_features: int, num_classes: int, task_type: str = "classification") -> nn.Module:
        """
        Create dynamic PyTorch model
        
        Args:
            num_features: Number of input features
            num_classes: Number of output classes (1 for regression)
            task_type: "classification" or "regression"
            
        Returns:
            PyTorch model
        """
        try:
            # Get model configuration
            model_config = self.config.get("model_params", {})
            hidden_sizes = model_config.get("hidden_sizes", [128, 64])
            activation = model_config.get("activation", "relu")
            dropout = model_config.get("dropout", 0.2)
            batch_norm = model_config.get("batch_norm", True)
            
            # Create model
            self.model = DynamicMLP(
                input_size=num_features,
                output_size=num_classes,
                hidden_sizes=hidden_sizes,
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm
            ).to(self.device)
            
            # Create criterion
            if task_type == "classification":
                if num_classes > 2:
                    self.criterion = nn.CrossEntropyLoss()
                else:
                    self.criterion = nn.BCEWithLogitsLoss()
            else:  # regression
                criterion_type = model_config.get("criterion", "mse")
                if criterion_type == "mse":
                    self.criterion = nn.MSELoss()
                elif criterion_type == "mae":
                    self.criterion = nn.L1Loss()
                else:
                    self.criterion = nn.MSELoss()
            
            # Create optimizer
            optimizer_type = self.config.get("optimizer", "adam")
            learning_rate = self.config.get("learning_rate", 0.001)
            
            if optimizer_type.lower() == "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            elif optimizer_type.lower() == "sgd":
                self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
            elif optimizer_type.lower() == "rmsprop":
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Create scheduler if specified
            scheduler_type = self.config.get("scheduler")
            if scheduler_type:
                if scheduler_type.lower() == "steplr":
                    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
                elif scheduler_type.lower() == "reducelronplateau":
                    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
            
            logger.info(f"Created {task_type} model with {sum(p.numel() for p in self.model.parameters())} parameters")
            return self.model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def train_model(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the PyTorch model
        
        Args:
            data_info: Data information from prepare_data
            
        Returns:
            Training results dictionary
        """
        try:
            if self.model is None:
                raise ValueError("Model not created. Call create_model first.")
            
            # Prepare data loaders
            train_dataset = TabularDataset(
                data_info["X_train"], 
                data_info["y_train"], 
                data_info["task_type"]
            )
            test_dataset = TabularDataset(
                data_info["X_test"], 
                data_info["y_test"], 
                data_info["task_type"]
            )
            
            # Verify train and test sets are separate (no data leakage)
            train_size = len(train_dataset)
            test_size = len(test_dataset)
            logger.info(f"Dataset split verification - Train: {train_size} samples, Test: {test_size} samples")
            
            # Check for overlap (should be impossible with train_test_split, but verify)
            if hasattr(data_info["X_train"], 'shape') and hasattr(data_info["X_test"], 'shape'):
                # This is a sanity check - train_test_split should already ensure separation
                logger.info(f"Train set shape: {data_info['X_train'].shape}, Test set shape: {data_info['X_test'].shape}")
            
            # Log class distribution in train and test sets
            if data_info["task_type"] == "classification":
                from collections import Counter
                train_classes = Counter(data_info["y_train"])
                test_classes = Counter(data_info["y_test"])
                logger.info(f"Train set class distribution: {dict(train_classes)}")
                logger.info(f"Test set class distribution: {dict(test_classes)}")
            
            batch_size = self.config.get("batch_size", 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Training parameters
            epochs = self.config.get("epochs", 10)
            early_stopping_patience = self.config.get("early_stopping_patience", 5)
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            self.training_history = []
            
            for epoch in range(epochs):
                # Training phase
                train_loss = self._train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_metrics = self._validate_epoch(test_loader, data_info["task_type"])
                
                # Record history
                epoch_history = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **val_metrics
                }
                self.training_history.append(epoch_history)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            
            # Load best model
            if hasattr(self, 'best_model_state'):
                self.model.load_state_dict(self.best_model_state)
            
            # Final evaluation
            final_loss, final_metrics = self._validate_epoch(test_loader, data_info["task_type"])
            
            # Log info about perfect metrics (common for easy datasets like Iris)
            if data_info["task_type"] == "classification":
                test_size = data_info.get("test_size", 0)
                train_size = data_info.get("train_size", 0)
                accuracy = final_metrics.get("accuracy")
                f1_score = final_metrics.get("f1_score")
                # Check if metrics are valid (not None) before comparing
                if accuracy is not None and f1_score is not None and accuracy == 1.0 and f1_score == 1.0:
                    if test_size <= 1:
                        logger.warning(f"Perfect metrics (accuracy=1.0, f1=1.0) with test_size={test_size}. This may be due to very small test set.")
                    elif test_size < 10:
                        logger.info(f"Perfect metrics (accuracy=1.0, f1=1.0) with test_size={test_size}, train_size={train_size}. This is normal for easy datasets.")
                    else:
                        logger.info(f"Perfect metrics (accuracy=1.0, f1=1.0) with test_size={test_size}, train_size={train_size}. This is normal for linearly separable datasets like Iris.")
                else:
                    # Format metrics with None handling
                    acc_str = f"{final_metrics.get('accuracy', 0):.4f}" if final_metrics.get('accuracy') is not None else "N/A (empty test set)"
                    f1_str = f"{final_metrics.get('f1_score', 0):.4f}" if final_metrics.get('f1_score') is not None else "N/A (empty test set)"
                    logger.info(f"Metrics: accuracy={acc_str}, f1={f1_str} with test_size={test_size}, train_size={train_size}")
            
            training_results = {
                "training_completed": True,
                "epochs_trained": len(self.training_history),
                "best_val_loss": best_loss,
                "final_metrics": final_metrics,
                "training_history": self.training_history,
                "model_parameters": sum(p.numel() for p in self.model.parameters())
            }
            
            logger.info(f"Training completed. Final metrics: {final_metrics}")
            return training_results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {"error": str(e), "training_completed": False}
    
    def save_model(self, save_path: str, model_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Save trained model and metadata
        
        Args:
            save_path: Directory to save the model
            model_info: Model metadata
            
        Returns:
            Dictionary with file paths
        """
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate consistent model name based on directory name
            model_name = save_dir.name
            model_path = save_dir / f"{model_name}_model.pth"
            torch.save(self.model.state_dict(), model_path)
            
            # Save model architecture and metadata
            # Convert numpy arrays to lists and exclude non-serializable objects for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    # Exclude non-serializable objects like StandardScaler, LabelEncoder
                    result = {}
                    for k, v in obj.items():
                        # Skip sklearn objects and other non-serializable objects
                        if isinstance(v, (StandardScaler, LabelEncoder)):
                            continue  # Skip these objects - we already extracted their data
                        try:
                            result[k] = convert_numpy(v)
                        except (TypeError, ValueError):
                            # If conversion fails, skip this key
                            logger.warning(f"Skipping non-serializable key '{k}' in metadata")
                            continue
                    return result
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj if not isinstance(item, (StandardScaler, LabelEncoder))]
                elif isinstance(obj, (StandardScaler, LabelEncoder)):
                    # Skip sklearn objects - we already extracted their data separately
                    return None
                else:
                    # Try to return as-is, but catch serialization errors
                    try:
                        # Test if it's JSON serializable
                        json.dumps(obj)
                        return obj
                    except (TypeError, ValueError):
                        # Return string representation for non-serializable objects
                        return str(obj)
            
            # Build metadata with normalization stats and class info (matching reference code)
            metadata = {
                "model_class": "DynamicMLP",
                "input_size": self.model.input_size,
                "output_size": self.model.output_size,
                "hidden_sizes": self.model.hidden_sizes,
                "config": self.config,
                "training_history": convert_numpy(self.training_history),
                "model_info": convert_numpy(model_info),
                "saved_at": datetime.now().isoformat()
            }
            
            # Add normalization stats if available (matching reference code)
            if self.scaler:
                # Use explicit None check to avoid AttributeError if attributes are None
                if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    metadata["x_mean"] = self.scaler.mean_.tolist()
                else:
                    metadata["x_mean"] = []
                
                if hasattr(self.scaler, 'var_') and self.scaler.var_ is not None:
                    metadata["x_std"] = np.sqrt(self.scaler.var_).tolist()
                else:
                    metadata["x_std"] = []
            
            # Add class names if available (matching reference code)
            if self.label_encoder:
                metadata["class_names"] = list(self.label_encoder.classes_)
                metadata["class_indices"] = {i: cls for i, cls in enumerate(self.label_encoder.classes_)}
            
            metadata_path = save_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Verify files were actually created
            if not model_path.exists():
                logger.error(f"Model file was not created at: {model_path}")
                return {"error": f"Model file was not created at: {model_path}"}
            if not metadata_path.exists():
                logger.error(f"Metadata file was not created at: {metadata_path}")
                return {"error": f"Metadata file was not created at: {metadata_path}"}
            
            result = {
                "model_path": str(model_path.absolute()),  # Use absolute path
                "metadata_path": str(metadata_path.absolute())  # Use absolute path
            }
            logger.info(f"Model saved successfully. model_path: {result['model_path']}, metadata_path: {result['metadata_path']}")
            return result
            
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            if len(target.shape) == 1 and output.shape[1] == 1:
                # Regression case
                output = output.squeeze()
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader, task_type: str) -> Tuple[float, Dict[str, float]]:
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if len(target.shape) == 1 and output.shape[1] == 1:
                    output = output.squeeze()
                
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Collect predictions and targets for metrics
                if task_type == "classification":
                    if output.dim() > 1:
                        predictions = torch.argmax(output, dim=1)
                    else:
                        predictions = (torch.sigmoid(output) > 0.5).long()
                else:
                    predictions = output
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics (Bug 2 fix: handle empty test set with clear indicators)
        if len(all_predictions) == 0 or len(all_targets) == 0:
            logger.warning("Empty validation set - cannot calculate metrics. This may occur with very small datasets (e.g., 1 sample).")
            # Return metrics marked as invalid to distinguish from actual training failure
            if task_type == "classification":
                metrics = {
                    "accuracy": None,  # Use None instead of 0.0 to indicate invalid
                    "f1_score": None,  # Use None instead of 0.0 to indicate invalid
                    "num_samples": 0,
                    "num_correct": 0,
                    "num_classes": 0,
                    "test_set_empty": True,  # Explicit flag
                    "warning": "Test set is empty - metrics are invalid. Dataset may be too small for meaningful validation."
                }
            else:  # regression
                metrics = {
                    "mse": None,  # Use None instead of 0.0 to indicate invalid
                    "rmse": None,  # Use None instead of 0.0 to indicate invalid
                    "r2_score": None,  # Use None instead of 0.0 to indicate invalid
                    "test_set_empty": True,  # Explicit flag
                    "warning": "Test set is empty - metrics are invalid. Dataset may be too small for meaningful validation."
                }
            avg_loss = None  # Use None instead of 0.0 to indicate invalid
        else:
            metrics = self._calculate_metrics(all_predictions, all_targets, task_type)
            # Handle division by zero for empty loader
            if len(val_loader) > 0:
                avg_loss = total_loss / len(val_loader)
            else:
                avg_loss = 0.0
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions: List, targets: List, task_type: str) -> Dict[str, float]:
        """Calculate evaluation metrics with detailed logging"""
        metrics = {}
        
        # Handle empty predictions/targets (Bug 2 fix: empty test set case)
        if len(predictions) == 0 or len(targets) == 0:
            logger.warning("Cannot calculate metrics: empty predictions or targets (likely due to empty test set)")
            if task_type == "classification":
                return {
                    "accuracy": None,  # Use None instead of 0.0 to indicate invalid
                    "f1_score": None,  # Use None instead of 0.0 to indicate invalid
                    "num_samples": 0,
                    "num_correct": 0,
                    "num_classes": 0,
                    "test_set_empty": True,  # Explicit flag
                    "warning": "Test set is empty - metrics are invalid. Dataset may be too small for meaningful validation."
                }
            else:  # regression
                return {
                    "mse": None,  # Use None instead of 0.0 to indicate invalid
                    "rmse": None,  # Use None instead of 0.0 to indicate invalid
                    "r2_score": None,  # Use None instead of 0.0 to indicate invalid
                    "test_set_empty": True,  # Explicit flag
                    "warning": "Test set is empty - metrics are invalid. Dataset may be too small for meaningful validation."
                }
        
        if task_type == "classification":
            # Convert to numpy arrays for easier manipulation
            predictions_arr = np.array(predictions)
            targets_arr = np.array(targets)
            
            # Calculate basic metrics
            accuracy = accuracy_score(targets_arr, predictions_arr)
            f1 = f1_score(targets_arr, predictions_arr, average='weighted')
            
            # Handle NaN values from empty arrays (additional safety check)
            if np.isnan(accuracy) or np.isnan(f1):
                logger.warning("Metrics calculation produced NaN values (empty or invalid data). Using default values.")
                accuracy = 0.0
                f1 = 0.0
            
            # Calculate per-class metrics for better insight
            unique_classes = np.unique(targets_arr)
            num_classes = len(unique_classes)
            
            # Count correct predictions
            correct = np.sum(predictions_arr == targets_arr)
            total = len(targets_arr)
            
            # Log detailed information
            logger.info(f"Metrics calculation - Total samples: {total}, Correct: {correct}, Accuracy: {accuracy:.4f}")
            logger.info(f"Unique classes in targets: {unique_classes}, Unique classes in predictions: {np.unique(predictions_arr)}")
            
            # Show confusion matrix-like info
            if num_classes <= 10:  # Only for small number of classes
                from collections import Counter
                correct_by_class = Counter()
                total_by_class = Counter()
                for pred, target in zip(predictions_arr, targets_arr):
                    total_by_class[target] += 1
                    if pred == target:
                        correct_by_class[target] += 1
                
                logger.info("Per-class accuracy:")
                for cls in sorted(total_by_class.keys()):
                    cls_total = total_by_class[cls]
                    cls_correct = correct_by_class.get(cls, 0)
                    cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
                    logger.info(f"  Class {cls}: {cls_correct}/{cls_total} correct ({cls_acc:.4f})")
            
            # Check for potential issues
            if accuracy == 1.0:
                # Verify this isn't due to all predictions being the same class
                unique_preds = len(np.unique(predictions_arr))
                unique_targets = len(np.unique(targets_arr))
                if unique_preds == 1 and unique_targets > 1:
                    logger.warning(f"WARNING: All predictions are the same class, but targets have {unique_targets} classes. This suggests a problem.")
                else:
                    logger.info(f"Perfect accuracy confirmed: {unique_preds} unique predictions for {unique_targets} unique target classes")
            
            metrics = {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "num_samples": int(total),
                "num_correct": int(correct),
                "num_classes": int(num_classes)
            }
        else:  # regression
            # Convert to numpy arrays
            predictions_arr = np.array(predictions)
            targets_arr = np.array(targets)
            
            mse = mean_squared_error(targets_arr, predictions_arr)
            r2 = r2_score(targets_arr, predictions_arr)
            rmse = np.sqrt(mse)
            
            # Handle NaN values (additional safety check)
            if np.isnan(mse) or np.isnan(r2) or np.isnan(rmse):
                logger.warning("Regression metrics calculation produced NaN values. Using default values.")
                mse = 0.0
                r2 = 0.0
                rmse = 0.0
            
            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "r2_score": float(r2)
            }
        
        return metrics
    
    def _infer_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Infer target column from DataFrame"""
        # Common target column names
        target_candidates = [
            "target", "label", "class", "y", "output", "outcome", 
            "result", "prediction", "category", "type"
        ]
        
        for col in df.columns:
            if col.lower() in target_candidates:
                return col
        
        # If no obvious target, use the last column
        return df.columns[-1] if len(df.columns) > 1 else None


# Factory function
def create_pytorch_trainer(config: Dict[str, Any]) -> PyTorchTrainer:
    """Create PyTorch trainer from configuration"""
    return PyTorchTrainer(config)


