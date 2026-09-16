"""
Configuration Management for Model Training Agent

Provides configuration handling following Hydra patterns and integrating
with the existing Avaloka project structure.
"""

import logging
import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Basic training parameters
    model_type: str = "pytorch_classification"
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    random_seed: int = 42
    
    # Validation parameters
    validation_split: float = 0.2
    k_folds: int = 5
    metric_threshold: float = 0.8
    early_stopping_patience: int = 5
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Optimizer parameters
    optimizer: str = "adam"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler parameters
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure and compute resources"""
    
    # Compute resources
    use_gpu: bool = False
    num_workers: int = 1
    memory_limit: str = "4Gi"
    cpu_limit: str = "2"
    
    # Ray configuration
    use_ray: bool = False
    ray_num_workers: int = 2
    ray_resources_per_worker: Dict[str, Any] = field(default_factory=lambda: {"CPU": 1, "memory": 2000000000})
    
    # Storage configuration
    model_storage_path: str = "./models"
    experiment_storage_path: str = "./experiments"
    
    # Kubernetes configuration
    use_k8s: bool = False
    k8s_namespace: str = "default"
    k8s_service_account: Optional[str] = None


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking"""
    
    # MLflow settings
    enabled: bool = True
    tracking_uri: Optional[str] = None
    experiment_name: str = "avaloka_mta"
    
    # GCP Configuration
    use_gcp: bool = False
    gcp_project_id: Optional[str] = None
    gcp_backend_store_uri: Optional[str] = None
    gcp_artifact_root: Optional[str] = None
    
    # Artifact logging
    log_models: bool = True
    log_code: bool = True
    log_datasets: bool = True
    
    # Model registry
    model_registry_uri: Optional[str] = None
    register_best_model: bool = True


@dataclass
class MTAConfig:
    """Complete MTA configuration"""
    
    # Configuration metadata
    config_name: str = "default"
    config_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Sub-configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    
    # Task-specific overrides
    task_overrides: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages configuration for the Model Training Agent"""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Create default config files if they don't exist
        self._create_default_configs()
    
    def load_config(self, config_name: str = "default") -> MTAConfig:
        """
        Load configuration from file
        
        Args:
            config_name: Name of the configuration to load
            
        Returns:
            MTAConfig instance
        """
        try:
            config_path = self.config_dir / f"{config_name}.yaml"
            
            if not config_path.exists():
                logger.warning(f"Config file {config_path} not found, using default config")
                return MTAConfig(config_name=config_name)
            
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert dict to MTAConfig
            return self._dict_to_config(config_dict, config_name)
            
        except Exception as e:
            logger.error(f"Failed to load config {config_name}: {e}")
            return MTAConfig(config_name=config_name)
    
    def save_config(self, config: MTAConfig, config_name: Optional[str] = None) -> bool:
        """
        Save configuration to file
        
        Args:
            config: MTAConfig instance to save
            config_name: Name for the configuration file (uses config.config_name if None)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            name = config_name or config.config_name
            config_path = self.config_dir / f"{name}.yaml"
            
            # Convert config to dict
            config_dict = self._config_to_dict(config)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, indent=2, default_flow_style=False)
            
            logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def create_task_config(self, base_config: str = "default", **overrides) -> MTAConfig:
        """
        Create configuration for a specific task with overrides
        
        Args:
            base_config: Base configuration to start from
            **overrides: Configuration overrides
            
        Returns:
            MTAConfig with task-specific settings
        """
        try:
            config = self.load_config(base_config)
            
            # Apply overrides
            for key, value in overrides.items():
                self._apply_override(config, key, value)
            
            # Store overrides for reference
            config.task_overrides = overrides
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to create task config: {e}")
            return MTAConfig()
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific model type
        
        Args:
            model_type: Type of model (e.g., 'pytorch_classification')
            
        Returns:
            Model-specific configuration dictionary
        """
        model_configs = {
            "pytorch_classification": {
                "hidden_sizes": [128, 64],
                "activation": "relu",
                "dropout": 0.2,
                "batch_norm": True,
                "criterion": "cross_entropy"
            },
            "pytorch_regression": {
                "hidden_sizes": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.1,
                "batch_norm": True,
                "criterion": "mse"
            },
            "svm": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True
            },
            "gam": {
                "n_splines": 20,
                "spline_order": 3,
                "lam": 0.6,
                "max_iter": 100
            },
            "liquid_nn": {
                "num_units": 128,
                "time_constant": 0.1,
                "sensory_sigma": 0.1,
                "motor_sigma": 0.1
            }
        }
        
        return model_configs.get(model_type, {})
    
    def _create_default_configs(self):
        """Create default configuration files if they don't exist"""
        default_config_path = self.config_dir / "default.yaml"
        
        if not default_config_path.exists():
            default_config = MTAConfig()
            self.save_config(default_config, "default")
    
    def _dict_to_config(self, config_dict: Dict[str, Any], config_name: str) -> MTAConfig:
        """Convert dictionary to MTAConfig"""
        try:
            # Extract sub-configurations
            training_dict = config_dict.get("training", {})
            infrastructure_dict = config_dict.get("infrastructure", {})
            mlflow_dict = config_dict.get("mlflow", {})
            
            # Create sub-config objects
            training_config = TrainingConfig(**training_dict)
            infrastructure_config = InfrastructureConfig(**infrastructure_dict)
            mlflow_config = MLflowConfig(**mlflow_dict)
            
            # Create main config
            main_config = MTAConfig(
                config_name=config_name,
                config_version=config_dict.get("config_version", "1.0.0"),
                created_at=config_dict.get("created_at", datetime.now().isoformat()),
                training=training_config,
                infrastructure=infrastructure_config,
                mlflow=mlflow_config,
                task_overrides=config_dict.get("task_overrides", {})
            )
            
            return main_config
            
        except Exception as e:
            logger.error(f"Error converting dict to config: {e}")
            return MTAConfig(config_name=config_name)
    
    def _config_to_dict(self, config: MTAConfig) -> Dict[str, Any]:
        """Convert MTAConfig to dictionary"""
        return {
            "config_name": config.config_name,
            "config_version": config.config_version,
            "created_at": config.created_at,
            "training": {
                "model_type": config.training.model_type,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "epochs": config.training.epochs,
                "random_seed": config.training.random_seed,
                "validation_split": config.training.validation_split,
                "k_folds": config.training.k_folds,
                "metric_threshold": config.training.metric_threshold,
                "early_stopping_patience": config.training.early_stopping_patience,
                "model_params": config.training.model_params,
                "optimizer": config.training.optimizer,
                "optimizer_params": config.training.optimizer_params,
                "scheduler": config.training.scheduler,
                "scheduler_params": config.training.scheduler_params
            },
            "infrastructure": {
                "use_gpu": config.infrastructure.use_gpu,
                "num_workers": config.infrastructure.num_workers,
                "memory_limit": config.infrastructure.memory_limit,
                "cpu_limit": config.infrastructure.cpu_limit,
                "use_ray": config.infrastructure.use_ray,
                "ray_num_workers": config.infrastructure.ray_num_workers,
                "ray_resources_per_worker": config.infrastructure.ray_resources_per_worker,
                "model_storage_path": config.infrastructure.model_storage_path,
                "experiment_storage_path": config.infrastructure.experiment_storage_path,
                "use_k8s": config.infrastructure.use_k8s,
                "k8s_namespace": config.infrastructure.k8s_namespace,
                "k8s_service_account": config.infrastructure.k8s_service_account
            },
            "mlflow": {
                "enabled": config.mlflow.enabled,
                "tracking_uri": config.mlflow.tracking_uri,
                "experiment_name": config.mlflow.experiment_name,
                "use_gcp": config.mlflow.use_gcp,
                "gcp_project_id": config.mlflow.gcp_project_id,
                "gcp_backend_store_uri": config.mlflow.gcp_backend_store_uri,
                "gcp_artifact_root": config.mlflow.gcp_artifact_root,
                "log_models": config.mlflow.log_models,
                "log_code": config.mlflow.log_code,
                "log_datasets": config.mlflow.log_datasets,
                "model_registry_uri": config.mlflow.model_registry_uri,
                "register_best_model": config.mlflow.register_best_model
            },
            "task_overrides": config.task_overrides
        }
    
    def _apply_override(self, config: MTAConfig, key: str, value: Any):
        """Apply configuration override using dot notation"""
        try:
            # Handle dot notation (e.g., "training.learning_rate")
            parts = key.split(".")
            
            if len(parts) == 1:
                # Top-level override
                setattr(config, key, value)
            elif len(parts) == 2:
                # Sub-config override
                sub_config_name, param_name = parts
                sub_config = getattr(config, sub_config_name)
                setattr(sub_config, param_name, value)
            else:
                logger.warning(f"Override key {key} too complex, skipping")
                
        except Exception as e:
            logger.error(f"Failed to apply override {key}={value}: {e}")


# Factory functions
def create_default_config() -> MTAConfig:
    """Create default MTA configuration"""
    return MTAConfig()


def create_config_from_task(task) -> MTAConfig:
    """
    Create MTA configuration from training task
    
    Args:
        task: Training task with validated structure (TrainingTask or dict)
        
    Returns:
        MTAConfig configured for the task
    """
    config = MTAConfig()
    
    # Apply task-specific settings with type safety
    config.training.model_type = task.model_type
    config.training.task_type = task.task_type
    config.training.training_strategy = task.training_strategy
    
    # Apply hyperparameters if provided
    if task.hyperparameters:
        hyperparams = task.hyperparameters
        if hyperparams.learning_rate is not None:
            config.training.learning_rate = hyperparams.learning_rate
        if hyperparams.epochs is not None:
            config.training.epochs = hyperparams.epochs
        if hyperparams.batch_size is not None:
            config.training.batch_size = hyperparams.batch_size
        if hyperparams.dropout_rate is not None:
            config.training.dropout_rate = hyperparams.dropout_rate
        if hyperparams.weight_decay is not None:
            config.training.weight_decay = hyperparams.weight_decay
        if hyperparams.momentum is not None:
            config.training.momentum = hyperparams.momentum
    
    # Apply Ray configuration if enabled
    if task.use_ray and task.ray_config:
        config.infrastructure.use_ray = True
        ray_config = task.ray_config
        if ray_config.num_workers is not None:
            config.infrastructure.ray_num_workers = ray_config.num_workers
        if ray_config.use_gpu is not None:
            config.infrastructure.use_gpu = ray_config.use_gpu
        if ray_config.memory_per_worker is not None:
            config.infrastructure.ray_memory_per_worker = ray_config.memory_per_worker
        if ray_config.cpu_per_worker is not None:
            config.infrastructure.ray_cpu_per_worker = ray_config.cpu_per_worker
    
    # Apply validation parameters if provided
    if task.validation_params:
        val_params = task.validation_params
        if val_params.k_folds is not None:
            config.training.k_folds = val_params.k_folds
        if val_params.test_size is not None:
            config.training.test_size = val_params.test_size
        if val_params.validation_size is not None:
            config.training.validation_size = val_params.validation_size
        if val_params.metric_threshold is not None:
            config.training.metric_threshold = val_params.metric_threshold
        if val_params.early_stopping_patience is not None:
            config.training.early_stopping_patience = val_params.early_stopping_patience
    
    # Apply data configuration (store in task_overrides since MTAConfig doesn't have data attribute)
    if task.data_config:
        config.task_overrides['data_source'] = task.data_config.data_source
        if task.data_config.target_column:
            config.task_overrides['target_column'] = task.data_config.target_column
        if task.data_config.feature_columns:
            config.task_overrides['feature_columns'] = task.data_config.feature_columns
    
    # Apply training goals and metrics
    if task.primary_metric:
        config.training.primary_metric = task.primary_metric
    if task.training_goals:
        config.training.training_goals = task.training_goals
    
    return config

