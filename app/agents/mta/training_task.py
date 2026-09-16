"""
Training Task Pydantic Models for Model Training Agent

Defines structured data models for training tasks with validation,
type safety, and clear field definitions.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from datetime import datetime

from .model_types import ModelType, TaskType, TrainingStrategy


class HyperparameterConfig(BaseModel):
    """Configuration for model hyperparameters"""
    
    learning_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Learning rate for training")
    epochs: Optional[int] = Field(None, ge=1, le=10000, description="Number of training epochs")
    batch_size: Optional[int] = Field(None, ge=1, le=10000, description="Batch size for training")
    dropout_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dropout rate for regularization")
    weight_decay: Optional[float] = Field(None, ge=0.0, description="Weight decay for regularization")
    momentum: Optional[float] = Field(None, ge=0.0, le=1.0, description="Momentum for optimizer")
    
    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Learning rate must be positive')
        return v


class ValidationConfig(BaseModel):
    """Configuration for model validation"""
    
    k_folds: Optional[int] = Field(None, ge=2, le=20, description="Number of k-fold cross-validation folds")
    test_size: Optional[float] = Field(None, ge=0.0, le=1.0, description="Proportion of data for testing")
    validation_size: Optional[float] = Field(None, ge=0.0, le=1.0, description="Proportion of data for validation")
    metric_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum metric threshold for success")
    early_stopping_patience: Optional[int] = Field(None, ge=1, description="Patience for early stopping")
    
    @field_validator('test_size', 'validation_size')
    @classmethod
    def validate_split_sizes(cls, v):
        if v is not None and (v <= 0 or v >= 1):
            raise ValueError('Split sizes must be between 0 and 1')
        return v


class RayConfig(BaseModel):
    """Configuration for Ray distributed training"""
    
    num_workers: Optional[int] = Field(None, ge=1, le=100, description="Number of Ray workers")
    use_gpu: Optional[bool] = Field(False, description="Whether to use GPU for training")
    memory_per_worker: Optional[str] = Field(None, description="Memory allocation per worker")
    cpu_per_worker: Optional[float] = Field(None, ge=0.1, description="CPU allocation per worker")
    
    @field_validator('memory_per_worker')
    @classmethod
    def validate_memory_format(cls, v):
        if v is not None and not v.endswith(('MB', 'GB', 'TB')):
            raise ValueError('Memory must be specified with unit (MB, GB, TB)')
        return v


class DataConfig(BaseModel):
    """Configuration for training data"""
    
    data_source: str = Field(..., description="Path to training data")
    target_column: Optional[str] = Field(None, description="Name of target column for supervised learning")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature column names")
    data_preprocessing: Optional[Dict[str, Any]] = Field(None, description="Data preprocessing configuration")
    data_validation: Optional[Dict[str, Any]] = Field(None, description="Data validation rules")
    
    @field_validator('data_source')
    @classmethod
    def validate_data_source(cls, v):
        if not v or not v.strip():
            raise ValueError('Data source cannot be empty')
        return v.strip()


class TrainingTask(BaseModel):
    """
    Comprehensive training task definition with validation
    
    This model replaces the generic Dict[str, Any] with a structured,
    validated representation of training tasks.
    """
    
    # Core task identification
    task_id: str = Field(..., description="Unique identifier for the training task")
    task_name: Optional[str] = Field(None, description="Human-readable name for the task")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation timestamp")
    
    # Model and task configuration
    model_name: str = Field(..., description="Name of the model to train")
    model_description: str = Field(..., description="Description of the model")
    model_version: str = Field(..., description="Version of the model")
    model_type: ModelType = Field(..., description="Type of ML model to train")
    task_type: TaskType = Field(..., description="Type of ML task (classification, regression, etc.)")
    training_strategy: TrainingStrategy = Field(
        default=TrainingStrategy.SINGLE_MODEL, 
        description="Training strategy to use"
    )
    
    # Data configuration
    data_config: DataConfig = Field(..., description="Configuration for training data")
    
    # Training parameters
    hyperparameters: Optional[HyperparameterConfig] = Field(None, description="Model hyperparameters")
    validation_params: Optional[ValidationConfig] = Field(None, description="Validation configuration")
    
    # Infrastructure configuration
    ray_config: Optional[RayConfig] = Field(None, description="Ray distributed training configuration")
    use_ray: bool = Field(False, description="Whether to use Ray for distributed training")
    
    # Training goals and metrics
    primary_metric: Optional[str] = Field(None, description="Primary metric to optimize")
    success_criteria: Optional[Dict[str, Any]] = Field(None, description="Criteria for training success")
    training_goals: Optional[List[str]] = Field(None, description="List of training objectives")
    
    # Metadata
    source: Optional[str] = Field(None, description="Source of the training task (CGA, manual, etc.)")
    description: Optional[str] = Field(None, description="Description of the training task")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the task")
    
    # Execution context
    execution_context: Optional[Dict[str, Any]] = Field(None, description="Additional execution context")
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"  # Prevent additional fields
    )
    
    @field_validator('task_id')
    @classmethod
    def validate_task_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Task ID cannot be empty')
        return v.strip()
    
    @field_validator('hyperparameters')
    @classmethod
    def validate_hyperparameters(cls, v):
        if v is not None:
            # Additional validation logic can be added here
            pass
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility"""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingTask':
        """Create from dictionary for backward compatibility"""
        return cls(**data)
    
    def get_effective_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters with defaults applied"""
        if self.hyperparameters is None:
            return {}
        
        return {
            k: v for k, v in self.hyperparameters.model_dump().items() 
            if v is not None
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration with defaults"""
        if self.validation_params is None:
            return {}
        
        return {
            k: v for k, v in self.validation_params.model_dump().items() 
            if v is not None
        }
    
    def get_ray_config(self) -> Dict[str, Any]:
        """Get Ray configuration if enabled"""
        if not self.use_ray or self.ray_config is None:
            return {}
        
        return {
            k: v for k, v in self.ray_config.model_dump().items() 
            if v is not None
        }


class TrainingTaskSummary(BaseModel):
    """Summary view of training task for display purposes"""
    
    task_id: str
    task_name: Optional[str]
    model_type: ModelType
    model_name: str
    model_description: str
    model_version: str
    task_type: TaskType
    data_source: str
    created_at: datetime
    status: Optional[str] = None
    
    model_config = ConfigDict(use_enum_values=True)
