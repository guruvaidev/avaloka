"""
Model Type Enumerations for Model Training Agent

This module defines all supported model types as enums to ensure
type safety and exhaustive validation.
"""

from enum import Enum
from typing import List, Set


class ModelType(Enum):
    """Enumeration of all supported model types"""
    
    # PyTorch Models
    PYTORCH_CLASSIFICATION = "pytorch_classification"
    PYTORCH_REGRESSION = "pytorch_regression"
    
    # Traditional ML Models
    SVM = "svm"
    GAM = "gam"
    
    # Advanced Models
    LIQUID_NN = "liquid_nn"
    
    @classmethod
    def from_string(cls, model_type_str: str) -> 'ModelType':
        """
        Convert string to ModelType enum
        
        Args:
            model_type_str: String representation of model type
            
        Returns:
            ModelType enum
            
        Raises:
            ValueError: If model type is not supported
        """
        try:
            return cls(model_type_str.lower())
        except ValueError:
            valid_types = [mt.value for mt in cls]
            raise ValueError(
                f"Unsupported model type: '{model_type_str}'. "
                f"Valid types: {', '.join(valid_types)}"
            )
    
    @classmethod
    def is_valid(cls, model_type_str: str) -> bool:
        """Check if a model type string is valid"""
        try:
            cls.from_string(model_type_str)
            return True
        except ValueError:
            return False
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get list of all supported model types"""
        return [mt.value for mt in cls]
    
    @classmethod
    def get_pytorch_types(cls) -> List[str]:
        """Get list of PyTorch model types"""
        return [
            cls.PYTORCH_CLASSIFICATION.value,
            cls.PYTORCH_REGRESSION.value
        ]
    
    @classmethod
    def get_traditional_ml_types(cls) -> List[str]:
        """Get list of traditional ML model types"""
        return [
            cls.SVM.value,
            cls.GAM.value
        ]
    
    @classmethod
    def get_advanced_types(cls) -> List[str]:
        """Get list of advanced model types"""
        return [cls.LIQUID_NN.value]
    
    def is_pytorch(self) -> bool:
        """Check if this is a PyTorch model"""
        return self in [ModelType.PYTORCH_CLASSIFICATION, ModelType.PYTORCH_REGRESSION]
    
    def is_regression(self) -> bool:
        """Check if this is a regression model"""
        return self == ModelType.PYTORCH_REGRESSION
    
    def is_classification(self) -> bool:
        """Check if this is a classification model"""
        return self in [ModelType.PYTORCH_CLASSIFICATION, ModelType.SVM]


class TaskType(Enum):
    """Enumeration of ML task types"""
    
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    
    @classmethod
    def from_string(cls, task_type_str: str) -> 'TaskType':
        """Convert string to TaskType enum"""
        try:
            return cls(task_type_str.lower())
        except ValueError:
            valid_types = [tt.value for tt in cls]
            raise ValueError(
                f"Unsupported task type: '{task_type_str}'. "
                f"Valid types: {', '.join(valid_types)}"
            )
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get list of all supported task types"""
        return [tt.value for tt in cls]


class TrainingStrategy(Enum):
    """Enumeration of training strategies"""
    
    SINGLE_MODEL = "single_model"  # Train one model
    COMPETITION = "competition"     # Try multiple models, pick best
    ENSEMBLE = "ensemble"          # Combine multiple models
    
    @classmethod
    def from_string(cls, strategy_str: str) -> 'TrainingStrategy':
        """Convert string to TrainingStrategy enum"""
        try:
            return cls(strategy_str.lower())
        except ValueError:
            valid_strategies = [s.value for s in cls]
            raise ValueError(
                f"Unsupported training strategy: '{strategy_str}'. "
                f"Valid strategies: {', '.join(valid_strategies)}"
            )


# Helper functions for backward compatibility
def validate_model_type(model_type: str) -> bool:
    """Validate if model type is supported"""
    return ModelType.is_valid(model_type)


def get_supported_model_types() -> List[str]:
    """Get list of all supported model types"""
    return ModelType.get_all_types()


def validate_task_type(task_type: str) -> bool:
    """Validate if task type is supported"""
    try:
        TaskType.from_string(task_type)
        return True
    except ValueError:
        return False


def get_supported_task_types() -> List[str]:
    """Get list of all supported task types"""
    return TaskType.get_all_types()

