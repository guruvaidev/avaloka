"""
Error types and handling for Model Training Agent

This module provides structured error types using enums to avoid
string matching and enable better error collection and reporting.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class MTAErrorType(Enum):
    """Enumeration of all possible MTA error types"""
    
    # Validation errors
    VALIDATION_MISSING_FIELDS = "validation_missing_fields"
    VALIDATION_UNSUPPORTED_MODEL = "validation_unsupported_model"
    VALIDATION_GENERAL = "validation_general"
    
    # Data errors
    DATA_SOURCE_NOT_FOUND = "data_source_not_found"
    DATA_PREPARATION_FAILED = "data_preparation_failed"
    DATA_TOO_SMALL = "data_too_small"
    DATA_INVALID_FORMAT = "data_invalid_format"
    
    # Training errors
    TRAINING_EXECUTION_FAILED = "training_execution_failed"
    TRAINING_MODEL_CONVERGENCE = "training_model_convergence"
    TRAINING_RESOURCE_ERROR = "training_resource_error"
    
    # Planning errors
    PLANNING_FAILED = "planning_failed"
    PLANNING_INVALID_CONFIG = "planning_invalid_config"
    
    # Environment errors
    ENV_SETUP_FAILED = "env_setup_failed"
    ENV_MISSING_DEPENDENCIES = "env_missing_dependencies"
    
    # MLflow errors
    MLFLOW_EXPERIMENT_SETUP = "mlflow_experiment_setup"
    MLFLOW_LOGGING_FAILED = "mlflow_logging_failed"
    
    # Task errors
    TASK_CREATION_FAILED = "task_creation_failed"
    TASK_INVALID_STATE = "task_invalid_state"
    
    # Model errors
    MODEL_SAVE_FAILED = "model_save_failed"
    MODEL_EXPORT_FAILED = "model_export_failed"
    
    # General errors
    UNKNOWN_ERROR = "unknown_error"


class MTAErrorSeverity(Enum):
    """Error severity levels"""
    WARNING = "warning"  # Can continue with degraded functionality
    ERROR = "error"      # Cannot continue, but recoverable
    CRITICAL = "critical"  # Fatal error, cannot recover


@dataclass
class MTAError:
    """Structured error object for MTA operations"""
    error_type: MTAErrorType
    message: str
    severity: MTAErrorSeverity = MTAErrorSeverity.ERROR
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format"""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "severity": self.severity.value,
            "suggestion": self.suggestion,
            "details": self.details or {}
        }
    
    def __str__(self) -> str:
        """String representation for logging"""
        parts = [f"[{self.severity.value.upper()}] {self.error_type.value}: {self.message}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return "\n".join(parts)


class MTAErrorCollection:
    """Collection to accumulate multiple errors"""
    
    def __init__(self):
        self.errors: List[MTAError] = []
    
    def add_error(
        self,
        error_type: MTAErrorType,
        message: str,
        severity: MTAErrorSeverity = MTAErrorSeverity.ERROR,
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an error to the collection"""
        error = MTAError(
            error_type=error_type,
            message=message,
            severity=severity,
            suggestion=suggestion,
            details=details
        )
        self.errors.append(error)
    
    def add_mta_error(self, error: MTAError) -> None:
        """Add an MTAError object directly"""
        self.errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if collection has any errors"""
        return len(self.errors) > 0
    
    def has_critical_errors(self) -> bool:
        """Check if collection has critical errors"""
        return any(e.severity == MTAErrorSeverity.CRITICAL for e in self.errors)
    
    def get_errors_by_severity(self, severity: MTAErrorSeverity) -> List[MTAError]:
        """Get errors filtered by severity"""
        return [e for e in self.errors if e.severity == severity]
    
    def get_errors_by_type(self, error_type: MTAErrorType) -> List[MTAError]:
        """Get errors filtered by type"""
        return [e for e in self.errors if e.error_type == error_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary"""
        return {
            "has_errors": self.has_errors(),
            "error_count": len(self.errors),
            "has_critical": self.has_critical_errors(),
            "errors": [e.to_dict() for e in self.errors]
        }
    
    def to_summary_string(self) -> str:
        """Generate a human-readable summary of all errors"""
        if not self.has_errors():
            return "No errors"
        
        lines = [f"Total Errors: {len(self.errors)}"]
        lines.append("=" * 50)
        
        # Group by severity
        for severity in MTAErrorSeverity:
            severity_errors = self.get_errors_by_severity(severity)
            if severity_errors:
                lines.append(f"\n{severity.value.upper()} ({len(severity_errors)}):")
                for i, error in enumerate(severity_errors, 1):
                    lines.append(f"\n{i}. {error.error_type.value}")
                    lines.append(f"   Message: {error.message}")
                    if error.suggestion:
                        lines.append(f"   Suggestion: {error.suggestion}")
                    if error.details:
                        lines.append(f"   Details: {error.details}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all errors"""
        self.errors = []


def create_error_result(
    error_type: MTAErrorType,
    message: str,
    severity: MTAErrorSeverity = MTAErrorSeverity.ERROR,
    suggestion: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    **additional_fields
) -> Dict[str, Any]:
    """
    Create a standardized error result dictionary
    
    This replaces the old pattern of returning {"error": "message"}
    with a structured error object.
    """
    error = MTAError(
        error_type=error_type,
        message=message,
        severity=severity,
        suggestion=suggestion,
        details=details
    )
    
    result = {
        "success": False,
        "error": error.to_dict(),
        **additional_fields
    }
    
    return result


def create_success_result(**fields) -> Dict[str, Any]:
    """Create a standardized success result dictionary"""
    return {
        "success": True,
        "error": None,
        **fields
    }

