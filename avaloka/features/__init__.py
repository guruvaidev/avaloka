"""Evidence-preserving feature engineering."""

from .core import (FeatureRecipe, FeatureResult, JoinDiagnostics, TransformStep,
                   execute_feature_recipe, normalize_identifier,
                   parse_structured_string, project_nested, verified_join)

__all__ = [
    "FeatureRecipe", "FeatureResult", "JoinDiagnostics", "TransformStep",
    "execute_feature_recipe", "normalize_identifier", "parse_structured_string",
    "project_nested", "verified_join",
]
