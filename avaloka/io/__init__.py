"""Source materialization and defensive dataset ingestion."""

from avaloka.io.complex import (ComplexDataError, ComplexDataOptions,
                                MissingCapabilityError,
                                complex_data_capabilities,
                                detect_complex_format, load_complex_data,
                                normalize_rlds_episodes)
from avaloka.io.loader import (DatasetHandle, load_dataset, normalize_records,
                               normalize_text)
from avaloka.io.schema import (SchemaIssue, SchemaValidationResult,
                               infer_complex_schema,
                               validate_complex_loader_code,
                               validate_complex_schema,
                               validate_lerobot_frames,
                               validate_rlds_episodes)
from avaloka.io.sources import (ResolvedSource, SourceError, materialize,
                                parse_uri, route)

__all__ = [
    "DatasetHandle",
    "ComplexDataError",
    "ComplexDataOptions",
    "MissingCapabilityError",
    "complex_data_capabilities",
    "detect_complex_format",
    "load_complex_data",
    "normalize_rlds_episodes",
    "load_dataset",
    "normalize_records",
    "normalize_text",
    "SchemaIssue",
    "SchemaValidationResult",
    "infer_complex_schema",
    "validate_complex_schema",
    "validate_complex_loader_code",
    "validate_lerobot_frames",
    "validate_rlds_episodes",
    "ResolvedSource",
    "SourceError",
    "materialize",
    "parse_uri",
    "route",
]
