from typing_extensions import NotRequired, TypedDict
from typing import Literal, List, Dict, Optional, Any

class HyperparameterConfig(TypedDict):
    learning_rate: float                                = 0.001
    epochs: int                                         = 30
    batch_size: int                                     = 32
    optimizer_name: Literal["adam", "sgd", "rmsprop"]   = "adam"
    random_seed: int                                    = 42
    early_stopping_patience: int                        = 5
    early_stopping_min_delta: NotRequired[float]
    hidden_layer_sizes: List[int]                       = [128, 64]
    activation: Literal["relu", "tanh", "sigmoid"]      = "relu"
    dropout_rate: float                                 = 0.2
    batch_norm: bool                                    = True

class RayConfig(TypedDict):
    num_workers: int                                    = 4
    use_gpu: bool                                       = False
    memory_per_worker: str                              = "16Gi"
    cpu_per_worker: int                                 = 4

class DataConfig(TypedDict):
    dataset_uri: str
    feature_columns: List[str]
    target_column: str
    excluded_identifier_columns: NotRequired[List[str]]
    # Ordering metadata, not a model input. When present, MTA sorts by this
    # column and uses the final expanding-window TimeSeriesSplit fold.
    time_column: NotRequired[str]
    pii_key_required_columns: NotRequired[List[str]]
    # Safe model-side contract only: column -> {kind, strategy}. The secret and
    # source values are never stored in the plan or model metadata.
    pii_transformations: NotRequired[Dict[str, Dict[str, str]]]

class TrainingPlan(TypedDict):
    model_type: Literal["classification", "regression"] = "classification"
    model_name: str
    model_description: str
    model_version: str

    hyperparameter_config: HyperparameterConfig
    ray_config: Optional[RayConfig] = None
    data_config: DataConfig
    # The executor is selected by the planner from data accessibility and
    # size, rather than inferred later solely from the storage URI.
    execution_backend: NotRequired[Literal["local", "ray"]]
    execution_reason: NotRequired[str]
    execution_dataset_size_bytes: NotRequired[Optional[int]]
    execution_dataset_uri: NotRequired[str]

class TrainingFailure(TypedDict, total=False):
    code: str
    title: str
    summary: str
    actions: List[str]
    user_message: str
    affected_columns: List[str]
    duplicate_count: Optional[int]
    duplicate_ratio: Optional[float]
    excluded_identifier_columns: List[str]
    provider: Optional[str]
    technical_details: str
    retryable: bool
    job_id: Optional[str]
    namespace: Optional[str]
    pod_name: Optional[str]
    container_name: Optional[str]
    kubernetes_reason: Optional[str]
    exit_code: Optional[int]
    logs_tail: Optional[str]

class TrainingResult(TypedDict):
    task_id: str
    mlflow_run_id: str
    dashboard_url: Optional[str]
    ray_job_id: NotRequired[str]
    ray_job_status: NotRequired[str]
    ray_namespace: NotRequired[str]
    user_id: str
    session_id: str
    model_type: str
    model_name: str
    model_description: str
    model_version: str
    runtime_s: float
    rows_processed: int
    num_epochs_trained: int
    final_train_loss: float
    final_val_loss: float
    final_accuracy: NotRequired[float]
    final_f1_score: NotRequired[float]
    final_train_mae: NotRequired[float]
    final_mae: NotRequired[float]
    final_train_rmse: NotRequired[float]
    final_rmse: NotRequired[float]
    final_train_r2_score: NotRequired[float]
    final_r2_score: NotRequired[float]
    num_features: int
    num_classes: int
    created_at: str
    status: NotRequired[str]
    error: NotRequired[str]
    failure: NotRequired[TrainingFailure]
    integrity_report: NotRequired[Dict[str, Any]]
    evaluation_report: NotRequired[Dict[str, Any]]

class ModelSummary(TypedDict):
    mlflow_run_id:     str
    user_id:           str
    session_id:        str
    model_name:        str
    model_type:        str
    model_description: str
    model_version:     str
    experiment_name:   str
    status:            str
    created_at:        str
    final_val_loss:    Optional[float]
    final_accuracy:    NotRequired[Optional[float]]
    final_f1_score:    NotRequired[Optional[float]]
    final_mae:         NotRequired[Optional[float]]
    final_rmse:        NotRequired[Optional[float]]
    final_r2_score:    NotRequired[Optional[float]]

class ModelDetail(TypedDict):
    mlflow_run_id:     str
    task_id:           str
    user_id:           str
    session_id:        str
    model_name:        str
    model_type:        str
    model_description: str
    model_version:     str
    experiment_name:   str
    experiment_id:     str
    status:            str
    created_at:        str

    input_size:    int
    output_size:   int
    hidden_sizes:  List[int]
    activation:    str
    dropout:       float
    batch_norm:    bool

    feature_names: List[str]
    class_names:   List[str]
    target_column: str
    num_features:  int
    num_classes:   int
    num_samples:   int

    hyperparameters: Dict[str, Any]

    metrics: Dict[str, float]

    model_info:       Dict[str, Any]
    training_history: List[Dict[str, Any]]
    best_epoch:       Dict[str, Any]

    artifact_uri:       str 
    pth_artifact_path:  str
    onnx_artifact_path: str

    runtime_s:          float
    rows_processed:     int
    num_epochs_trained: int

    inference_service_details: Optional[Dict[str, Any]]
