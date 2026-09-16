from typing_extensions import NotRequired, TypedDict
from typing import Literal, List, Dict, Optional, Any

class HyperparameterConfig(TypedDict):
    learning_rate: float                                = 0.001
    epochs: int                                         = 30
    batch_size: int                                     = 32
    optimizer_name: Literal["adam", "sgd", "rmsprop"]   = "adam"
    random_seed: int                                    = 42
    early_stopping_patience: int                        = 5
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
    time_column: NotRequired[str]

class TrainingPlan(TypedDict):
    model_type: Literal["classification", "regression"] = "classification"
    model_name: str
    model_description: str
    model_version: str

    hyperparameter_config: HyperparameterConfig
    ray_config: Optional[RayConfig] = None
    data_config: DataConfig

class TrainingResult(TypedDict):
    task_id: str
    mlflow_run_id: str
    dashboard_url: Optional[str]
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
    final_accuracy: float
    final_f1_score: float
    num_features: int
    num_classes: int
    created_at: str

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
