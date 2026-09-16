"""
model.py — DynamicMLP Architecture
====================================
Standalone model definition file.

Imported by:
  - ray_job.py       (training)
  - mlflow_manager.py (loading weights into arch for inference)
  - inference.py     (prediction)

Design notes:
  - No training logic here — pure architecture + serialization helpers.
  - ModelConfig is a plain dataclass so it can be JSON-serialised and stored
    alongside model artifacts in MLflow metadata.
  - from_config() / to_config() round-trip cleanly through JSON.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Optional

import torch
import torch.nn as nn


# =========================================================
# Model Config (serialisable)
# =========================================================


@dataclass
class ModelConfig:
    """
    All information needed to reconstruct the DynamicMLP architecture.
    Stored as metadata/model_config.json in every MLflow run.
    """
    input_size:   int
    output_size:  int
    hidden_sizes: List[int]
    activation:   str   = "relu"
    dropout:      float = 0.2
    batch_norm:   bool  = True

    # Data / label metadata — stored for inference convenience
    feature_names:  List[str] = field(default_factory=list)
    class_names:    List[str] = field(default_factory=list)
    target_column:  str       = ""
    model_type:     str       = "classification"   # "classification" | "regression"
    preprocessing:  Dict      = field(default_factory=dict)

    # Identity
    model_name:        str = ""
    model_description: str = ""
    model_version:     str = ""
    task_id:           str = ""
    mlflow_run_id:     str = ""

    # Inference Service
    inference_service_details: Optional[Dict] = field(default=None)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        # Accept extra keys gracefully (forward-compat)
        valid = {k: d.get(k, None) for k, _ in cls.__dataclass_fields__.items()}
        return cls(**valid)

    @classmethod
    def from_json(cls, s: str) -> "ModelConfig":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_json_file(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =========================================================
# DynamicMLP
# =========================================================


class DynamicMLP(nn.Module):
    """
    Dynamic Multi-Layer Perceptron for tabular classification / regression.

    Architecture is fully determined by ModelConfig so weights and arch
    can always be reconstructed from the config alone.

    Layers per hidden unit:
      Linear → [BatchNorm1d] → Activation → [Dropout]
    Final layer: Linear(last_hidden → output_size), no activation.
    """

    ACTIVATIONS = {
        "relu":       nn.ReLU,
        "tanh":       nn.Tanh,
        "sigmoid":    nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "elu":        nn.ELU,
    }

    def __init__(
        self,
        input_size:   int,
        output_size:  int,
        hidden_sizes: List[int],
        activation:   str   = "relu",
        dropout:      float = 0.2,
        batch_norm:   bool  = True,
    ) -> None:
        super().__init__()
        self.input_size   = input_size
        self.output_size  = output_size
        self.hidden_sizes = list(hidden_sizes)
        self.activation   = activation
        self.dropout      = dropout
        self.batch_norm   = batch_norm

        layers: List[nn.Module] = []
        prev = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            act_cls = self.ACTIVATIONS.get(activation.lower(), nn.ReLU)
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    # ---- Convenience constructors ----

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "DynamicMLP":
        return cls(
            input_size=cfg.input_size,
            output_size=cfg.output_size,
            hidden_sizes=cfg.hidden_sizes,
            activation=cfg.activation,
            dropout=cfg.dropout,
            batch_norm=cfg.batch_norm,
        )

    def load_weights(self, path: str, device: str = "cpu") -> None:
        """Load a .pth / .pt state dict, stripping DDP 'module.' prefixes."""
        state = torch.load(path, map_location=device)
        state = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state.items()
        }
        self.load_state_dict(state)

    # ---- Export helpers ----

    def save_pth(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def save_onnx(self, path: str, device: str = "cpu") -> None:
        self.eval()
        dummy = torch.zeros(1, self.input_size, device=device)
        torch.onnx.export(
            self,
            dummy,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
            external_data=False,
        )

    def __repr__(self) -> str:
        return (
            f"DynamicMLP(in={self.input_size}, hidden={self.hidden_sizes}, "
            f"out={self.output_size}, act={self.activation}, "
            f"dropout={self.dropout}, bn={self.batch_norm})"
        )
