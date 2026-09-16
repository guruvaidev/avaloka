"""
ONNX Export Module for Model Training Agent

Provides model export functionality to ONNX format for cross-framework compatibility.
Supports PyTorch models with validation and optimization.
"""

import logging
import warnings
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Handles model export to ONNX format"""
    
    def __init__(self):
        """Initialize ONNX exporter"""
        self.onnx_model = None
        self.validation_results = {}
    
    def export_pytorch_model(self, 
                           model: torch.nn.Module, 
                           input_shape: Tuple[int, ...],
                           export_path: str,
                           model_name: str = "model",
                           opset_version: int = 18,
                           dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export PyTorch model to ONNX format (single-file, opset 18)
        
        Args:
            model: Trained PyTorch model
            input_shape: Shape of input tensor (excluding batch dimension)
            export_path: Path to save ONNX model
            model_name: Name of the model
            opset_version: ONNX opset version (default: 18)
            dynamic_axes: Dynamic axes configuration for variable batch size (e.g., {"input": {0: "batch"}, "logits": {0: "batch"}})
            metadata: Additional metadata (x_mean, x_std, class_names, etc.)
            
        Returns:
            Export result dictionary
        """
        try:
            # Create export directory
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy input
            batch_size = 1
            device = next(model.parameters()).device
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            # ONNX export path
            onnx_path = export_dir / f"{model_name}.onnx"
            
            # Define input and output names
            input_names = ["input"]
            output_names = ["logits"]  # Match reference code
            
            # Export to ONNX with opset 18 (matches reference code)
            # Suppress UserWarning about dynamic_axes when dynamo=True
            # We're using the traditional export path, so dynamic_axes is correct
            with torch.no_grad():
                with warnings.catch_warnings():
                    # Suppress warning about dynamic_axes with dynamo=True
                    # The actual warning message: "'dynamic_axes' is not recommended when dynamo=True..."
                    # We're using the traditional export path, so this warning is not applicable
                    warnings.filterwarnings("ignore", message=".*dynamic_axes.*dynamo.*", category=UserWarning)
                    warnings.filterwarnings("ignore", message=".*'dynamic_axes'.*not recommended.*dynamo.*", category=UserWarning)
                    warnings.filterwarnings("ignore", message=".*dynamic_axes.*dynamo.*", category=FutureWarning)
                    torch.onnx.export(
                        model,
                        dummy_input,
                        str(onnx_path),
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,  # Dynamic axes for variable batch size
                        verbose=False
                    )
            
            # Rewrite ONNX as single file (remove external data if any)
            # This matches the reference code pattern
            try:
                onnx_model = onnx.load(str(onnx_path), load_external_data=True)
                # Save again but force NO external data (single file)
                onnx.save_model(
                    onnx_model,
                    str(onnx_path),
                    save_as_external_data=False
                )
                # Optionally delete sidecar file if it exists
                sidecar = str(onnx_path) + ".data"
                if os.path.exists(sidecar):
                    os.remove(sidecar)
                logger.info(f"Exported SINGLE-FILE ONNX model to {onnx_path}")
            except Exception as e:
                logger.warning(f"Could not rewrite ONNX as single file: {e}. Continuing with existing file.")
            
            # Validate exported model
            validation_result = self.validate_onnx_model(str(onnx_path), model, dummy_input)
            
            # Create metadata (merge with provided metadata)
            onnx_metadata = self._create_onnx_metadata(
                model_name, input_shape, opset_version, 
                dynamic_axes, validation_result
            )
            
            # Merge with provided metadata (x_mean, x_std, class_names, etc.)
            if metadata:
                onnx_metadata.update(metadata)
            
            # Save metadata
            metadata_path = export_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(onnx_metadata, f, indent=2)
            
            export_result = {
                "success": True,
                "onnx_path": str(onnx_path),
                "metadata_path": str(metadata_path),
                "model_size_mb": self._get_file_size_mb(onnx_path),
                "validation": validation_result,
                "input_shape": input_shape,
                "opset_version": opset_version
            }
            
            logger.info(f"Successfully exported model to ONNX: {onnx_path}")
            return export_result
            
        except Exception as e:
            logger.error(f"Error exporting model to ONNX: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_onnx_model(self, 
                           onnx_path: str, 
                           original_model: torch.nn.Module, 
                           test_input: torch.Tensor,
                           tolerance: float = 1e-5) -> Dict[str, Any]:
        """
        Validate ONNX model against original PyTorch model
        
        Args:
            onnx_path: Path to ONNX model
            original_model: Original PyTorch model
            test_input: Test input tensor
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Validation results dictionary
        """
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get original PyTorch output
            original_model.eval()
            with torch.no_grad():
                pytorch_output = original_model(test_input).numpy()
            
            # Get ONNX output
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            # Compare outputs
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            is_close = np.allclose(pytorch_output, onnx_output, atol=tolerance, rtol=tolerance)
            
            # Get model info
            input_info = self._get_onnx_input_info(ort_session)
            output_info = self._get_onnx_output_info(ort_session)
            
            validation_result = {
                "valid": True,
                "onnx_loadable": True,
                "outputs_match": is_close,
                "max_difference": float(max_diff),
                "mean_difference": float(mean_diff),
                "tolerance": tolerance,
                "input_info": input_info,
                "output_info": output_info,
                "test_successful": True
            }
            
            if not is_close:
                logger.warning(f"ONNX validation: outputs don't match within tolerance. Max diff: {max_diff}")
            else:
                logger.info("ONNX validation: outputs match within tolerance")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating ONNX model: {e}")
            return {
                "valid": False,
                "error": str(e),
                "test_successful": False
            }
    
    def optimize_onnx_model(self, onnx_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize ONNX model for inference
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save optimized model (overwrites original if None)
            
        Returns:
            Optimization results
        """
        try:
            # Load model
            model = onnx.load(onnx_path)
            
            # Basic optimizations
            from onnx import optimizer
            
            # Define optimization passes
            passes = [
                'eliminate_identity',
                'eliminate_nop_dropout', 
                'eliminate_nop_transpose',
                'fuse_consecutive_transposes',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm'
            ]
            
            # Apply optimizations
            optimized_model = optimizer.optimize(model, passes)
            
            # Save optimized model
            output_file = output_path if output_path else onnx_path
            onnx.save(optimized_model, output_file)
            
            # Get size comparison
            original_size = self._get_file_size_mb(onnx_path)
            optimized_size = self._get_file_size_mb(output_file)
            size_reduction = ((original_size - optimized_size) / original_size) * 100
            
            result = {
                "success": True,
                "optimized_path": output_file,
                "original_size_mb": original_size,
                "optimized_size_mb": optimized_size,
                "size_reduction_percent": size_reduction,
                "optimization_passes": passes
            }
            
            logger.info(f"ONNX optimization completed. Size reduction: {size_reduction:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_inference_session(self, onnx_path: str, providers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create ONNX Runtime inference session
        
        Args:
            onnx_path: Path to ONNX model
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            
        Returns:
            Session information and session object
        """
        try:
            # Default providers
            if providers is None:
                providers = ['CPUExecutionProvider']
                if ort.get_device() == 'GPU':
                    providers.insert(0, 'CUDAExecutionProvider')
            
            # Create session
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Get session info
            input_info = self._get_onnx_input_info(session)
            output_info = self._get_onnx_output_info(session)
            
            session_info = {
                "success": True,
                "session": session,
                "providers": session.get_providers(),
                "input_info": input_info,
                "output_info": output_info,
                "model_path": onnx_path
            }
            
            logger.info(f"Created ONNX inference session with providers: {session.get_providers()}")
            return session_info
            
        except Exception as e:
            logger.error(f"Error creating ONNX inference session: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_inference(self, 
                     session: ort.InferenceSession, 
                     input_data: np.ndarray,
                     input_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run inference on ONNX model
        
        Args:
            session: ONNX Runtime inference session
            input_data: Input data as numpy array
            input_name: Name of input (uses first input if None)
            
        Returns:
            Inference results
        """
        try:
            # Get input name if not provided
            if input_name is None:
                input_name = session.get_inputs()[0].name
            
            # Prepare input
            ort_inputs = {input_name: input_data}
            
            # Run inference
            outputs = session.run(None, ort_inputs)
            
            # Get output names
            output_names = [output.name for output in session.get_outputs()]
            
            result = {
                "success": True,
                "outputs": outputs,
                "output_names": output_names,
                "input_shape": input_data.shape,
                "output_shapes": [output.shape for output in outputs]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error running ONNX inference: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_onnx_metadata(self, 
                             model_name: str, 
                             input_shape: Tuple[int, ...], 
                             opset_version: int,
                             dynamic_axes: Dict[str, Dict[int, str]], 
                             validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for ONNX export"""
        return {
            "model_name": model_name,
            "export_timestamp": datetime.now().isoformat(),
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "opset_version": opset_version,
            "input_shape": input_shape,
            "dynamic_axes": dynamic_axes,
            "validation": validation_result,
            "framework": "pytorch",
            "exporter_version": "1.0.0"
        }
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        size_bytes = Path(file_path).stat().st_size
        return size_bytes / (1024 * 1024)
    
    def _get_onnx_input_info(self, session: ort.InferenceSession) -> List[Dict[str, Any]]:
        """Get input information from ONNX session"""
        input_info = []
        for input_tensor in session.get_inputs():
            info = {
                "name": input_tensor.name,
                "type": input_tensor.type,
                "shape": input_tensor.shape
            }
            input_info.append(info)
        return input_info
    
    def _get_onnx_output_info(self, session: ort.InferenceSession) -> List[Dict[str, Any]]:
        """Get output information from ONNX session"""
        output_info = []
        for output_tensor in session.get_outputs():
            info = {
                "name": output_tensor.name,
                "type": output_tensor.type,
                "shape": output_tensor.shape
            }
            output_info.append(info)
        return output_info


# Factory function
def create_onnx_exporter() -> ONNXExporter:
    """Create ONNX exporter instance"""
    return ONNXExporter()


# Utility functions
def check_onnx_runtime_providers() -> List[str]:
    """Get available ONNX Runtime execution providers"""
    return ort.get_available_providers()


def get_onnx_model_info(onnx_path: str) -> Dict[str, Any]:
    """Get information about an ONNX model"""
    try:
        model = onnx.load(onnx_path)
        
        # Get model info
        info = {
            "model_version": model.model_version,
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "domain": model.domain,
            "doc_string": model.doc_string,
            "graph_name": model.graph.name,
            "num_nodes": len(model.graph.node),
            "opset_version": None
        }
        
        # Get opset version
        if model.opset_import:
            info["opset_version"] = model.opset_import[0].version
        
        # Get input/output info
        inputs = []
        for input_tensor in model.graph.input:
            inputs.append({
                "name": input_tensor.name,
                "type": str(input_tensor.type)
            })
        
        outputs = []
        for output_tensor in model.graph.output:
            outputs.append({
                "name": output_tensor.name,
                "type": str(output_tensor.type)
            })
        
        info["inputs"] = inputs
        info["outputs"] = outputs
        
        return info
        
    except Exception as e:
        return {"error": str(e)}
