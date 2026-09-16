#!/usr/bin/env python3
"""
Integration test for direct input inference with real Kubernetes workloads

This test:
1. Creates a real Kubernetes job
2. Runs inference using the actual model
3. Waits for completion
4. Retrieves and verifies results

Requirements:
- Kubernetes cluster access
- GCS access for blob storage
- MLflow tracking server access
- Proper service account with GCS permissions
"""

import sys
import os
from pathlib import Path
import time

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment from: {env_path}")
else:
    # Try loading from current directory
    load_dotenv()
    print("✓ Loaded environment from .env (if exists)")

# Add project root to path (tests folder is one level down from project root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.mta.temporary_inference import (
    detect_inference_request,
    parse_direct_input,
    extract_csv_from_state,
    TemporaryInferenceManager
)
from app.graph.etl_state import ETLState
from langchain_core.messages import HumanMessage
import pandas as pd

# Configuration
MODEL_RUN_ID = "dcde705d65634bbca22e0753f2b48709"
MODEL_URI = f"runs:/{MODEL_RUN_ID}/model"
TEST_INPUT = "SepalLengthCm 5.1 SepalWidthCm 3.5 PetalLengthCm 1.4 PetalWidthCm 0.2"

# Get configuration from environment
GCS_BUCKET = os.getenv("GCS_INFERENCE_BUCKET") or os.getenv("GCS_BUCKET", "avaloka-test-user-filestore")
GCS_PREFIX = os.getenv("GCS_INFERENCE_PREFIX", "tmp-inference")
# Use "default" namespace to avoid service account issues
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "default")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_BACKEND_STORE_URI") or os.getenv("MLFLOW_TRACKING_URI")
# Don't set service account - let it use default
K8S_SERVICE_ACCOUNT = os.getenv("K8S_SERVICE_ACCOUNT") or os.getenv("INFERENCE_SERVICE_ACCOUNT") or None

def test_direct_input_parsing():
    """Test 1: Verify direct input parsing works"""
    print("=" * 60)
    print("Test 1: Direct Input Parsing")
    print("=" * 60)
    
    try:
        df = parse_direct_input(TEST_INPUT)
        
        if df is None or df.empty:
            print("  ✗ FAILED: Could not parse direct input")
            return False
        
        print(f"  ✓ Parsed successfully")
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
        print(f"    Values: {df.iloc[0].to_dict()}")
        
        assert len(df) == 1, "Should have 1 row"
        assert len(df.columns) == 4, "Should have 4 columns"
        assert "SepalLengthCm" in df.columns, "Should have SepalLengthCm"
        assert df.iloc[0]["SepalLengthCm"] == 5.1, "Should have correct value"
        
        print("  ✓ PASSED\n")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_detection():
    """Test 2: Verify detection works"""
    print("=" * 60)
    print("Test 2: Direct Input Detection")
    print("=" * 60)
    
    try:
        messages = [HumanMessage(content=f'Run inference on "{TEST_INPUT}"')]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": MODEL_RUN_ID
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        
        print(f"  Detection: is_request={is_request}, path={path}, mode={inference_mode}")
        
        if not is_request or path != "direct_input":
            print("  ✗ FAILED: Detection failed")
            return False
        
        print("  ✓ PASSED\n")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_workflow():
    """Test 3: Full inference workflow with real K8s job"""
    print("=" * 60)
    print("Test 3: Full Inference Workflow (Real K8s Job)")
    print("=" * 60)
    
    # Check required environment variables
    if not MLFLOW_TRACKING_URI:
        print("  ⚠ SKIPPED: MLFLOW_TRACKING_URI not set")
        print("    Set MLFLOW_BACKEND_STORE_URI or MLFLOW_TRACKING_URI to run this test")
        return None
    
    # Check if we have blob_store (required for staging)
    try:
        from app.core.storage import GCSBlobStore
        # Create blob store directly
        blob_store = GCSBlobStore(bucket=GCS_BUCKET, prefix=GCS_PREFIX)
        print("  ✓ GCSBlobStore created")
    except ImportError:
        try:
            from app.core.storage import get_blob_store
            blob_store = get_blob_store()
            if blob_store is None:
                print("  ⚠ SKIPPED: blob_store not configured")
                print("    Configure GCS or S3 storage to run this test")
                return None
        except Exception as e:
            print(f"  ⚠ SKIPPED: Could not get blob_store: {e}")
            return None
    except Exception as e:
        print(f"  ⚠ SKIPPED: Could not create blob_store: {e}")
        print(f"    Error: {str(e)}")
        return None
    
    print(f"  Configuration:")
    print(f"    Model URI: {MODEL_URI}")
    print(f"    MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"    GCS Bucket: {GCS_BUCKET}")
    print(f"    GCS Prefix: {GCS_PREFIX}")
    print(f"    K8s Namespace: {K8S_NAMESPACE}")
    print(f"    Input: {TEST_INPUT}")
    print()
    
    # Create manager (keep jobs for debugging)
    try:
        manager = TemporaryInferenceManager(
            blob_store=blob_store,
            k8s_namespace=K8S_NAMESPACE,
            gcs_bucket=GCS_BUCKET,
            gcs_prefix=GCS_PREFIX,
            keep_jobs=True  # Keep jobs for inspection
        )
        print("  ✓ TemporaryInferenceManager created (jobs will be kept for debugging)")
    except Exception as e:
        print(f"  ✗ FAILED: Could not create manager: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Set up state and messages
    messages = [HumanMessage(content=f'Run inference on "{TEST_INPUT}"')]
    state = ETLState({
        "training_completed": True,
        "mlflow_run_id": MODEL_RUN_ID
    })
    
    print("  Starting inference workflow...")
    print("  (This will create a real K8s job and may take several minutes)")
    print()
    
    start_time = time.time()
    
    try:
        # Run inference
        result = manager.handle_inference_request(state, messages)
        
        elapsed = time.time() - start_time
        
        print(f"  Workflow completed in {elapsed:.1f} seconds")
        print()
        
        if not result.get("success"):
            print(f"  ✗ FAILED: {result.get('error')}")
            if result.get("message"):
                print(f"    Message: {result.get('message')[:200]}...")
            return False
        
        # Verify results
        predictions_df = result.get("predictions_df")
        if predictions_df is None or predictions_df.empty:
            print("  ✗ FAILED: No predictions returned")
            return False
        
        print(f"  ✓ Inference completed successfully!")
        print(f"    Job Name: {result.get('job_name')}")
        print(f"    Output GCS Path: {result.get('output_gcs_path')}")
        print(f"    Predictions DataFrame shape: {predictions_df.shape}")
        print(f"    Columns: {list(predictions_df.columns)}")
        
        # Check for prediction columns
        if "Predicted" in predictions_df.columns:
            print(f"    Prediction: {predictions_df.iloc[0]['Predicted']}")
        if "Confidence" in predictions_df.columns:
            print(f"    Confidence: {predictions_df.iloc[0]['Confidence']:.3f}")
        
        print()
        print("  Predictions DataFrame:")
        print(predictions_df.to_string())
        print()
        
        # Verify prediction columns exist
        assert "Predicted" in predictions_df.columns, "Should have Predicted column"
        assert "Confidence" in predictions_df.columns, "Should have Confidence column"
        assert len(predictions_df) == 1, "Should have 1 prediction"
        
        print("  ✓ PASSED\n")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ FAILED after {elapsed:.1f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Direct Input Inference Integration Test")
    print("=" * 60)
    print()
    print("This test will:")
    print("  1. Parse direct input")
    print("  2. Detect inference request")
    print("  3. Create real Kubernetes job")
    print("  4. Run inference with actual model")
    print("  5. Retrieve and verify results")
    print()
    print("Required environment variables:")
    print("  - MLFLOW_BACKEND_STORE_URI or MLFLOW_TRACKING_URI")
    print("  - GCS_INFERENCE_BUCKET or GCS_BUCKET")
    print("  - K8S_NAMESPACE (optional, defaults to 'default')")
    print()
    
    results = []
    
    # Test 1: Parsing
    results.append(("Direct input parsing", test_direct_input_parsing()))
    
    # Test 2: Detection
    results.append(("Direct input detection", test_detection()))
    
    # Test 3: Full workflow (may be skipped if env not configured)
    workflow_result = test_inference_workflow()
    if workflow_result is not None:
        results.append(("Full inference workflow", workflow_result))
    else:
        results.append(("Full inference workflow", "SKIPPED"))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result == "SKIPPED")
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⚠ SKIPPED"
        print(f"  {test_name}: {status}")
    
    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped out of {total} tests")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All executed tests passed!")
        if skipped > 0:
            print(f"⚠️  {skipped} test(s) were skipped (check environment configuration)")
        return 0
    elif failed > 0:
        print(f"\n⚠️  {failed} test(s) failed. Please review the output above.")
        return 1
    else:
        print("\n⚠️  No tests were executed. Please check your environment configuration.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
