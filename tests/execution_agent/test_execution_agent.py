""" Unit tests for the execution agent """

from tests._quarantine import requires_api

requires_api("app.agents.execution_agent", "prepare_infrastructure", "execute_code_on_k8s", replacement="prepare_infrastructure was removed; infrastructure is provisioned via the provision_infra graph node now. Repair against the current node API or remove this file.")


import os
import sys
import json
import pytest
import pandas as pd
from unittest import mock
from pathlib import Path
from datetime import datetime
import base64

# Add parent directory to path to import execution_agent and app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# agent APIs
from app.agents.execution_agent import (
    execution_agent_node,
    prepare_infrastructure,
    execute_code_on_k8s,
)

# import execution helper
from app.execution_helper import (
    process_execution_results,
    save_output_to_file,
)
from app.graph.etl_state import ETLState

@pytest.fixture
def sample_code():
    """Sample Python code from the coding agent""" 
    return """
import pandas as pd

# Read input data
df = pd.read_csv('input.csv')

# Perform a simple calculation
result = df.groupby('Species').mean().reset_index()

# Write output
result.to_csv('output.csv', index=False)
print("Processing complete!")
"""

@pytest.fixture
def mock_infra_success_response():
    """Mock successful infrastructure response"""
    return ETLState(
        messages=[],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=True,
        coder_definition={},
        infrastructure_request={},
        infrastructure_provisioned={
            "type": "gcp",
            "app_type": "python-docker",
            "status": "provisioned",
            "details": [
                {
                    "step_name": "Get infra-agent-service IP/Port",
                    "status": "SUCCESS",
                    "message": "Service infra-agent-service accessible at 34.123.456.789:8080",
                    "details": {
                        "ip": "34.123.456.789",
                        "port": "8080"
                    }
                }
            ]
        }
    )

@pytest.fixture
def mock_execution_success_response():
    """Mock successful execution response"""
    iris_sample = pd.DataFrame({
        'Species': ['setosa', 'versicolor', 'virginica'],
        'sepal_length': [5.1, 5.9, 6.3],
        'sepal_width': [3.5, 2.7, 2.9],
        'petal_length': [1.4, 4.2, 5.6],
        'petal_width': [0.2, 1.3, 1.8]
    })
    csv_data = iris_sample.to_csv(index=False)
    
    return {
        "status": "success",
        "message": "Code executed successfully",
        "execution_time": 1.23,
        "stdout": "Processing complete!",
        "stderr": "",
        "output": f"data:text/csv;base64,{base64.b64encode(csv_data.encode()).decode()}"
    }

def test_prepare_infrastructure_success(mock_infra_success_response):
    """Test prepare_infrastructure with successful provisioning"""
    with mock.patch('execution_agent.infra_agent_node', return_value=mock_infra_success_response):
        state = ETLState(
            messages=[],
            planner_definition={},
            ready_to_summarize=False,
            ready_to_code=True,
            coder_definition={},
            infrastructure_request={},
            infrastructure_provisioned=None,
            infrastructure_preference="gcp"
        )
        
        result = prepare_infrastructure(state)
        
        assert result["status"] == "success"
        assert result["service_ip"] == "34.123.456.789"
        assert result["service_port"] == "8080"

def test_execute_code_on_k8s():
    """Test execute_code_on_k8s function"""
    with mock.patch('requests.post') as mock_post:
        # Setup mock responses
        mock_post.return_value = mock.MagicMock(
            status_code=200,
            json=lambda: {"status": "success", "message": "Execution completed"}
        )
        
        # Test execution
        result = execute_code_on_k8s(
            code="print('hello')",
            service_ip="192.168.1.1",
            service_port="8080",
            data_source_location="/data/input.csv"
        )
        
        # Verify result
        assert result["status"] == "success"
        assert "Execution completed" in result["message"]
        assert "execution_time" in result
        
        # Verify API call
        mock_post.assert_called_once()

def test_save_output_to_file():
    """Test save_output_to_file function"""
    # Test with DataFrame
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    result = save_output_to_file(df)
    
    assert result["status"] == "success"
    assert "output_file" in result
    assert os.path.exists(result["output_file"])
    
    # Clean up test file
    os.remove(result["output_file"])

def test_process_execution_results(mock_execution_success_response):
    """Test process_execution_results function"""
    with mock.patch('execution_helper.save_output_to_file', return_value={
        "status": "success",
        "output_file": "/path/to/output.csv"
    }):
        result = process_execution_results(mock_execution_success_response)
        
        assert result["status"] == "success"
        assert "output" in result
        assert result["output"]["type"] == "dataframe"
        assert "output_file" in result

@mock.patch('execution_agent.prepare_infrastructure')
@mock.patch('execution_agent.execute_code_on_k8s')
@mock.patch('execution_helper.process_execution_results')

def test_execution_agent_node_success(

    mock_process_results, 
    mock_execute, 
    mock_prepare_infra
):
    """Test execution_agent_node with successful execution"""
    # Setup mocks
    mock_prepare_infra.return_value = {
        "status": "success",
        "service_ip": "192.168.1.1",
        "service_port": "8080",
        "updated_state": ETLState(
            messages=[],
            planner_definition={},
            ready_to_summarize=False,
            ready_to_code=True,
            coder_definition={},
            infrastructure_request={},
            infrastructure_provisioned={"status": "provisioned"}
        )
    }
    
    mock_execute.return_value = {
        "status": "success",
        "stdout": "Processing complete!",
        "execution_time": 1.5
    }
    
    mock_process_results.return_value = {
        "status": "success",
        "execution_time": 1.5,
        "output": {
            "type": "dataframe",
            "rows": 3,
            "columns": ["Species", "sepal_length_mean", "sepal_width_mean"]
        },
        "output_file": {
            "status": "success",
            "output_file": "/path/to/output.csv"
        }
    }
    
    # Create test state
    state = ETLState(
        messages=[],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=True,
        coder_definition={
            "code": "print('Test code')"
        },
        infrastructure_request={},
        infrastructure_provisioned=None,
        data_source_location="data/iris.csv",
        sample_data="data/iris.csv"
    )
    
    # Run execution agent
    result_state = execution_agent_node(state)
    
    # Verify result state
    assert "execution_result" in result_state
    assert result_state["execution_result"]["status"] == "success"
    assert len(result_state["messages"]) == 1  # Should have added result message

@mock.patch('execution_agent.prepare_infrastructure')
def test_execution_agent_node_infra_failure(mock_prepare_infra):
    """Test execution_agent_node with infrastructure failure"""
    # Setup mock to fail
    mock_prepare_infra.return_value = {
        "status": "failed",
        "message": "Infrastructure provisioning failed: GCP_PROJECT_ID not set",
        "details": {}
    }
    
    # Create test state
    state = ETLState(
        messages=[],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=True,
        coder_definition={
            "code": "print('Test code')"
        },
        infrastructure_request={},
        infrastructure_provisioned=None
    )
    
    # Run execution agent
    result_state = execution_agent_node(state)
    
    # Verify result state
    assert "execution_result" in result_state
    assert result_state["execution_result"]["status"] == "failed"
    assert "Infrastructure provisioning failed" in result_state["execution_result"]["message"]
    assert len(result_state["messages"]) == 1  # Should have added error message

def test_execution_agent_node_no_code():
    """Test execution_agent_node with no code provided"""
    # Create test state with no code
    state = ETLState(
        messages=[],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=True,
        coder_definition={},  # Empty code definition
        infrastructure_request={},
        infrastructure_provisioned=None
    )
    
    # Run execution agent
    result_state = execution_agent_node(state)
    
    # Verify result state
    assert "execution_result" in result_state
    assert result_state["execution_result"]["status"] == "failed"
    assert "No code provided" in result_state["execution_result"]["message"]
    assert len(result_state["messages"]) == 1  # Should have added error message 