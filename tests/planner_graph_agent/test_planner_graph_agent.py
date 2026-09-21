"""
Test script for the planner graph agent integration
Run this to test the planner graph functionality before integrating with the full workflow
"""

import os
import sys
import tempfile
import pandas as pd
from pathlib import Path
import logging
import pytest

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app.agents.planner_graph_agent import planner_graph_agent_node, PlannerGraphAgent, _convert_plan_to_graph_format
    from app.graph.etl_state import ETLState
    logger.info("✓ Successfully imported planner graph agent modules")
except ImportError as exc:  # pragma: no cover - environment-dependent
    # `sys.exit(1)` here used to abort the ENTIRE pytest session, not this
    # module: pytest reports it as INTERNALERROR and no test in the repository
    # runs, whatever was wrong. This file began life as a standalone script and
    # kept the script's error handling when it became a test.
    pytest.skip(
        f"planner graph agent dependencies unavailable: {exc}",
        allow_module_level=True,
    )


def test_planner_graph_agent_standalone():
    """Test the planner graph agent in isolation"""
    print("\n=== Testing Planner Graph Agent (Standalone) ===")
    
    # Create test plan data
    sample_plan = {
        "plan_id": "test_sales_analysis",
        "title": "Test Sales Analysis Workflow",
        "steps": [
            {
                "step_id": 1,
                "name": "Load Sales Data",
                "type": "data_input",
                "description": "Load sales data from CSV file",
                "inputs": ["sales.csv"],
                "outputs": ["sales_df"],
                "sql_query": None,
                "parameters": None
            },
            {
                "step_id": 2,
                "name": "Data Cleaning",
                "type": "cleaning",
                "description": "Remove duplicates and handle missing values",
                "inputs": ["sales_df"],
                "outputs": ["clean_sales_df"],
                "sql_query": None,
                "parameters": None
            },
            {
                "step_id": 3,
                "name": "Group by Region",
                "type": "aggregation",
                "description": "Group data by region and calculate totals",
                "inputs": ["clean_sales_df"],
                "outputs": ["regional_summary"],
                "sql_query": "SELECT region, SUM(amount) as total_sales FROM sales GROUP BY region",
                "parameters": None
            },
            {
                "step_id": 4,
                "name": "Export Results",
                "type": "data_output",
                "description": "Save results to CSV file",
                "inputs": ["regional_summary"],
                "outputs": ["results.csv"],
                "sql_query": None,
                "parameters": None
            }
        ],
        "connections": [
            {"from_step": 1, "to_step": 2, "edge_type": "data_flow"},
            {"from_step": 2, "to_step": 3, "edge_type": "data_flow"},
            {"from_step": 3, "to_step": 4, "edge_type": "data_flow"}
        ]
    }
    
    # Test the agent directly
    graph_agent = PlannerGraphAgent()
    result = graph_agent.generate_planner_graph(sample_plan)
    
    print(f"Status: {result.get('status')}")
    print(f"Planner Graph Path: {result.get('planner_graph_path')}")
    print(f"Plan ID: {result.get('plan_id')}")
    
    # Check that base64 is no longer returned
    assert 'base64_image' not in result, "Base64 image should not be in result"
    print("✓ Base64 conversion correctly removed")
    
    # Check status format and verify file creation on success
    status = result.get('status', '')
    assert status == 'success', f"Expected status 'success', but got '{status}'"
    
    graph_path = result.get('planner_graph_path', '')
    assert graph_path and Path(graph_path).exists(), "Planner graph file was not found"
    print("✓ Planner graph file successfully created")
    print("✓ Standalone planner graph test passed")


def test_planner_graph_node():
    """Test the planner graph agent as a LangGraph node"""
    print("\n=== Testing Planner Graph Agent (LangGraph Node) ===")
    
    # Create mock ETL state with updated structure
    mock_state: ETLState = {
        "messages": [],
        "planner_definition": {
            "job_name": "Test ETL Job",
            "source": "employees.csv",
            "transformations": [
                {
                    "name": "Filter Active Employees",
                    "type": "filter",
                    "columns": ["status"],
                },
                {
                    "name": "Group by Department",
                    "type": "aggregation",
                    "columns": ["department"],
                    "aggregation": "count"
                }
            ],
            "destination": "department_counts.csv",
            "schedule": "daily",
            "notes": "Count active employees by department"
        },
        "ready_to_summarize": True,
        "ready_to_code": False,
        "coder_definition": {},
        "data_source_location": "",
        "output_location": "",
        "infrastructure_request": {},
        "infrastructure_provisioned": {},
        "execution_result": {},
        "output_file_data": {},
        "planner_graph_path": "",
        "planner_graph_status": ""
    }
    
    # Test the node function
    updated_state = planner_graph_agent_node(mock_state)
    
    print(f"Planner Graph Status: {updated_state.get('planner_graph_status')}")
    print(f"Planner Graph Path: {updated_state.get('planner_graph_path')}")
    
    # Check that base64 field is not in updated state
    assert 'planner_graph_base64' not in updated_state or not updated_state.get('planner_graph_base64'), \
        "Base64 field should be empty or not present"
    print("✓ Base64 field correctly removed from state")
    
    # Check that error field is not separate from status
    assert 'planner_graph_error' not in updated_state or not updated_state.get('planner_graph_error'), \
        "Separate error field should not be used"
    print("✓ Error handling correctly consolidated into status")
    
    # Validate status format
    status = updated_state.get('planner_graph_status', '')
    assert status == 'success', f"Node function test failed with status: {status}"
    print("✓ Planner graph completed successfully")
    print("✓ Node function test passed")


def test_error_handling():
    """Test error handling scenarios"""
    print("\n=== Testing Error Handling ===")
    
    graph_agent = PlannerGraphAgent()
    
    # Test 1: Invalid plan data (missing required fields)
    invalid_plan = {"plan_id": "test"}  # Missing steps
    result = graph_agent.generate_planner_graph(invalid_plan)
    status = result.get('status', '')
    assert status.startswith('error:'), f"Expected error status, got: {status}"
    assert 'steps' in status.lower(), "Error should mention missing steps"
    print("✓ Missing steps validation works correctly")

    # Test 2: Invalid step structure
    invalid_plan_2 = {
        "plan_id": "test_invalid_step",
        "title": "Test Invalid Step",
        "steps": [
            {"step_id": 1, "name": "Test"}  # Missing type
        ]
    }
    result = graph_agent.generate_planner_graph(invalid_plan_2)
    status = result.get('status', '')
    assert status.startswith('error:'), f"Expected error status, got: {status}"
    print("✓ Invalid step structure validation works correctly")

    # Test 3: Empty state handling
    empty_state = {}
    updated_state = planner_graph_agent_node(empty_state)
    status = updated_state.get('planner_graph_status', '')
    assert status.startswith('skipped:'), f"Expected skipped status, got: {status}"
    print("✓ Empty state handling works correctly")
    print("✓ Error handling tests passed")


def test_with_sample_data():
    """Test with actual sample CSV data"""
    print("\n=== Testing with Sample CSV Data ===")
    
    # Create sample data
    sample_data = {
        'employee_id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
        'department': ['Sales', 'Engineering', 'Sales', 'HR', 'Engineering'],
        'salary': [50000, 75000, 55000, 60000, 80000],
        'status': ['Active', 'Active', 'Inactive', 'Active', 'Active']
    }
    
    # Create temporary CSV file
    df = pd.DataFrame(sample_data)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_csv_path = f.name
    
    try:
        # Create more realistic ETL state with updated structure
        realistic_state: ETLState = {
            "messages": [],
            "planner_definition": {
                "job_name": "Employee Analysis",
                "source": temp_csv_path,
                "transformations": [
                    {
                        "name": "Filter Active Employees",
                        "type": "filter",
                        "columns": ["status"],
                    },
                    {
                        "name": "Calculate Average Salary by Department",
                        "type": "aggregation",
                        "columns": ["department", "salary"],
                        "aggregation": "avg"
                    }
                ],
                "destination": "avg_salary_by_dept.csv",
                "schedule": "weekly",
                "notes": "Calculate average salary for active employees by department"
            },
            "ready_to_summarize": True,
            "ready_to_code": False,
            "coder_definition": {},
            "data_source_location": temp_csv_path,
            "output_location": temp_csv_path.replace(".csv", "_output.csv"),
            "infrastructure_request": {},
            "infrastructure_provisioned": {},
            "execution_result": {},
            "output_file_data": {},
            "planner_graph_path": "",
            "planner_graph_status": ""
        }
        
        # Test the planner graph
        updated_state = planner_graph_agent_node(realistic_state)
        
        print(f"Data Source: {Path(temp_csv_path).name}")
        print(f"Sample Data Shape: {df.shape}")
        print(f"Planner Graph Status: {updated_state.get('planner_graph_status')}")
        print(f"Planner Graph Path: {updated_state.get('planner_graph_path')}")
        
        # Verify no base64 data is present
        assert not updated_state.get('planner_graph_base64'), "Base64 data should not be present"
        
        status = updated_state.get('planner_graph_status', '')
        assert status == 'success' or status.startswith('skipped:'), f"Sample data test failed with status: {status}"
        print("✓ Sample data test passed")
        
    finally:
        # Cleanup
        try:
            os.unlink(temp_csv_path)
        except:
            pass

def test_workflow_integration():
    """Test integration with the complete workflow"""
    print("\n=== Testing Workflow Integration ===")
    
    # Test that the node function properly handles state transitions
    test_state: ETLState = {
        "messages": [],
        "planner_definition": {
            "job_name": "Integration Test",
            "source": "test_data.csv",
            "transformations": [
                {
                    "name": "Data Transformation",
                    "type": "transformation",
                    "columns": ["col1", "col2"]
                }
            ],
            "destination": "output.csv"
        },
        "ready_to_summarize": True,
        "ready_to_code": False,
        "coder_definition": {},
        "data_source_location": "",
        "output_location": "",
        "infrastructure_request": {},
        "infrastructure_provisioned": {},
        "execution_result": {},
        "output_file_data": {},
        "planner_graph_path": "",
        "planner_graph_status": ""
    }
    
    # Test the state update
    result_state = planner_graph_agent_node(test_state)
    
    # Verify state structure matches ETLState
    required_fields = [
        "messages", "planner_definition", "ready_to_summarize", 
        "ready_to_code", "coder_definition", "data_source_location",
        "output_location", "infrastructure_request", "infrastructure_provisioned",
        "execution_result", "output_file_data", "planner_graph_path",
        "planner_graph_status"
    ]
    
    for field in required_fields:
        assert field in result_state, f"Missing required field: {field}"
    
    # Verify planner graph fields are properly set
    assert isinstance(result_state.get('planner_graph_status'), str), \
        "planner_graph_status should be a string"
    assert isinstance(result_state.get('planner_graph_path'), str), \
        "planner_graph_path should be a string"
            
    print("✓ Workflow integration test passed")

# Pytest-compatible test functions
def test_standalone_agent_with_pytest():
    """Pytest version of standalone agent test"""
    sample_plan = {
        "plan_id": "pytest_test",
        "title": "Pytest Test Plan",
        "steps": [
            {
                "step_id": 1,
                "name": "Test Step",
                "type": "data_input",
                "description": "Test step description",
                "inputs": ["input.csv"],
                "outputs": ["output.csv"],
                "sql_query": None,
                "parameters": None
            }
        ]
    }
    
    graph_agent = PlannerGraphAgent()
    result = graph_agent.generate_planner_graph(sample_plan)
    
    # Use proper pytest assertions
    assert 'status' in result
    assert 'plan_id' in result
    assert 'base64_image' not in result  # Should not have base64
    assert result['plan_id'] == 'pytest_test'
    
    status = result['status']
    assert status == 'success' or status.startswith('error:') or status.startswith('skipped:')

def test_error_handling_with_pytest():
    """Pytest version of error handling test"""
    graph_agent = PlannerGraphAgent()
    
    # Test invalid plan data
    invalid_plan = {"plan_id": "test"}  # Missing steps
    result = graph_agent.generate_planner_graph(invalid_plan)
    
    assert result['status'].startswith('error:')
    assert 'steps' in result['status'].lower()

def test_state_update_with_pytest():
    """Pytest version of state update test"""
    mock_state = {
        "messages": [],
        "planner_definition": {},
        "ready_to_summarize": True,
        "ready_to_code": False,
        "coder_definition": {},
        "data_source_location": "",
        "output_location": "",
        "infrastructure_request": {},
        "infrastructure_provisioned": {},
        "execution_result": {},
        "output_file_data": {},
        "planner_graph_path": "",
        "planner_graph_status": ""
    }
    
    result_state = planner_graph_agent_node(mock_state)
    
    assert 'planner_graph_status' in result_state
    assert 'planner_graph_path' in result_state
    assert result_state['planner_graph_status'].startswith('skipped:')  # No planner definition

def test_with_sample_data():
    print("\n=== Testing plan formatting ===")
    
    # Create more realistic ETL state with updated structure
    realistic_state: ETLState = {
        "messages": [],
        "planner_definition": {
            "job_name": "Employee Analysis",
            "source": "s3://test",
            "transformations": [
                {
                    "name": "Filter Active Employees",
                    "type": "filter",
                    "columns": ["status"],
                },
                {
                    "name": "Calculate Average Salary by Department",
                    "type": "aggregation",
                    "columns": ["department", "salary"],
                    "aggregation": "avg"
                }
            ],
            "destination": "avg_salary_by_dept.csv",
            "schedule": "weekly",
            "notes": "Calculate average salary for active employees by department"
        },
        "ready_to_summarize": True,
        "ready_to_code": False,
        "coder_definition": {},
        "data_source_location": "s3://test",
        "output_location": "test.csv".replace(".csv", "_output.csv"),
        "infrastructure_request": {},
        "infrastructure_provisioned": {},
        "execution_result": {},
        "output_file_data": {},
        "planner_graph_path": "",
        "planner_graph_status": ""
    }
    
    formatted_plan = _convert_plan_to_graph_format(realistic_state["planner_definition"], realistic_state)
    steps = formatted_plan.get("steps", [])

    assert len(steps) == 8, "Missing nodes"
    assert steps[0]["type"] == "data_input"
    assert steps[2]["type"] == "filter"
    assert steps[3]["type"] == "aggregation"
    assert steps[5]["type"] == "schedule_task"
    print("✓ Sample plan formatting passed")
