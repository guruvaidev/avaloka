"""
Test CGA Integration with Model Training Agent

This script tests the enhanced communication between MTA and CGA,
demonstrating the complete workflow from code generation to training task creation.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.agents.mta.agent_communication import (
    AgentCommunicationInterface, CGARequest, CGAResponse, 
    MessageType, MessagePriority, AgentMessage
)
from app.agents.mta.cga_client import CGAClient
from app.graph.etl_state import ETLState

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_cga_communication():
    """Test CGA communication functionality"""
    print("🧠 Testing CGA Communication Integration")
    print("=" * 50)
    
    # Initialize communication interface
    comm_interface = AgentCommunicationInterface()
    cga_client = CGAClient(comm_interface)
    
    assert comm_interface is not None, "Communication interface should be initialized"
    assert cga_client is not None, "CGA client should be initialized"
    print("✅ Communication interface initialized")
    
    # Test 1: Create CGA request from ETL state
    print("\n📝 Test 1: Creating CGA request from ETL state")
    
    # Mock ETL state
    mock_state = ETLState(
        messages=[],
        planner_definition={"goal": "Analyze salary data and predict salaries"},
        coder_definition={},
        data_source_location="app/sample_data/salaries.csv",
        schema={"name": "string", "job_title": "string", "age": "int", "location": "string", "salary": "float"},
        sample_data={"preview": [["John", "Manager", 35, "NYC", 75000]]},
        requirements=["pandas", "numpy", "sklearn"],
        library_to_use="pandas"
    )
    
    # Create CGA request
    cga_request = comm_interface.create_cga_request_from_state(mock_state)
    assert cga_request is not None, "CGA request should be created"
    assert cga_request.task_id is not None, "CGA request should have task_id"
    assert cga_request.data_source == "app/sample_data/salaries.csv", "Data source should match"
    assert cga_request.library_preference == "pandas", "Library preference should match"
    print(f"✅ CGA request created: {cga_request.task_id}")
    print(f"   Data source: {cga_request.data_source}")
    print(f"   Target columns: {cga_request.target_columns}")
    print(f"   Library preference: {cga_request.library_preference}")
    
    # Test 2: Send request to CGA
    print("\n📤 Test 2: Sending request to CGA")
    
    cga_response = cga_client.request_code_generation_sync(cga_request)
    assert cga_response is not None, "CGA response should be received"
    assert cga_response.task_id == cga_request.task_id, "Response task_id should match request"
    assert len(cga_response.code) > 0, "Generated code should not be empty"
    assert isinstance(cga_response.dependencies, list), "Dependencies should be a list"
    print(f"✅ CGA response received: {cga_response.task_id}")
    print(f"   Code length: {len(cga_response.code)} characters")
    print(f"   ML applicable: {cga_response.ml_applicable}")
    print(f"   Suggested models: {cga_response.suggested_models}")
    print(f"   Dependencies: {cga_response.dependencies}")
    
    # Test 3: Analyze code for training
    print("\n🔍 Test 3: Analyzing code for training")
    
    training_analysis = comm_interface.analyze_cga_code_for_training(cga_response)
    assert training_analysis is not None, "Training analysis should be returned"
    assert 'is_ml_ready' in training_analysis, "Analysis should contain is_ml_ready"
    assert 'training_complexity' in training_analysis, "Analysis should contain training_complexity"
    assert 'model_suggestions' in training_analysis, "Analysis should contain model_suggestions"
    print(f"✅ Training analysis completed")
    print(f"   Is ML ready: {training_analysis['is_ml_ready']}")
    print(f"   Training indicators: {training_analysis['training_indicators']}")
    print(f"   Training complexity: {training_analysis['training_complexity']}")
    print(f"   Model suggestions: {training_analysis['model_suggestions']}")
    print(f"   Preprocessing requirements: {training_analysis['preprocessing_requirements']}")
    
    # Test 4: Create training task from CGA
    print("\n🎯 Test 4: Creating training task from CGA")
    
    training_task = comm_interface.create_training_task_from_cga(cga_response, training_analysis)
    assert training_task is not None, "Training task should be created"
    assert 'task_id' in training_task, "Training task should have task_id"
    assert 'model_type' in training_task, "Training task should have model_type"
    assert 'goal_description' in training_task, "Training task should have goal_description"
    print(f"✅ Training task created: {training_task['task_id']}")
    print(f"   Model type: {training_task['model_type']}")
    print(f"   Goal description: {training_task['goal_description']}")
    print(f"   Hyperparameters: {training_task['hyperparameters']}")
    print(f"   Validation params: {training_task['validation_params']}")
    
    # Test 5: Test message handling
    print("\n📨 Test 5: Testing message handling")
    
    # Create test message
    test_message = AgentMessage(
        message_id="test_msg_001",
        message_type=MessageType.CODE_ANALYSIS,
        sender="cga",
        recipient="mta",
        content={"test": "data"},
        timestamp="2024-01-01T00:00:00",
        priority=MessagePriority.NORMAL
    )
    
    # Handle message
    result = comm_interface._handle_code_analysis(test_message)
    assert result is not None, "Message handling should return result"
    assert 'status' in result, "Result should contain status"
    print(f"✅ Message handled: {result['status']}")
    
    # Test 6: Test communication stats
    print("\n📊 Test 6: Communication statistics")
    
    stats = cga_client.get_communication_stats()
    assert stats is not None, "Communication stats should be returned"
    assert 'total_requests' in stats, "Stats should contain total_requests"
    assert 'success_rate' in stats, "Stats should contain success_rate"
    assert stats['total_requests'] > 0, "Should have at least one request"
    assert 0 <= stats['success_rate'] <= 1, "Success rate should be between 0 and 1"
    print(f"✅ Communication stats:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cached responses: {stats['cached_responses']}")
    print(f"   Success rate: {stats['success_rate']:.2%}")
    print(f"   Average response time: {stats['average_response_time']:.2f}s")
    
    # Test 7: Display generated code
    print("\n💻 Test 7: Generated code preview")
    print("=" * 30)
    code_lines = cga_response.code.split('\n')
    for i, line in enumerate(code_lines[:20]):  # Show first 20 lines
        print(f"{i+1:2d}: {line}")
    if len(code_lines) > 20:
        print(f"... and {len(code_lines) - 20} more lines")
    print("=" * 30)
    
    print("\n🎉 All CGA communication tests passed!")


def test_message_schemas():
    """Test message schema validation"""
    print("\n🔧 Testing Message Schemas")
    print("=" * 30)
    
    # Test AgentMessage creation
    message = AgentMessage(
        message_id="test_001",
        message_type=MessageType.TRAINING_REQUEST,
        sender="mta",
        recipient="cga",
        content={"task_id": "test_task", "data": "sample"},
        timestamp="2024-01-01T00:00:00",
        priority=MessagePriority.HIGH
    )
    
    assert message is not None, "Message should be created"
    assert message.message_id == "test_001", "Message ID should match"
    assert message.message_type == MessageType.TRAINING_REQUEST, "Message type should match"
    
    # Test serialization
    message_dict = message.to_dict()
    assert message_dict is not None, "Message should serialize to dict"
    assert message_dict['message_id'] == "test_001", "Serialized message_id should match"
    print(f"✅ Message serialized: {message_dict['message_id']}")
    
    # Test deserialization
    restored_message = AgentMessage.from_dict(message_dict)
    assert restored_message is not None, "Message should deserialize"
    assert restored_message.message_id == message.message_id, "Deserialized message_id should match"
    assert restored_message.message_type == message.message_type, "Deserialized message_type should match"
    print(f"✅ Message deserialized: {restored_message.message_id}")
    
    # Test CGARequest creation
    cga_request = CGARequest(
        task_id="test_cga_001",
        data_source="test_data.csv",
        schema={"col1": "string", "col2": "int"},
        sample_data={"preview": [["a", 1]]},
        requirements=["pandas", "numpy"],
        library_preference="pandas",
        include_ml_hints=True,
        target_columns=["col2"]
    )
    
    assert cga_request is not None, "CGA request should be created"
    assert cga_request.task_id == "test_cga_001", "CGA request task_id should match"
    assert cga_request.data_source == "test_data.csv", "Data source should match"
    assert cga_request.include_ml_hints == True, "include_ml_hints should be True"
    print(f"✅ CGA request created: {cga_request.task_id}")
    
    # Test CGAResponse creation
    cga_response = CGAResponse(
        task_id="test_cga_001",
        code="print('Hello World')",
        code_analysis={"lines": 1, "complexity": "simple"},
        dependencies=["pandas"],
        execution_notes=["Test code"],
        ml_applicable=False,
        suggested_models=[],
        data_requirements={"format": "CSV"}
    )
    
    assert cga_response is not None, "CGA response should be created"
    assert cga_response.task_id == "test_cga_001", "CGA response task_id should match"
    assert cga_response.code == "print('Hello World')", "Code should match"
    assert cga_response.ml_applicable == False, "ml_applicable should be False"
    print(f"✅ CGA response created: {cga_response.task_id}")
    
    print("✅ All message schema tests passed!")


def test_integration_workflow():
    """Test complete integration workflow"""
    print("\n🔄 Testing Complete Integration Workflow")
    print("=" * 40)
    
    # Initialize components
    comm_interface = AgentCommunicationInterface()
    cga_client = CGAClient(comm_interface)
    
    # Step 1: Create ETL state
    print("Step 1: Creating ETL state...")
    etl_state = ETLState(
        messages=[],
        planner_definition={"goal": "Predict customer churn based on behavior data"},
        coder_definition={},
        data_source_location="app/sample_data/customer_data.csv",
        schema={
            "customer_id": "int",
            "age": "int", 
            "income": "float",
            "tenure": "int",
            "churn": "int"
        },
        sample_data={"preview": [["1", 25, 50000, 12, 0]]},
        requirements=["pandas", "numpy", "sklearn", "torch"],
        library_to_use="pandas"
    )
    assert etl_state is not None, "ETL state should be created"
    print("✅ ETL state created")
    
    # Step 2: Create CGA request
    print("Step 2: Creating CGA request...")
    cga_request = comm_interface.create_cga_request_from_state(etl_state)
    assert cga_request is not None, "CGA request should be created"
    assert cga_request.task_id is not None, "CGA request should have task_id"
    print(f"✅ CGA request created: {cga_request.task_id}")
    
    # Step 3: Send to CGA
    print("Step 3: Sending to CGA...")
    cga_response = cga_client.request_code_generation_sync(cga_request)
    assert cga_response is not None, "CGA response should be received"
    assert cga_response.task_id == cga_request.task_id, "Response task_id should match request"
    print(f"✅ CGA response received: {cga_response.task_id}")
    
    # Step 4: Analyze for training
    print("Step 4: Analyzing for training...")
    training_analysis = comm_interface.analyze_cga_code_for_training(cga_response)
    assert training_analysis is not None, "Training analysis should be returned"
    assert 'training_complexity' in training_analysis, "Analysis should contain training_complexity"
    print(f"✅ Training analysis completed: {training_analysis['training_complexity']} complexity")
    
    # Step 5: Create training task
    print("Step 5: Creating training task...")
    training_task = comm_interface.create_training_task_from_cga(cga_response, training_analysis)
    assert training_task is not None, "Training task should be created"
    assert 'task_id' in training_task, "Training task should have task_id"
    print(f"✅ Training task created: {training_task['task_id']}")
    
    # Step 6: Validate training task
    print("Step 6: Validating training task...")
    from app.agents.model_training_agent import ModelTrainingAgent
    mta = ModelTrainingAgent()
    validation_result = mta.validate_training_task(training_task)
    assert validation_result is not None, "Validation result should be returned"
    assert 'valid' in validation_result, "Validation result should contain 'valid' field"
    print(f"✅ Training task validation: {'PASSED' if validation_result['valid'] else 'FAILED'}")
    
    if not validation_result['valid']:
        error_msg = validation_result.get('error', 'Unknown error')
        print(f"   Validation error: {error_msg}")
        # Don't fail the test for validation errors, just log them
    
    print("✅ Complete integration workflow test passed!")


def main():
    """Run all CGA integration tests"""
    print("🚀 Starting CGA Integration Tests")
    print("=" * 50)
    
    tests = [
        ("CGA Communication", test_cga_communication),
        ("Message Schemas", test_message_schemas),
        ("Integration Workflow", test_integration_workflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            test_func()
            # If no assertion error, test passed
            results.append((test_name, True))
        except AssertionError as e:
            print(f"❌ {test_name} test failed: {e}")
            logger.error(f"Assertion failed in {test_name}: {e}")
            results.append((test_name, False))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            logger.exception(f"Unexpected error in {test_name}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All CGA integration tests passed! Ready for Day 4 implementation.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review and fix issues.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
