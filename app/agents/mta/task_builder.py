"""
Training Task Builder for Model Training Agent

Creates and configures training tasks from various sources (CGA, state, planning).
Handles hyperparameter inference, validation configuration, and task metadata.
"""

import re
import os
import json
import logging
from typing import Dict, Any, List, Optional
from app.graph.etl_state import ETLState
from app.core.inference import build_chat_model
from .agent_communication import CGARequest, CGAResponse
from .training_task import TrainingTask, HyperparameterConfig, ValidationConfig, RayConfig, DataConfig
from .model_types import ModelType, TaskType, TrainingStrategy
from app.core.model_config import resolve as resolve_model
from app.core.model_fallback import attach_fallback

logger = logging.getLogger(__name__)


class TrainingTaskBuilder:
    """
    Builds training tasks with appropriate configuration
    
    Responsibilities:
    - Create training tasks from CGA responses
    - Create training tasks from ETL state
    - Generate task IDs
    - Infer hyperparameters
    - Infer validation parameters
    - Determine model types
    - Extract/infer training goals
    """
    
    def __init__(self):
        """Initialize training task builder"""
        pass
    
    def create_training_task_from_cga(self, cga_response: CGAResponse, 
                                    training_analysis: Dict[str, Any]) -> TrainingTask:
        """
        Create a comprehensive training task from CGA response and analysis
        
        Args:
            cga_response: Response from Code Generation Agent
            training_analysis: Analysis of the code for training
            
        Returns:
            Complete training task definition
        """
        try:
            task_id = f"mta_from_cga_{cga_response.task_id}"
            
            # Determine model type based on analysis
            model_type = self._determine_model_type_from_analysis(training_analysis)
            
            # Extract goal from code analysis
            goal_description = self._extract_goal_from_code(cga_response.code, training_analysis)
            
            # Create hyperparameters based on complexity
            hyperparameters = self._create_hyperparameters_from_analysis(training_analysis)
            
            # Create validation parameters
            validation_params = self._create_validation_params_from_analysis(training_analysis)

            model_name, model_description = self._determine_model_name_and_description()
            
            # Create data configuration
            data_config = DataConfig(
                data_source=cga_response.data_source,
                target_column=training_analysis.get("target_column"),
                feature_columns=training_analysis.get("feature_columns"),
                data_preprocessing=training_analysis.get("data_preprocessing"),
                data_validation=training_analysis.get("data_validation")
            )
            
            # Create hyperparameter configuration
            hyperparams_config = None
            if hyperparameters:
                hyperparams_config = HyperparameterConfig(**hyperparameters)
            
            # Create validation configuration
            validation_config = None
            if validation_params:
                validation_config = ValidationConfig(**validation_params)
            
            # Create training task with Pydantic model
            training_task = TrainingTask(
                task_id=task_id,
                task_name=f"Training from CGA: {cga_response.task_id}",
                model_type=model_type,
                model_name=model_name,
                model_description=model_description,
                model_version="v1.0",
                task_type=self._infer_task_type_from_analysis(training_analysis),
                training_strategy=self._infer_training_strategy_from_analysis(training_analysis),
                data_config=data_config,
                hyperparameters=hyperparams_config,
                validation_params=validation_config,
                primary_metric=self._infer_primary_metric_from_analysis(training_analysis),
                training_goals=[goal_description] if goal_description else None,
                source="CGA",
                description=f"Training task generated from CGA response: {cga_response.task_id}",
                tags=["cga_generated", "automated"],
                execution_context={
                    "code_analysis": cga_response.code_analysis,
                    "training_analysis": training_analysis,
                    "dependencies": cga_response.dependencies,
                    "execution_notes": cga_response.execution_notes,
                    "data_requirements": cga_response.data_requirements
                }
            )
            
            logger.info(f"Created training task from CGA: {task_id}")
            return training_task
            
        except Exception as e:
            logger.error(f"Error creating training task from CGA: {e}")
            return {"error": str(e)}

    def _determine_model_name_and_description(self, state: ETLState):
        """Determine model name and description from state information"""
        try:
            api_key = os.environ.get("GROQ_API_KEY_PLANNING_AGENT")
            llm = build_chat_model(
                role="planning",
                agent="MTA_TASK_BUILDER",
                tier="large",
                temperature=0.2,
                groq_model=resolve_model("mta_task_builder"),
                groq_api_key=api_key,
            )
            
            system_prompt = """
You are an assistant that helps determine the appropriate machine learning model name and a concise description of the model's purpose based on recent chat history.
Model name should be 2~3 words like "Finance Anomoly Detector" or "Customer Churn Predictor". Description should be a one-sentence summary of the model's goal (max 15 words).
Output name and description as following JSON format:
```{
    "model_name": str,
    "model_description": str
}```
"""
            messages = [{"role": "system", "content": system_prompt}] + state.get("messages")[-5:]
            response = llm.invoke(messages).content
            
            pattern = r"```\s*(.*?)\s*```"
            matche = re.findall(pattern, response, re.DOTALL)[0]

            if matche.startswith("json"):
                matche = matche[4:]
            
            model_data = json.loads(matche)
            logger.info(f"Determined model name and description from state: {model_data}")

            return model_data.get("model_name"), model_data.get("model_description")
        except Exception as e:
            logger.error(f"Error determining model name and description: {e}")
            return "Unknown Model", "No description available"
    
    def create_training_task_from_state(self, state: ETLState, 
                                       cga_info: Dict[str, Any],
                                       data_info: Dict[str, Any],
                                       planning_context: Dict[str, Any]) -> TrainingTask:
        """
        Create training task definition from current state
        
        Args:
            state: Current ETL state
            cga_info: Extracted CGA code information
            data_info: Extracted data information
            planning_context: Extracted planning context
            
        Returns:
            Training task dictionary
        """
        try:
            # Check if training task already exists in state
            existing_training_task = state.get("training_task")
            if existing_training_task and existing_training_task.get("task_id"):
                logger.info(f"Using existing training task ID: {existing_training_task['task_id']}")
                return existing_training_task
            
            # Create task ID based on available information
            task_id = self._generate_task_id(cga_info, data_info, planning_context)
            
            # Determine model type based on data and code analysis
            model_type = self._suggest_model_type(cga_info, data_info, planning_context)
            
            # Extract or infer business goal
            goal_description = self._extract_or_infer_goal(planning_context, cga_info, data_info)

            # If possible, determine model name and description from state messages
            model_name, model_description = self._determine_model_name_and_description(state)
            
            # Create training task
            training_task = {
                "task_id": task_id,
                "user_id": state.get("user_id"),
                "session_id": state.get("session_id"),
                "model_type": model_type,
                "model_name": model_name,
                "model_description": model_description,
                "model_version": "v1.0",
                "goal_description": goal_description,
                "created_from": "state_analysis",
                "source_agents": {
                    "cga_available": cga_info.get("has_code", False),
                    "data_available": data_info.get("has_data", False),
                    "planning_available": planning_context.get("has_planning", False)
                }
            }
            
            # Add hyperparameters if we can infer them
            hyperparameters = self._infer_hyperparameters(cga_info, data_info)
            if hyperparameters:
                training_task["hyperparameters"] = hyperparameters
            
            # Add validation parameters
            validation_params = self._infer_validation_params(data_info)
            if validation_params:
                training_task["validation_params"] = validation_params
            
            # Add data configuration with target column if extracted from messages
            extracted_target = data_info.get("extracted_target_column")
            if extracted_target:
                # Get data source
                data_source = state.get("data_source_location") or data_info.get("source_location", "")
                
                # Check if data_config already exists, merge instead of overwrite
                if "data_config" in training_task and isinstance(training_task["data_config"], dict):
                    # Merge new fields into existing data_config
                    training_task["data_config"]["target_column"] = extracted_target
                    if not training_task["data_config"].get("data_source"):
                        training_task["data_config"]["data_source"] = data_source
                    logger.info(f"Task Builder: Updated existing data_config with extracted target column '{extracted_target}'")
                else:
                    # Create new data_config if it doesn't exist
                    training_task["data_config"] = {
                        "data_source": data_source,
                        "target_column": extracted_target,
                        "feature_columns": None,  # Will be inferred during training
                        "data_preprocessing": None,
                        "data_validation": None
                    }
                    logger.info(f"Task Builder: Added extracted target column '{extracted_target}' to training task")
            
            logger.info(f"Created training task from state: {task_id}")
            return training_task
            
        except Exception as e:
            logger.error(f"Error creating training task from state: {e}")
            return {
                "error": f"Failed to create training task: {str(e)}"
            }
    
    def create_cga_request_from_state(self, state: ETLState) -> CGARequest:
        """
        Create a CGA request from the current ETL state
        
        Args:
            state: Current ETL state
            
        Returns:
            Structured CGA request
        """
        try:
            # Extract data from state
            data_source = state.get("data_source_location", "")
            schema = state.get("schema", {})
            sample_data = state.get("sample_data", {})
            requirements = state.get("requirements", [])
            library_preference = state.get("library_to_use", "pandas")
            
            # Generate task ID
            task_id = self._generate_task_id_from_state(state)
            
            # Determine target columns for ML
            target_columns = self._infer_target_columns(schema, sample_data)
            
            # Create CGA request
            request = CGARequest(
                task_id=task_id,
                data_source=data_source,
                schema=schema,
                sample_data=sample_data,
                requirements=requirements,
                library_preference=library_preference,
                code_style="clean_executable",
                include_ml_hints=True,
                target_columns=target_columns
            )
            
            logger.info(f"Created CGA request from state: {task_id}")
            return request
            
        except Exception as e:
            logger.error(f"Error creating CGA request from state: {e}")
            raise
    
    # Helper methods for task configuration
    
    def _determine_model_type_from_analysis(self, analysis: Dict[str, Any]) -> str:
        """Determine model type from training analysis"""
        suggestions = analysis.get("model_suggestions", [])
        
        if "pytorch_classification" in suggestions:
            return "pytorch_classification"
        elif "pytorch_regression" in suggestions:
            return "pytorch_regression"
        elif "sklearn_classification" in suggestions:
            return "sklearn_classification"
        elif "sklearn_regression" in suggestions:
            return "sklearn_regression"
        else:
            return "pytorch_classification"  # Default
    
    def _extract_goal_from_code(self, code: str, analysis: Dict[str, Any]) -> str:
        """Extract training goal from code analysis"""
        suggestions = analysis.get("model_suggestions", [])
        
        if "classification" in str(suggestions):
            return "Classify data samples into appropriate categories"
        elif "regression" in str(suggestions):
            return "Predict continuous values based on input features"
        elif "clustering" in str(suggestions):
            return "Group data samples into meaningful clusters"
        else:
            return "Train a machine learning model on the provided data"
    
    def _create_hyperparameters_from_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create hyperparameters based on training analysis"""
        complexity = analysis.get("training_complexity", "medium")
        
        if complexity == "simple":
            return {
                "learning_rate": 0.01,
                "epochs": 20,
                "batch_size": 16,
                "hidden_sizes": [32]
            }
        elif complexity == "complex":
            return {
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 64,
                "hidden_sizes": [128, 64, 32],
                "dropout": 0.3
            }
        else:  # medium
            return {
                "learning_rate": 0.005,
                "epochs": 50,
                "batch_size": 32,
                "hidden_sizes": [64, 32],
                "dropout": 0.2
            }
    
    def _create_validation_params_from_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation parameters from training analysis"""
        complexity = analysis.get("training_complexity", "medium")
        
        if complexity == "simple":
            return {
                "k_folds": 3,
                "metric_threshold": 0.7,
                "validation_split": 0.2
            }
        elif complexity == "complex":
            return {
                "k_folds": 10,
                "metric_threshold": 0.9,
                "validation_split": 0.1
            }
        else:  # medium
            return {
                "k_folds": 5,
                "metric_threshold": 0.8,
                "validation_split": 0.2
            }
    
    def _generate_task_id(self, cga_info: Dict, data_info: Dict, planning_context: Dict) -> str:
        """Generate unique task ID based on available information"""
        from app.utils import generate_filename_timestamp
        
        components = []
        
        if cga_info.get("has_code"):
            components.append("cga")
        if data_info.get("has_data"):
            components.append("data")
        if planning_context.get("has_planning"):
            components.append("planned")
        
        if not components:
            components.append("auto")
        
        return f"mta_{'_'.join(components)}_{generate_filename_timestamp()}"
    
    def _suggest_model_type(self, cga_info: Dict, data_info: Dict, planning_context: Dict) -> str:
        """Suggest appropriate model type based on available information"""
        # Default to classification
        default_type = "pytorch_classification"
        
        # Check data characteristics
        data_chars = data_info.get("data_characteristics", {})
        if data_chars.get("problem_type") == "regression":
            return "pytorch_regression"
        elif data_chars.get("problem_type") == "classification":
            return "pytorch_classification"
        
        # Check planning hints
        hints = planning_context.get("training_hints", [])
        if "regression_task" in hints:
            return "pytorch_regression"
        elif "classification_task" in hints:
            return "pytorch_classification"
        
        # Check code type
        if cga_info.get("code_type") == "machine_learning":
            # Could analyze code further to determine specific type
            pass
        
        return default_type
    
    def _extract_or_infer_goal(self, planning_context: Dict, cga_info: Dict, data_info: Dict) -> str:
        """Extract or infer the training goal"""
        # Try to extract from planning context first
        if planning_context.get("business_goal"):
            return planning_context["business_goal"]
        
        # Infer from data characteristics
        data_chars = data_info.get("data_characteristics", {})
        problem_type = data_chars.get("problem_type", "unknown")
        
        if problem_type == "classification":
            return "Classify data samples into appropriate categories"
        elif problem_type == "regression":
            return "Predict continuous values based on input features"
        
        # Default goal
        return "Train a machine learning model on the provided data"
    
    def _infer_hyperparameters(self, cga_info: Dict, data_info: Dict) -> Dict[str, Any]:
        """Infer appropriate hyperparameters based on data and code"""
        hyperparams = {}
        
        # Adjust based on data complexity
        data_chars = data_info.get("data_characteristics", {})
        complexity = data_chars.get("complexity", "medium")
        
        if complexity == "simple":
            hyperparams.update({
                "learning_rate": 0.01,
                "epochs": 20,
                "batch_size": 16
            })
        elif complexity == "complex":
            hyperparams.update({
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 64
            })
        else:  # medium
            hyperparams.update({
                "learning_rate": 0.005,
                "epochs": 30,
                "batch_size": 32
            })
        
        return hyperparams
    
    def _infer_validation_params(self, data_info: Dict) -> Dict[str, Any]:
        """Infer validation parameters based on data"""
        val_params = {}
        
        data_chars = data_info.get("data_characteristics", {})
        num_features = data_chars.get("num_features", 0)
        
        # Adjust k-folds based on data size/complexity
        if num_features < 5:
            val_params["k_folds"] = 3
        elif num_features > 20:
            val_params["k_folds"] = 10
        else:
            val_params["k_folds"] = 5
        
        val_params["metric_threshold"] = 0.8
        
        return val_params
    
    def _generate_task_id_from_state(self, state: ETLState) -> str:
        """Generate task ID from ETL state"""
        from app.utils import generate_filename_timestamp
        return f"cga_req_{generate_filename_timestamp()}"
    
    def _infer_target_columns(self, schema: Dict[str, Any], sample_data: Dict[str, Any]) -> List[str]:
        """Infer target columns for ML from schema and sample data"""
        target_indicators = ["target", "label", "class", "y", "outcome", "result", "price", "amount", "value"]
        target_columns = []
        
        for col in schema.keys():
            if any(indicator in col.lower() for indicator in target_indicators):
                target_columns.append(col)
        
        return target_columns


