"""
Code Generation Agent (CGA) Client for Model Training Agent

This module provides a client interface for communicating with the Code Generation Agent,
including request/response handling, validation, and integration with the MTA workflow.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from datetime import datetime

from app.agents.mta.agent_communication import (
    CGARequest, CGAResponse, AgentMessage, MessageType, MessagePriority
)
from app.graph.etl_state import ETLState
from app.agents.coder import coder_node

logger = logging.getLogger(__name__)


class CGAClient:
    """Client for communicating with the Code Generation Agent"""
    
    def __init__(self, communication_interface=None):
        """
        Initialize CGA client
        
        Args:
            communication_interface: AgentCommunicationInterface instance
        """
        self.communication_interface = communication_interface
        self.request_history = []
        self.response_cache = {}
    
    def request_code_generation_sync(self, request: CGARequest) -> CGAResponse:
        """
        Synchronous version of code generation request
        
        Args:
            request: CGA request with all necessary information
            
        Returns:
            CGAResponse from CGA
        """
        try:
            logger.info(f"Requesting code generation (sync) for task: {request.task_id}")
            
            # Create agent message
            if self.communication_interface:
                message = self.communication_interface.send_cga_request(request)
                self.request_history.append(message)
            
            # Simulate CGA processing (in real implementation, this would call CGA service)
            response_data = self._simulate_cga_processing_sync(request)
            
            # Process response
            if self.communication_interface:
                cga_response = self.communication_interface.process_cga_response(response_data)
            else:
                cga_response = self._create_cga_response_from_data(response_data)
            
            # Cache response
            self.response_cache[request.task_id] = cga_response
            
            logger.info(f"Received code generation response (sync) for task: {request.task_id}")
            return cga_response
            
        except Exception as e:
            logger.error(f"Error requesting code generation (sync): {e}")
            raise
    def _simulate_cga_processing_sync(self, request: CGARequest) -> Dict[str, Any]:
        """
        Call actual CGA (coder_node) to generate code using LLM
        
        Args:
            request: CGA request
            
        Returns:
            Response data with LLM-generated code
        """
        try:
            # Convert CGARequest to ETLState format for coder_node
            state = self._create_state_from_request(request)
            
            # Call actual coder_node (uses ChatGroq LLM)
            logger.info(f"Calling coder_node with LLM for task: {request.task_id}")
            result_state = coder_node(state)
            
            # Extract generated code from result
            code = result_state.get("coder_definition", {}).get("code", "")
            
            if not code:
                # Fallback to mock if coder_node fails
                logger.warning("coder_node returned no code, using fallback")
                code = self._generate_sample_code(request)
            
            # Analyze the generated code
            code_analysis = self._analyze_generated_code(code)
            
            # Create response data
            response_data = {
                "task_id": request.task_id,
                "code": code,
                "code_analysis": code_analysis,
                "dependencies": self._extract_dependencies(code),
                "execution_notes": self._generate_execution_notes(request),
                "ml_applicable": self._is_ml_applicable(code),
                "suggested_models": self._suggest_models(code),
                "data_requirements": self._extract_data_requirements(request),
                "error": None
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in CGA processing: {e}, falling back to mock")
            # Fallback to mock generation on error
            code = self._generate_sample_code(request)
            code_analysis = self._analyze_generated_code(code)
            
            return {
                "task_id": request.task_id,
                "code": code,
                "code_analysis": code_analysis,
                "dependencies": self._extract_dependencies(code),
                "execution_notes": self._generate_execution_notes(request),
                "ml_applicable": self._is_ml_applicable(code),
                "suggested_models": self._suggest_models(code),
                "data_requirements": self._extract_data_requirements(request),
                "error": str(e)
            }
    
    def _create_state_from_request(self, request: CGARequest) -> ETLState:
        """
        Convert CGARequest to ETLState format for coder_node
        
        Args:
            request: CGA request
            
        Returns:
            ETLState compatible with coder_node
        """
        # Convert sample_data dict to string format if needed
        sample_data_str = str(request.sample_data) if request.sample_data else "No sample available"
        
        # Create a minimal ETLState for coder_node
        state = ETLState(
            messages=[],
            data_source_location=request.data_source,
            schema=request.schema,
            sample_data=sample_data_str,
            planner_definition={
                "requirements": request.requirements,
                "library": request.library_preference
            },
            library_to_use=request.library_preference,
            requirements=request.requirements,
            code_validation_feedback=None,
            code=None
        )
        
        return state
    
    def _generate_sample_code(self, request: CGARequest) -> str:
        """Generate sample code based on CGA request"""
        library = request.library_preference
        data_source = request.data_source
        schema = request.schema
        target_columns = request.target_columns or []
        
        # Generate basic data processing code
        code_lines = [
            f"import {library} as pd",
            "import numpy as np",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler, LabelEncoder",
            "from sklearn.metrics import accuracy_score, mean_squared_error",
            "",
            f"# Load data from {data_source}",
            f"df = pd.read_csv('{data_source}')",
            "",
            "# Display basic information",
            "print('Dataset shape:', df.shape)",
            "print('\\nFirst few rows:')",
            "print(df.head())",
            "",
            "# Check for missing values",
            "print('\\nMissing values:')",
            "print(df.isnull().sum())",
            ""
        ]
        
        # Add target column handling if specified
        if target_columns:
            code_lines.extend([
                f"# Target columns: {', '.join(target_columns)}",
                f"target_cols = {target_columns}",
                ""
            ])
        
        # Add basic preprocessing
        if schema:
            numeric_cols = [col for col in schema.keys() if col not in target_columns]
            if numeric_cols:
                code_lines.extend([
                    "# Preprocess numeric features",
                    f"numeric_features = {numeric_cols}",
                    "scaler = StandardScaler()",
                    "df[numeric_features] = scaler.fit_transform(df[numeric_features])",
                    ""
                ])
        
        # Add ML-ready structure if ML hints are requested
        if request.include_ml_hints and target_columns:
            code_lines.extend([
                "# Prepare features and target for ML",
                f"X = df.drop(columns={target_columns})",
                f"y = df[{target_columns[0] if len(target_columns) == 1 else target_columns}]",
                "",
                "# Split data for training and testing",
                "X_train, X_test, y_train, y_test = train_test_split(",
                "    X, y, test_size=0.2, random_state=42",
                ")",
                "",
                "print(f'Training set shape: {X_train.shape}')",
                "print(f'Test set shape: {X_test.shape}')",
                ""
            ])
        
        # Add output saving
        code_lines.extend([
            "# Save processed data",
            "output_file = 'processed_data.csv'",
            "df.to_csv(output_file, index=False)",
            "print(f'Processed data saved to {output_file}')",
            ""
        ])
        
        return "\n".join(code_lines)
    
    def _analyze_generated_code(self, code: str) -> Dict[str, Any]:
        """Analyze the generated code"""
        import re
        
        analysis = {
            "lines_of_code": len(code.split('\n')),
            "has_imports": bool(re.search(r'^import\s+|^from\s+', code, re.MULTILINE)),
            "has_data_loading": bool(re.search(r'read_csv|read_', code, re.IGNORECASE)),
            "has_preprocessing": bool(re.search(r'scaler|encode|transform', code, re.IGNORECASE)),
            "has_ml_structure": bool(re.search(r'train_test_split|X_train|y_train', code, re.IGNORECASE)),
            "has_output_saving": bool(re.search(r'to_csv|save', code, re.IGNORECASE)),
            "complexity_score": self._calculate_complexity_score(code),
            "ml_libraries": self._extract_ml_libraries(code)
        }
        
        return analysis
    
    def _calculate_complexity_score(self, code: str) -> int:
        """Calculate complexity score for the code"""
        score = 0
        
        # Count lines
        lines = code.split('\n')
        score += min(len(lines) // 10, 5)  # Max 5 points for lines
        
        # Count operations
        operations = ['groupby', 'merge', 'apply', 'transform', 'pivot']
        for op in operations:
            if op in code.lower():
                score += 1
        
        # Count ML operations
        ml_ops = ['train_test_split', 'fit', 'predict', 'score']
        for op in ml_ops:
            if op in code.lower():
                score += 2
        
        return min(score, 10)  # Cap at 10
    
    def _extract_ml_libraries(self, code: str) -> List[str]:
        """Extract ML libraries from code"""
        import re
        
        libraries = []
        lib_patterns = {
            'sklearn': r'sklearn|sklearn\.',
            'pandas': r'pandas|pd\.',
            'numpy': r'numpy|np\.',
            'torch': r'torch|torch\.',
            'tensorflow': r'tensorflow|tf\.'
        }
        
        for lib, pattern in lib_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                libraries.append(lib)
        
        return libraries
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from code"""
        dependencies = []
        
        # Common dependencies
        if 'pandas' in code:
            dependencies.append('pandas')
        if 'numpy' in code:
            dependencies.append('numpy')
        if 'sklearn' in code:
            dependencies.append('scikit-learn')
        if 'torch' in code:
            dependencies.append('torch')
        if 'tensorflow' in code:
            dependencies.append('tensorflow')
        
        return list(set(dependencies))
    
    def _generate_execution_notes(self, request: CGARequest) -> List[str]:
        """Generate execution notes for the code"""
        notes = [
            f"Code generated for task: {request.task_id}",
            f"Data source: {request.data_source}",
            f"Library preference: {request.library_preference}",
            f"Code style: {request.code_style}"
        ]
        
        if request.target_columns:
            notes.append(f"Target columns: {', '.join(request.target_columns)}")
        
        if request.include_ml_hints:
            notes.append("Code includes ML-ready structure")
        
        return notes
    
    def _is_ml_applicable(self, code: str) -> bool:
        """Determine if code is applicable for ML training"""
        ml_indicators = [
            'train_test_split', 'X_train', 'y_train', 'fit', 'predict',
            'accuracy_score', 'mean_squared_error', 'scaler'
        ]
        
        return any(indicator in code for indicator in ml_indicators)
    
    def _suggest_models(self, code: str) -> List[str]:
        """Suggest appropriate models based on code"""
        suggestions = []
        
        if 'train_test_split' in code:
            if any(word in code.lower() for word in ['classify', 'category', 'label']):
                suggestions.extend(['pytorch_classification', 'sklearn_classification'])
            elif any(word in code.lower() for word in ['predict', 'regress', 'value', 'amount']):
                suggestions.extend(['pytorch_regression', 'sklearn_regression'])
        
        return list(set(suggestions))
    
    def _extract_data_requirements(self, request: CGARequest) -> Dict[str, Any]:
        """Extract data requirements from request"""
        return {
            "data_source": request.data_source,
            "schema": request.schema,
            "target_columns": request.target_columns,
            "library_preference": request.library_preference,
            "expected_format": "CSV",
            "required_columns": list(request.schema.keys()) if request.schema else []
        }
    
    def _create_cga_response_from_data(self, response_data: Dict[str, Any]) -> CGAResponse:
        """Create CGAResponse from response data"""
        return CGAResponse(
            task_id=response_data["task_id"],
            code=response_data["code"],
            code_analysis=response_data.get("code_analysis", {}),
            dependencies=response_data.get("dependencies", []),
            execution_notes=response_data.get("execution_notes", []),
            ml_applicable=response_data.get("ml_applicable", False),
            suggested_models=response_data.get("suggested_models", []),
            data_requirements=response_data.get("data_requirements", {}),
            error=response_data.get("error")
        )
    
    def get_cached_response(self, task_id: str) -> Optional[CGAResponse]:
        """Get cached response by task ID"""
        return self.response_cache.get(task_id)
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("CGA response cache cleared")
    
    def get_request_history(self) -> List[AgentMessage]:
        """Get request history"""
        return self.request_history.copy()
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            "total_requests": len(self.request_history),
            "cached_responses": len(self.response_cache),
            "success_rate": self._calculate_success_rate(),
            "average_response_time": self._calculate_avg_response_time()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate of requests"""
        if not self.request_history:
            return 0.0
        
        successful_requests = sum(1 for req in self.request_history 
                               if req.message_type != MessageType.ERROR)
        return successful_requests / len(self.request_history)
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time (simulated)"""
        # In real implementation, this would track actual response times
        return 0.5  # Simulated 500ms average
