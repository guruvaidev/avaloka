"""
Code Analyzer for Model Training Agent

Analyzes code from CGA to determine ML applicability and requirements.
Extracts features, identifies patterns, and suggests models.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from app.graph.etl_state import ETLState
from .agent_communication import CGAResponse

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Analyzes code for ML training applicability
    
    Responsibilities:
    - Detect code type (ETL, ML, analysis, etc.)
    - Find ML indicators and patterns
    - Analyze data flow
    - Suggest appropriate models
    - Identify preprocessing requirements
    - Find feature engineering opportunities
    """
    
    def __init__(self):
        """Initialize code analyzer"""
        pass
    
    def analyze_cga_code_for_training(self, cga_response: CGAResponse) -> Dict[str, Any]:
        """
        Analyze CGA code to determine training applicability and requirements
        
        Args:
            cga_response: Response from Code Generation Agent
            
        Returns:
            Analysis results for training decisions
        """
        try:
            code = cga_response.code
            analysis = cga_response.code_analysis
            
            training_analysis = {
                "is_ml_ready": False,
                "training_indicators": [],
                "data_flow_analysis": {},
                "model_suggestions": [],
                "preprocessing_requirements": [],
                "feature_engineering_opportunities": [],
                "training_complexity": "unknown"
            }
            
            # Analyze code for ML indicators
            ml_indicators = self._find_ml_indicators(code)
            training_analysis["training_indicators"] = ml_indicators
            
            # Determine if code is ML-ready
            training_analysis["is_ml_ready"] = len(ml_indicators) > 0
            
            # Analyze data flow
            data_flow = self._analyze_data_flow(code)
            training_analysis["data_flow_analysis"] = data_flow
            
            # Suggest models based on code analysis
            model_suggestions = self._suggest_models_from_code(code, analysis)
            training_analysis["model_suggestions"] = model_suggestions
            
            # Identify preprocessing requirements
            preprocessing = self._identify_preprocessing_requirements(code)
            training_analysis["preprocessing_requirements"] = preprocessing
            
            # Find feature engineering opportunities
            feature_eng = self._find_feature_engineering_opportunities(code)
            training_analysis["feature_engineering_opportunities"] = feature_eng
            
            # Assess training complexity
            complexity = self._assess_training_complexity(code, analysis)
            training_analysis["training_complexity"] = complexity
            
            logger.info(f"Analyzed CGA code for training: {cga_response.task_id}")
            return training_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing CGA code for training: {e}")
            return {"error": str(e)}
    
    def extract_cga_code(self, state: ETLState) -> Dict[str, Any]:
        """
        Extract and analyze code from Code Generation Agent
        
        Args:
            state: Current ETL state containing CGA results
            
        Returns:
            Dictionary with code analysis and extracted information
        """
        try:
            coder_definition = state.get("coder_definition", {})
            code = coder_definition.get("code", "")
            
            if not code:
                return {
                    "has_code": False,
                    "error": "No code found from CGA"
                }
            
            # Analyze the code to extract training-relevant information
            code_analysis = self._analyze_code(code)
            
            return {
                "has_code": True,
                "code": code,
                "analysis": code_analysis,
                "code_type": self._detect_code_type(code),
                "training_applicable": self._is_training_applicable(code)
            }
            
        except Exception as e:
            logger.error(f"Error extracting CGA code: {e}")
            return {
                "has_code": False,
                "error": f"Failed to extract CGA code: {str(e)}"
            }
    
    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code to extract training-relevant information"""
        analysis = {
            "has_imports": False,
            "ml_libraries": [],
            "data_operations": [],
            "complexity": "simple"
        }
        
        # Check for ML library imports
        ml_libs = ["sklearn", "torch", "tensorflow", "xgboost", "lightgbm", "pandas", "numpy"]
        for lib in ml_libs:
            if lib in code:
                analysis["ml_libraries"].append(lib)
                analysis["has_imports"] = True
        
        # Check for data operations
        data_ops = ["read_csv", "DataFrame", "fit", "train", "predict", "transform"]
        for op in data_ops:
            if op in code:
                analysis["data_operations"].append(op)
        
        # Assess complexity
        lines = code.split('\n')
        if len(lines) > 50:
            analysis["complexity"] = "complex"
        elif len(lines) > 20:
            analysis["complexity"] = "medium"
        
        return analysis
    
    def _detect_code_type(self, code: str) -> str:
        """Detect the type of code (ETL, ML, analysis, etc.)"""
        if any(keyword in code.lower() for keyword in ["train", "fit", "model", "predict"]):
            return "machine_learning"
        elif any(keyword in code.lower() for keyword in ["read_csv", "transform", "groupby", "merge"]):
            return "data_processing"
        elif any(keyword in code.lower() for keyword in ["plot", "matplotlib", "seaborn", "visualization"]):
            return "analysis"
        else:
            return "general"
    
    def _is_training_applicable(self, code: str) -> bool:
        """Determine if the code is applicable for training models"""
        training_indicators = [
            "predict", "classification", "regression", "model", "target", "label",
            "feature", "train", "test", "validation", "accuracy", "score"
        ]
        
        return any(indicator in code.lower() for indicator in training_indicators)
    
    def _find_ml_indicators(self, code: str) -> List[str]:
        """Find ML-related indicators in code"""
        ml_patterns = [
            r'\b(train|fit|predict|model|classify|regress)\b',
            r'\b(sklearn|torch|tensorflow|xgboost|lightgbm)\b',
            r'\b(accuracy|precision|recall|f1|mse|mae|r2)\b',
            r'\b(feature|target|label|X|y)\b',
            r'\b(cross_val|validation|test_split)\b'
        ]
        
        indicators = []
        for pattern in ml_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            indicators.extend(matches)
        
        return list(set(indicators))
    
    def _analyze_data_flow(self, code: str) -> Dict[str, Any]:
        """Analyze data flow in the code"""
        data_flow = {
            "has_data_loading": bool(re.search(r'\b(read_csv|read_|load|pandas|pd\.)\b', code, re.IGNORECASE)),
            "has_data_processing": bool(re.search(r'\b(groupby|merge|join|transform|apply)\b', code, re.IGNORECASE)),
            "has_feature_engineering": bool(re.search(r'\b(create|engineer|feature|transform|scale|encode)\b', code, re.IGNORECASE)),
            "has_data_splitting": bool(re.search(r'\b(train_test_split|split|train|test|validation)\b', code, re.IGNORECASE)),
            "has_output_saving": bool(re.search(r'\b(to_csv|save|write|export)\b', code, re.IGNORECASE))
        }
        
        return data_flow
    
    def _suggest_models_from_code(self, code: str, analysis: Dict[str, Any]) -> List[str]:
        """Suggest appropriate models based on code analysis"""
        suggestions = []
        
        # Check for classification indicators
        if any(word in code.lower() for word in ["classify", "category", "class", "label"]):
            suggestions.extend(["pytorch_classification", "sklearn_classification"])
        
        # Check for regression indicators
        if any(word in code.lower() for word in ["predict", "regress", "value", "amount", "price"]):
            suggestions.extend(["pytorch_regression", "sklearn_regression"])
        
        # Check for time series indicators
        if any(word in code.lower() for word in ["time", "date", "timestamp", "sequence"]):
            suggestions.append("pytorch_lstm")
        
        # Check for clustering indicators
        if any(word in code.lower() for word in ["cluster", "group", "segment"]):
            suggestions.append("sklearn_clustering")
        
        return list(set(suggestions))
    
    def _identify_preprocessing_requirements(self, code: str) -> List[str]:
        """Identify preprocessing requirements from code"""
        requirements = []
        
        if re.search(r'\b(missing|nan|null|fillna)\b', code, re.IGNORECASE):
            requirements.append("handle_missing_values")
        
        if re.search(r'\b(scale|normalize|standardize)\b', code, re.IGNORECASE):
            requirements.append("feature_scaling")
        
        if re.search(r'\b(encode|categorical|dummy|onehot)\b', code, re.IGNORECASE):
            requirements.append("categorical_encoding")
        
        if re.search(r'\b(select|feature|importance|correlation)\b', code, re.IGNORECASE):
            requirements.append("feature_selection")
        
        return requirements
    
    def _find_feature_engineering_opportunities(self, code: str) -> List[str]:
        """Find feature engineering opportunities in code"""
        opportunities = []
        
        if re.search(r'\b(age|date|time)\b', code, re.IGNORECASE):
            opportunities.append("temporal_features")
        
        if re.search(r'\b(price|amount|value|count)\b', code, re.IGNORECASE):
            opportunities.append("numerical_features")
        
        if re.search(r'\b(category|type|class|group)\b', code, re.IGNORECASE):
            opportunities.append("categorical_features")
        
        if re.search(r'\b(interaction|product|ratio|difference)\b', code, re.IGNORECASE):
            opportunities.append("interaction_features")
        
        return opportunities
    
    def _assess_training_complexity(self, code: str, analysis: Dict[str, Any]) -> str:
        """Assess training complexity based on code and analysis"""
        complexity_score = 0
        
        # Count lines of code
        lines = code.split('\n')
        if len(lines) > 100:
            complexity_score += 2
        elif len(lines) > 50:
            complexity_score += 1
        
        # Check for complex operations
        complex_ops = ["groupby", "merge", "apply", "transform", "pivot"]
        for op in complex_ops:
            if op in code.lower():
                complexity_score += 1
        
        # Check for ML libraries
        ml_libs = analysis.get("ml_libraries", [])
        complexity_score += len(ml_libs)
        
        # Determine complexity level
        if complexity_score >= 5:
            return "complex"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "simple"


