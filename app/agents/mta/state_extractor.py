"""
State Extractor for Model Training Agent

Extracts information from ETL state for training task creation.
Consolidates data from various agents (CGA, Planning, Data).
"""

import logging
from typing import Dict, Any, List, Optional
from app.graph.etl_state import ETLState

logger = logging.getLogger(__name__)


class StateExtractor:
    """
    Extracts and consolidates information from ETL state
    
    Responsibilities:
    - Extract CGA code information
    - Extract data information (schema, samples, characteristics)
    - Extract planning context and business goals
    - Extract ETL context
    - Analyze data characteristics
    """
    
    def __init__(self):
        """Initialize state extractor"""
        pass
    
    def extract_data_info(self, state: ETLState) -> Dict[str, Any]:
        """
        Extract data information from various sources in the state
        
        Args:
            state: Current ETL state containing data information
            
        Returns:
            Dictionary with consolidated data information
        """
        try:
            data_info = {
                "has_data": False,
                "source_location": None,
                "schema": [],
                "sample_data": {},
                "data_characteristics": {}
            }
            
            # Extract data source location
            if state.get("data_source_location"):
                data_info["source_location"] = state["data_source_location"]
                data_info["has_data"] = True
            
            # Extract schema information (from sampling agent or other sources)
            if "schema" in state:
                data_info["schema"] = state["schema"]
            elif "uploaded_csv_columns" in state:
                data_info["schema"] = state["uploaded_csv_columns"]
            
            # Extract sample data
            if "sample_data" in state:
                data_info["sample_data"] = state["sample_data"]
            elif "uploaded_csv_preview" in state:
                data_info["sample_data"] = {"preview": state["uploaded_csv_preview"]}
            
            # Analyze data characteristics
            if data_info["schema"]:
                data_info["data_characteristics"] = self._analyze_data_characteristics(
                    data_info["schema"], 
                    data_info["sample_data"]
                )
            
            return data_info
            
        except Exception as e:
            logger.error(f"Error extracting data info: {e}")
            return {
                "has_data": False,
                "error": f"Failed to extract data info: {str(e)}"
            }
    
    def extract_planning_context(self, state: ETLState) -> Dict[str, Any]:
        """
        Extract planning context from Planning Agent
        
        Args:
            state: Current ETL state containing planning information
            
        Returns:
            Dictionary with planning context and goals
        """
        try:
            planner_definition = state.get("planner_definition", {})
            
            # Extract business goals and requirements
            planning_context = {
                "has_planning": bool(planner_definition),
                "business_goal": self._extract_business_goal(state),
                "requirements": planner_definition,
                "etl_context": self._extract_etl_context(state),
                "training_hints": self._extract_training_hints(planner_definition)
            }
            
            return planning_context
            
        except Exception as e:
            logger.error(f"Error extracting planning context: {e}")
            return {
                "has_planning": False,
                "error": f"Failed to extract planning context: {str(e)}"
            }
    
    def _analyze_data_characteristics(self, schema: List[str], sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data characteristics to inform training decisions"""
        characteristics = {
            "num_features": len(schema) if schema else 0,
            "feature_types": [],
            "has_target": False,
            "problem_type": "unknown",
            "complexity": "unknown"
        }
        
        if schema:
            # Look for common target column names
            target_indicators = ["target", "label", "class", "y", "outcome", "result"]
            characteristics["has_target"] = any(
                indicator in col.lower() for col in schema for indicator in target_indicators
            )
            
            # Determine problem type based on schema
            if any(word in " ".join(schema).lower() for word in ["class", "category", "type"]):
                characteristics["problem_type"] = "classification"
            elif any(word in " ".join(schema).lower() for word in ["price", "amount", "value", "count"]):
                characteristics["problem_type"] = "regression"
            
            # Assess complexity based on number of features
            num_features = len(schema)
            if num_features < 5:
                characteristics["complexity"] = "simple"
            elif num_features < 20:
                characteristics["complexity"] = "medium"
            else:
                characteristics["complexity"] = "complex"
        
        return characteristics
    
    def _extract_business_goal(self, state: ETLState) -> str:
        """Extract business goal from state"""
        # Look in various places for business context
        from langchain_core.messages import HumanMessage
        
        messages = state.get("messages", [])
        for message in reversed(messages):  # Start from most recent
            # Only look at user (HumanMessage) messages, not AI messages
            if not isinstance(message, HumanMessage):
                continue
                
            if hasattr(message, 'content') and message.content:
                content = message.content
                content_lower = content.lower()
                
                # Skip system messages (like "Model Training Agent: ...")
                if content.startswith("Model Training Agent:") or content.startswith("**"):
                    continue
                
                # Check for training-related keywords
                if any(word in content_lower for word in ["predict", "classify", "forecast", "analyze", "train", "model"]):
                    # Extract a clean goal (first 200 chars, but avoid duplication)
                    goal = content[:200].strip()
                    # Remove any training plan keywords to avoid circular references
                    training_plan_keywords = [
                        "create a training plan", "create training plan", "training plan",
                        "plan training", "design training", "training strategy"
                    ]
                    # Check if message contains training plan keywords
                    has_training_plan_keyword = any(keyword in goal.lower() for keyword in training_plan_keywords)
                    
                    if has_training_plan_keyword:
                        # If the message is just a training plan request (short), skip it
                        if len(goal) < 50:  # Short message, likely just a request
                            continue
                        # Otherwise, it's a longer message with business context
                        # Remove training plan keywords and extract the business intent
                        import re
                        cleaned_goal = goal
                        for keyword in training_plan_keywords:
                            cleaned_goal = re.sub(re.escape(keyword), "", cleaned_goal, flags=re.IGNORECASE)
                        cleaned_goal = cleaned_goal.strip()
                        # If we still have meaningful content after removing keywords, use it
                        if len(cleaned_goal) > 10:
                            return cleaned_goal
                        # Otherwise, skip this message
                        continue
                    else:
                        return goal
        
        # Default goal
        return "Analyze and model the provided data"
    
    def _extract_etl_context(self, state: ETLState) -> Dict[str, Any]:
        """Extract ETL context from state"""
        return {
            "has_source": bool(state.get("data_source_location")),
            "has_output": bool(state.get("output_location")),
            "ready_to_code": state.get("ready_to_code", False),
            "ready_to_summarize": state.get("ready_to_summarize", False)
        }
    
    def _extract_training_hints(self, planner_definition: Dict[str, Any]) -> List[str]:
        """Extract hints about training requirements from planning"""
        hints = []
        
        if planner_definition:
            # Look for training-related keywords in planning
            planning_text = str(planner_definition).lower()
            
            if "predict" in planning_text:
                hints.append("prediction_task")
            if "classify" in planning_text:
                hints.append("classification_task")
            if "regression" in planning_text:
                hints.append("regression_task")
            if "time series" in planning_text:
                hints.append("time_series_task")
            if "cluster" in planning_text:
                hints.append("clustering_task")
        
        return hints

    def extract_target_column_from_messages(self, state: ETLState, schema: Optional[List[str]] = None) -> Optional[str]:
        """
        Extract target column name from user messages and validate against schema
        
        Looks for patterns like:
        - "predict 'price'"
        - "target column is 'sales'"
        - "train with 'label' as target"
        - "target='column_name'"
        
        Args:
            state: Current ETL state containing messages
            schema: Optional list of column names to validate extracted column against
            
        Returns:
            Target column name if found and validated, None otherwise
        """
        import re
        from langchain_core.messages import HumanMessage
        
        # Handle schema validation logic
        # According to contract: "Target column name if found and validated, None otherwise"
        # - If schema is None: extract without validation
        # - If schema is provided and has columns: extract AND validate
        # - If schema is provided but empty: extract without validation (schema not available yet)
        schema_columns = None
        if schema is not None:
            # Handle both list and dict schemas
            if isinstance(schema, dict):
                schema_columns = list(schema.keys())
            elif isinstance(schema, list):
                schema_columns = schema
            else:
                schema_columns = []
            
            # If schema is provided but empty, treat as "schema not available yet"
            # Extract without validation rather than abandoning extraction
            if not schema_columns:
                logger.info(
                    "State Extractor: Schema provided but empty - extracting target column without validation "
                    "(schema may not be populated yet in early workflow stages)."
                )
                schema_columns = None  # Treat as no schema for validation purposes
        
        messages = state.get("messages", [])
        
        # Patterns to look for target column specification
        # Include both quoted and unquoted variants for better usability (Bug 2 fix)
        patterns = [
            # Quoted patterns (original)
            r"target\s*[=:]\s*['\"]([^'\"]+)['\"]",  # target='column' or target:"column"
            r"target\s+column\s+['\"]([^'\"]+)['\"]",  # target column 'column'
            r"target\s+column\s+is\s+['\"]([^'\"]+)['\"]",  # target column is 'column'
            r"predict\s+['\"]([^'\"]+)['\"]",  # predict 'column'
            r"predict\s+the\s+['\"]([^'\"]+)['\"]",  # predict the 'column'
            r"with\s+['\"]([^'\"]+)['\"]\s+as\s+target",  # with 'column' as target
            r"target\s+is\s+['\"]([^'\"]+)['\"]",  # target is 'column'
            r"train\s+with\s+['\"]([^'\"]+)['\"]",  # train with 'column'
            r"classify\s+['\"]([^'\"]+)['\"]",  # classify 'column'
            r"forecast\s+['\"]([^'\"]+)['\"]",  # forecast 'column'
            # Unquoted patterns (new - Bug 2 fix)
            # Match column names: word characters (letters, digits, underscore) and hyphens
            # Stop at whitespace, end of string, or common punctuation
            r"target\s*[=:]\s+([\w-]+)(?:\s|$|,|\.|;|'|\")",  # target=column or target:column
            r"target\s+column\s+([\w-]+)(?:\s|$|,|\.|;|'|\")",  # target column column
            r"target\s+column\s+is\s+([\w-]+)(?:\s|$|,|\.|;|'|\")",  # target column is column
            r"predict\s+([\w-]+)(?:\s|$|,|\.|;|'|\")",  # predict column
            r"predict\s+the\s+([\w-]+)(?:\s|$|,|\.|;|'|\")",  # predict the column
            r"with\s+([\w-]+)\s+as\s+target",  # with column as target
            r"target\s+is\s+([\w-]+)(?:\s|$|,|\.|;|'|\")",  # target is column
            r"train\s+with\s+([\w-]+)(?:\s|$|,|\.|;|'|\")",  # train with column
            r"classify\s+([\w-]+)(?:\s|$|,|\.|;|'|\")",  # classify column
            r"forecast\s+([\w-]+)(?:\s|$|,|\.|;|'|\")",  # forecast column
        ]
        
        # Check messages from most recent to oldest
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
                
            content = getattr(message, "content", "") or ""
            if not content:
                continue
            
            # Try each pattern
            validation_failed = False
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    target_col = match.group(1).strip()
                    
                    # Skip if target column is empty after stripping (whitespace-only capture group)
                    if not target_col:
                        continue
                    
                    # Validate that the column exists in schema if schema is provided and has columns
                    # According to docstring: "Target column name if found and validated, None otherwise"
                    # - If schema_columns is None: no validation (extract without validation)
                    # - If schema_columns has items: validate column exists
                    if schema_columns is not None:
                        # Schema has columns - validate column exists (case-insensitive)
                        column_found = any(
                            col.lower() == target_col.lower() 
                            for col in schema_columns
                        )
                        if not column_found:
                            logger.warning(
                                f"State Extractor: Extracted target column '{target_col}' not found in schema. "
                                f"Available columns: {schema_columns[:10]}{'...' if len(schema_columns) > 10 else ''}"
                            )
                            # Mark validation as failed and continue to try next pattern in same message
                            # This allows other patterns to be tried before moving to the next message
                            validation_failed = True
                            continue
                        else:
                            # Validation passed - return the validated column
                            logger.info(f"State Extractor: Extracted and validated target column from message: '{target_col}'")
                            return target_col
                    else:
                        # No validation needed - return the extracted column
                        logger.info(f"State Extractor: Extracted target column from message (no validation): '{target_col}'")
                        return target_col
            
            # If all patterns in this message failed validation, continue to next message
            if validation_failed:
                continue
        
        logger.debug("State Extractor: No target column found in user messages")
        return None


