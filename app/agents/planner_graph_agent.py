from __future__ import annotations
import os
import json
import re
import uuid
import logging
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from typing_extensions import TypedDict
from pathlib import Path

from graphviz import Digraph

from app.graph.etl_state import ETLState

logger = logging.getLogger(__name__)

class PlannerGraphConfig:
    """Configuration for planner graph styling and output"""
    
    NODE_COLORS = {
        "data_input": "#90EE90",      # Light green
        "transformation": "#87CEEB",  # Sky blue
        "data_output": "#FFA500",     # Orange
        "decision": "#FFD700",        # Gold
        "ml_model": "#DDA0DD",        # Plum
        "analysis": "#F0E68C",        # Khaki
        "validation": "#FF6347",      # Tomato
        "aggregation": "#98FB98",      # Pale green
        "filter": "#87CEFA",          # Light sky blue
        "cleaning": "#DEB887"         # Burlywood
    }

    NODE_SHAPES = {
        "data_input": "box",
        "transformation": "ellipse",
        "data_output": "box",
        "decision": "diamond",
        "ml_model": "hexagon",
        "analysis": "octagon",
        "validation": "trapezium",
        "aggregation": "ellipse",
        "filter": "ellipse",
        "cleaning": "ellipse"
    }

    EDGE_STYLES = {
        "data_flow": "solid",
        "control_flow": "dashed",
        "error_flow": "dotted"
    }

    OUTPUT_DIR = os.getenv("PLANNER_GRAPH_OUTPUT_DIR", "/tmp/planner_graphs/")
    MAX_FILES = 100
    IMAGE_FORMAT = "PNG"

_PLAN_STEP_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)\s*$")


def _planner_steps_from_plan_text(plan_text: str) -> List[StepNode]:
    """Parse a free-text (numbered or bulleted) plan into StepNodes.

    planner.py's generate_code branch writes a prose plan into state["plan"]
    and never populates state["planner_definition"], so on a normal analysis
    turn this is the only structured source the graph can be built from.
    A leading assumption sentence has no number/bullet, so it's skipped here.
    """
    if not isinstance(plan_text, str) or not plan_text.strip():
        return []
    steps: List[StepNode] = []
    step_id = 1
    for line in plan_text.splitlines():
        m = _PLAN_STEP_RE.match(line)
        if not m:
            continue
        text = m.group(1).strip()
        if not text:
            continue
        name = text if len(text) <= 60 else text[:57].rstrip() + "..."
        steps.append({
            "step_id": step_id,
            "name": name,
            "type": _infer_step_type(text),
            "description": text,
            "inputs": [],
            "outputs": [],
            "sql_query": None,
            "parameters": None,
        })
        step_id += 1
    return steps


def _graph_plan_from_plan_text(plan_text: str, state: ETLState) -> Optional[PlannerGraphPlan]:
    """Build a PlannerGraphPlan directly from prose plan text.

    Bypasses _convert_plan_to_graph_format (which needs the structured
    source/transformations/schedule dict and would only emit the fixed
    Load->Generate->Validate->Execute->Visualize scaffold). Gated at >= 2
    steps so a trivial one-liner doesn't produce a single-node graph.
    """
    steps = _planner_steps_from_plan_text(plan_text)
    if len(steps) < 2:
        return None
    connections: List[PlanConnection] = [
        {"from_step": s["step_id"], "to_step": s["step_id"] + 1}
        for s in steps[:-1]
    ]
    return {
        "plan_id": f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "title": "Data Processing Plan",
        "steps": steps,
        "connections": connections,
        "metadata": {"created_at": datetime.now().isoformat(), "source": "plan_text"},
    }

class StepNode(TypedDict):
    """Type definition for a workflow step"""
    step_id: int
    name: str
    type: str
    description: str
    inputs: List[str]
    outputs: List[str]
    sql_query: Optional[str]
    parameters: Optional[Dict]


class PlanConnection(TypedDict):
    """Type definition for step connections"""
    from_step: int
    to_step: int
    condition: Optional[str]
    edge_type: Optional[str]


class PlannerGraphPlan(TypedDict):
    """Type definition for planner graph plan"""
    plan_id: str
    title: str
    steps: List[StepNode]
    connections: List[PlanConnection]
    metadata: Optional[Dict]


class PlannerGraphAgent:
    """Agent responsible for generating visual representations of data processing workflows"""

    def __init__(self, config: PlannerGraphConfig = None):
        self.config = config or PlannerGraphConfig()
        # Re-resolve at construction time: the class attribute is frozen at import,
        # which can precede load_dotenv(). Path(None) here was the source of the
        # "expected str, bytes or os.PathLike object, not NoneType" graph status.
        output_dir = (
            self.config.OUTPUT_DIR
            or os.getenv("PLANNER_GRAPH_OUTPUT_DIR")
            or os.path.join(tempfile.gettempdir(), "planner_graphs")
        )
        self.output_dir = Path(output_dir)
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {str(e)}")
            raise

    def generate_planner_graph(self, plan_data: PlannerGraphPlan) -> Dict[str, str]:
        """Generate a planner graph from plan data"""
        
        if Digraph is None:
            error_msg = "Graphviz Python package not available - planner graph cannot be generated"
            logger.warning(error_msg)
            return {
                "plan_id": plan_data.get("plan_id", "unknown"),
                "status": f"error: {error_msg}"
            }
        
        # Test if Graphviz executables are available
        try:
            test_graph = Digraph()
            test_graph.node('test', 'Test')
            # Try to render to test if executables are available
            test_graph.pipe(format='png', quiet=True)
        except Exception as e:
            if "failed to execute" in str(e) or "dot" in str(e):
                error_msg = "Graphviz system executables not found. Please install Graphviz system package."
                logger.warning(f"{error_msg} Original error: {str(e)}")
                return {
                    "plan_id": plan_data.get("plan_id", "unknown"),
                    "status": f"error: {error_msg}"
                }
            else:
                logger.warning(f"Graphviz test failed: {str(e)}")
                # Continue with generation attempt
        
        try:
            # Validate input
            self._validate_plan_data(plan_data)

            # Create Graphviz graph
            graph = self._create_graphviz_graph(plan_data)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plan_{plan_data['plan_id']}_{timestamp}"

            # Render and save
            file_path = self._render_and_save(graph, filename)

            # Clean up old files
            self._cleanup_old_files()

            result = {
                "planner_graph_path": str(file_path),
                "plan_id": plan_data["plan_id"],
                "timestamp": timestamp,
                "status": "success"
            }

            logger.info(f"Planner graph generated successfully for plan {plan_data['plan_id']}")
            return result

        except ValueError as e:
            error_msg = f"Invalid plan data: {str(e)}"
            logger.error(error_msg)
            return {
                "plan_id": plan_data.get("plan_id", "unknown"),
                "status": f"error: {error_msg}"
            }
        except FileNotFoundError as e:
            error_msg = f"File system error: {str(e)}"
            logger.error(error_msg)
            return {
                "plan_id": plan_data.get("plan_id", "unknown"),
                "status": f"error: {error_msg}"
            }
        except PermissionError as e:
            error_msg = f"Permission denied: {str(e)}"
            logger.error(error_msg)
            return {
                "plan_id": plan_data.get("plan_id", "unknown"),
                "status": f"error: {error_msg}"
            }
        except Exception as e:
            error_msg = f"Unexpected error generating planner graph: {str(e)}"
            logger.error(error_msg)
            return {
                "plan_id": plan_data.get("plan_id", "unknown"),
                "status": f"error: {error_msg}"
            }

    def _validate_plan_data(self, plan_data: PlannerGraphPlan) -> None:
        """Validate the input plan data structure"""
        if not isinstance(plan_data, dict):
            raise ValueError("Plan data must be a dictionary")
            
        required_fields = ["plan_id", "steps"]
        
        for field in required_fields:
            if field not in plan_data:
                raise ValueError(f"Missing required field: {field}")
        
        if not plan_data["steps"]:
            raise ValueError("Plan must contain at least one step")

        # Validate step structure
        for i, step in enumerate(plan_data["steps"]):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} must be a dictionary")
                
            required_step_fields = ["step_id", "name", "type"]
            for field in required_step_fields:
                if field not in step:
                    raise ValueError(f"Step {i} missing required field: {field}")
                    
            # Validate step_id is a number
            if not isinstance(step["step_id"], (int, float)):
                raise ValueError(f"Step {i} step_id must be a number")

    def _create_graphviz_graph(self, plan_data: PlannerGraphPlan) -> Digraph:
        """Create a Graphviz graph from plan data"""
        
        try:
            # Initialize graph
            graph = Digraph(comment=f"Workflow Plan: {plan_data['plan_id']}")
            graph.attr(rankdir='LR', size='12,8', dpi='300')
            graph.attr('node', fontname='Arial', fontsize='10')
            graph.attr('edge', fontname='Arial', fontsize='9')

            # Add title
            title = plan_data.get("title", f"Data Processing Plan: {plan_data['plan_id']}")
            graph.attr(
                label=f"{title}\n\n", labelloc='t', labeljust='c', fontsize='16', fontname='Arial Bold'
            )

            # Add nodes for each step
            for step in plan_data["steps"]:
                self._add_step_node(graph, step)

            # Add connections
            connections = plan_data.get("connections", [])
            if connections:
                self._add_connections(graph, connections)
            else:
                # If no explicit connections, create linear flow
                self._add_linear_connections(graph, plan_data["steps"])

            return graph
            
        except Exception as e:
            logger.error(f"Error creating Graphviz graph: {str(e)}")
            raise

    def _add_step_node(self, graph: Digraph, step: StepNode) -> None:
        """Add a single step node to the graph"""
        
        try:
            step_id = str(step["step_id"])
            step_name = step["name"]
            step_type = step["type"]
            description = step.get("description", "")

            # Get styling based on step type
            color = self.config.NODE_COLORS.get(step_type, "#E6E6FA")
            shape = self.config.NODE_SHAPES.get(step_type, "box")

            # Create node label
            label = f"{step_name}"
            if description:
                # Truncate long descriptions
                desc_display = description[:80] + "..." if len(description) > 80 else description
                label += f"\\n{desc_display}"

            # Add SQL query if present
            if step.get("sql_query"):
                sql_preview = step["sql_query"][:50] + "..." if len(step["sql_query"]) > 50 else step["sql_query"]
                label += f"\\n[SQL: {sql_preview}]"

            # Add inputs/outputs info
            inputs = step.get("inputs", [])
            outputs = step.get("outputs", [])

            if inputs:
                inputs_str = ', '.join(inputs[:3])  # Limit to first 3 items
                if len(inputs) > 3:
                    inputs_str += f" +{len(inputs)-3} more"
                label += f"\\nInputs: {inputs_str}"
                
            if outputs:
                outputs_str = ', '.join(outputs[:3])  # Limit to first 3 items
                if len(outputs) > 3:
                    outputs_str += f" +{len(outputs)-3} more"
                label += f"\\nOutputs: {outputs_str}"

            # Add node to graph
            graph.node(
                step_id,
                label,
                shape=shape,
                fillcolor=color,
                style='filled,rounded',
                fontname='Arial',
                fontsize='9'
            )
            
        except KeyError as e:
            logger.error(f"Missing required step field: {str(e)}")
            raise ValueError(f"Step missing required field: {str(e)}")
        except Exception as e:
            logger.error(f"Error adding step node: {str(e)}")
            raise

    def _add_connections(self, graph: Digraph, connections: List[PlanConnection]) -> None:
        """Add connections between steps"""
        
        try:
            for conn in connections:
                if not isinstance(conn, dict):
                    logger.warning(f"Skipping invalid connection: {conn}")
                    continue
                    
                if "from_step" not in conn or "to_step" not in conn:
                    logger.warning(f"Skipping connection missing required fields: {conn}")
                    continue
                    
                from_id = str(conn["from_step"])
                to_id = str(conn["to_step"])
                condition = conn.get("condition")
                edge_type = conn.get("edge_type", "data_flow")

                # Get edge style
                style = self.config.EDGE_STYLES.get(edge_type, "solid")

                # Create edge label
                label = condition if condition else ""
                
                headport = 'ne' if from_id > to_id else None
                tailport = 'nw' if from_id > to_id else None

                # Add edge
                graph.edge(
                    from_id,
                    to_id,
                    label=label,
                    style=style,
                    fontsize='8',
                    headport=headport,
                    tailport=tailport
                )
        except Exception as e:
            logger.error(f"Error adding connections: {str(e)}")
            raise

    def _add_linear_connections(self, graph: Digraph, steps: List[StepNode]) -> None:
        """Add linear connections between steps"""
        
        try:
            for i in range(len(steps) - 1):
                from_id = str(steps[i]["step_id"])
                to_id = str(steps[i + 1]["step_id"])
                
                graph.edge(from_id, to_id, style='solid')
        except Exception as e:
            logger.error(f"Error adding linear connections: {str(e)}")
            raise

    def _render_and_save(self, graph: Digraph, filename: str) -> Path:
        """Render the graph and save as PNG"""
        
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")
            
        try:
            output_path = self.output_dir / filename
            
            # Render the graph
            graph.render(
                str(output_path),
                format=self.config.IMAGE_FORMAT.lower(),
                cleanup=True
            )
            
            # Return path to generated file
            generated_file = output_path.with_suffix(f".{self.config.IMAGE_FORMAT.lower()}")
            
            # Verify file was created
            if not generated_file.exists():
                raise FileNotFoundError(f"Generated file not found: {generated_file}")
                
            return generated_file
            
        except Exception as e:
            logger.error(f"Error rendering graph: {str(e)}")
            raise

    def _cleanup_old_files(self) -> None:
        """Clean up old planner graph files"""
        
        try:
            if not self.output_dir.exists():
                return
                
            files = list(self.output_dir.glob("*.png"))
            
            if len(files) > self.config.MAX_FILES:
                # Sort by creation time and remove oldest
                files.sort(key=lambda x: x.stat().st_ctime)
                files_to_remove = files[:-self.config.MAX_FILES]
                
                removed_count = 0
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove file {file_path}: {str(e)}")
                        
                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} old planner graph files")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def planner_graph_agent_node(state: ETLState) -> ETLState:
    """
    LangGraph node function for planner graph agent
    
    Args:
        state: Current workflow state containing plan data
    
    Returns:
        Updated state with planner graph information
    """
    
    try:
        # Validate state
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")
            
        # Prefer a structured planner_definition when present; otherwise fall
        # back to the prose plan in state["plan"] (the generate_code branch
        # writes that and leaves planner_definition empty on normal turns).
        plan_data = state.get("planner_definition", {})
        if plan_data:
            graph_plan = _convert_plan_to_graph_format(plan_data, state)
        else:
            graph_plan = _graph_plan_from_plan_text(state.get("plan"), state)

        if not graph_plan or not graph_plan.get("steps"):
            logger.warning(
                "No planner definition or usable plan text - skipping planner graph"
            )
            updated_state = state.copy()
            updated_state["planner_graph_status"] = "skipped: no planner definition"
            return updated_state

        # Create planner graph agent
        graph_agent = PlannerGraphAgent()
        
        # Generate planner graph
        result = graph_agent.generate_planner_graph(graph_plan)
        
        # Update state
        updated_state = state.copy()
        updated_state["planner_graph_path"] = result.get("planner_graph_path", "")
        updated_state["planner_graph_status"] = result.get("status", "error: unknown status")
        
        if result.get("status", "").startswith("error:"):
            logger.warning(f"Planner graph failed: {result['status']}")
        else:
            logger.info("Planner graph generated successfully")
        
        return updated_state
        
    except ValueError as e:
        logger.error(f"Validation error in planner graph agent: {str(e)}")
        updated_state = state.copy() if isinstance(state, dict) else {}
        updated_state["planner_graph_status"] = f"error: validation failed - {str(e)}"
        return updated_state
    except Exception as e:
        logger.error(f"Unexpected error in planner graph agent: {str(e)}")
        updated_state = state.copy() if isinstance(state, dict) else {}
        updated_state["planner_graph_status"] = f"error: unexpected error - {str(e)}"
        return updated_state


def _convert_plan_to_graph_format(plan_data: Dict, state: ETLState) -> PlannerGraphPlan:
    """Convert ETL plan data to planner graph format"""
    
    try:
        if not isinstance(plan_data, dict):
            raise ValueError("Plan data must be a dictionary")
            
        # Generate a unique plan ID
        plan_id = f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create planner graph plan
        graph_plan: PlannerGraphPlan = {
            "plan_id": plan_id,
            "title": f"ETL Workflow - {plan_data.get('job_name', 'Unnamed Job')}",
            "steps": [],
            "connections": [],
            "metadata": {
                "plan_data": plan_data,
                "created_at": datetime.now().isoformat()
            }
        }
        
        step_counter = 1
        
        # Add data input step
        source = plan_data.get("source")
        if source:
            if "s3" in state["data_source_location"]:
                name = "Load Input from S3 Bucket"
                description = "Load data from Amazon Cloud storage"
            elif "gc" in state["data_source_location"]:
                name = "Load Input from GCS"
                description = "Load data from Google Cloud Storage"
            else:
                name = "Load Input from CSV"
                description = f"Load data from {source}"
            input_step: StepNode = {
                "step_id": step_counter,
                "name": name,
                "type": "data_input",
                "description": description,
                "inputs": [],
                "outputs": [],
                "sql_query": None,
                "parameters": None
            }
            graph_plan["steps"].append(input_step)
            graph_plan["connections"].append({
                "from_step": step_counter,
                "to_step": step_counter + 1
            })
            step_counter += 1

        input_step: StepNode = {
            "step_id": step_counter,
            "name": "Generate Code",
            "type": "code_generation",
            "description": "Generate code for execution",
            "inputs": [],
            "outputs": [],
            "sql_query": None,
            "parameters": None
        }
        graph_plan["steps"].append(input_step)
        graph_plan["connections"].append({
            "from_step": step_counter,
            "to_step": step_counter + 1
        })
        step_counter += 1
    
        # Add transformation steps
        transformations = plan_data.get("transformations", [])
        if isinstance(transformations, list):
            for i, transform in enumerate(transformations):
                if not isinstance(transform, dict):
                    logger.warning(f"Skipping invalid transformation at index {i}")
                    continue
                    
                transform_step: StepNode = {
                    "step_id": step_counter,
                    "name": transform.get("name", f"Transformation {i+1}"),
                    "type": transform.get("type", "transformation"),
                    "description": f"Apply Transformation {i+1}",
                    "inputs": [],
                    "outputs": [],
                    "sql_query": None,
                    "parameters": {
                        "columns": transform.get("columns", []),
                        "aggregation": transform.get("aggregation")
                    }
                }
                graph_plan["steps"].append(transform_step)
                graph_plan["connections"].append({
                    "from_step": step_counter,
                    "to_step": step_counter + 1
                })
                step_counter += 1

        input_step: StepNode = {
            "step_id": step_counter,
            "name": "Validate Code",
            "type": "validate_code",
            "description": "Run quality check on generated script",
            "inputs": [],
            "outputs": [],
            "sql_query": None,
            "parameters": None,
        }
        graph_plan["steps"].append(input_step)
        graph_plan["connections"].append({
            "from_step": step_counter,
            "to_step": 2,
            "condition": "Failed to validate",
            "edge_type": "error_flow"
        })
        graph_plan["connections"].append({
            "from_step": step_counter,
            "to_step": step_counter + 1,
            "condition": "Execute Now"
        })

        step_counter += 1

        # Add schedule step
        schedule = plan_data.get("schedule", "")
        if schedule:
            graph_plan["connections"][-1]["condition"] = "Execute Later"

            transform_step: StepNode = {
                "step_id": step_counter,
                "name": "Schedule Task",
                "type": "schedule_task",
                "description": f"Execute the script {schedule.lower()}",
                "inputs": [],
                "outputs": [],
                "sql_query": None,
                "parameters": None
            }
            graph_plan["steps"].append(transform_step)
            graph_plan["connections"].append({
                "from_step": step_counter,
                "to_step": step_counter + 1,
            })
            step_counter += 1

        input_step: StepNode = {
            "step_id": step_counter,
            "name": "Execute Code",
            "type": "code_exection",
            "description": "Run the ETL script",
            "inputs": [],
            "outputs": [],
            "sql_query": None,
            "parameters": None,
        }
        graph_plan["steps"].append(input_step)
        graph_plan["connections"].append({
            "from_step": step_counter,
            "to_step": step_counter + 1,
            "condition": "If successful"
        })
        step_counter += 1

        input_step: StepNode = {
            "step_id": step_counter,
            "name": "Visualize Output",
            "type": "visualize_output",
            "description": "Generate a graph from the result",
            "inputs": [],
            "outputs": [],
            "sql_query": None,
            "parameters": None
        }
        graph_plan["steps"].append(input_step)
    
        return graph_plan
        
    except Exception as e:
        logger.error(f"Error converting plan to planner graph format: {str(e)}")
        raise


def _infer_step_type(step_description: str) -> str:
    """Infer step type from description"""
    
    if not isinstance(step_description, str):
        return "transformation"
        
    desc_lower = step_description.lower()
    
    if any(keyword in desc_lower for keyword in ["load", "read", "import", "input"]):
        return "data_input"
    elif any(keyword in desc_lower for keyword in ["save", "write", "export", "output"]):
        return "data_output"
    elif any(keyword in desc_lower for keyword in ["model", "train", "predict", "ml", "machine learning"]):
        return "ml_model"
    elif any(keyword in desc_lower for keyword in ["analyze", "explore", "eda", "statistics"]):
        return "analysis"
    elif any(keyword in desc_lower for keyword in ["validate", "check", "verify"]):
        return "validation"
    elif any(keyword in desc_lower for keyword in ["if", "condition", "decide", "choose"]):
        return "decision"
    elif any(keyword in desc_lower for keyword in ["group", "sum", "count", "avg", "aggregate"]):
        return "aggregation"
    elif any(keyword in desc_lower for keyword in ["filter", "where", "select"]):
        return "filter"
    elif any(keyword in desc_lower for keyword in ["clean", "remove", "fix", "handle"]):
        return "cleaning"
    else:
        return "transformation"