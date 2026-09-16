# state.py
from typing import Optional, Dict, List
from typing_extensions import TypedDict

class CodingAgentState(TypedDict):
    """Represents the state passed between agent nodes."""
    # --- Inputs from User & Planner ---
    user_prompt: str
    plan: str
    data_source_location: str
    output_location: str
    schema: Dict[str, str]
    input_data_type: str
    sample_data: Optional[str]
    requirements: Optional[str]
    uploaded_csv_preview: Optional[List[List[str]]]

    # Multi-dataset / fidelity / memory inputs. Must be declared here:
    # LangGraph drops any key that is not a state channel of the subgraph.
    multi_dataset_state: Optional[List[Dict]]
    datasets_context: Optional[List[Dict]]
    analysis_fidelity: Optional[str]
    selected_sample_name: Optional[str]
    memory_hints: Optional[List[str]]
    session_logic_signature: Optional[str]
    messages: Optional[List]

    # --- Artifacts & Feedback Loop ---
    generated_code: Optional[str]
    code_validation_feedback: Optional[str] # Feedback for the coder
    logical_review_feedback: Optional[str]  # Soft (non-blocking) reviewer notes
    coder_blocked_reason: Optional[str]     # Guard refusal to surface as the chat reply
    syntax_error: Optional[bool]
    static_semantic_error: Optional[bool]
    logical_semantic_error: Optional[bool]
    execution_stdout: Optional[str]
    execution_stderr: Optional[str]
    execution_error: Optional[str]
    execution_output_data: Optional[object]
    execution_output_preview: Optional[object]
    retry_count: Optional[int]
    llm_raw_response: Optional[str]
    primary_llm_response: Optional[str]
    coder_pseudocode: Optional[str]
