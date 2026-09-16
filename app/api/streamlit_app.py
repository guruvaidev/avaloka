import logging
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import csv
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.utils import generate_filename_timestamp
import pandas as pd
import streamlit as st
import tempfile
import base64
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from app.agents.sampling_agent import sample_data_from_source, DEFAULT_SAMPLE_MAX_ROWS
from app.api.workflow import build_graph, create_local_execution_state, create_cloud_execution_state
from file_handler.utils import is_delta

# Load environment variables from the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

TINY_SAMPLE_ROWS = int(os.getenv("TINY_SAMPLE_ROWS", "5"))

# Configure GCP MLflow if available
gcp_tracking_uri = os.getenv("MLFLOW_BACKEND_STORE_URI")
gcp_artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")

if gcp_tracking_uri and gcp_artifact_root:
    # Use GCP Cloud SQL and GCS
    os.environ['MLFLOW_TRACKING_URI'] = gcp_tracking_uri
    os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = gcp_artifact_root

    # Force MLflow to use GCS for artifacts
    os.environ['MLFLOW_DISABLE_MLFLOW'] = 'false'

    # Import and configure MLflow globally
    import mlflow
    mlflow.set_tracking_uri(gcp_tracking_uri)

    print(f"🔗 Streamlit: Using GCP MLflow - {gcp_tracking_uri}")
    print(f"🔗 Streamlit: Artifact root - {gcp_artifact_root}")
    print(f"🔗 Streamlit: MLflow tracking URI set globally")
else:
    print("📁 Streamlit: Using local MLflow storage (GCP not configured)")
# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API Key Checks
if "GROQ_API_KEY_PLANNING_AGENT" not in os.environ:
    st.error("GROQ_API_KEY_PLANNING_AGENT not set.")
    st.stop()
if "GROQ_API_KEY_CODING_AGENT" not in os.environ:
    st.error("GROQ_API_KEY_CODING_AGENT not set.")
    st.stop()


def _write_rows_to_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


# LangGraph State
# Pass checkpointer to build_graph() so subgraphs inherit it for state persistence
checkpointer = MemorySaver()
graph = build_graph(checkpointer=checkpointer).compile(checkpointer=checkpointer)

# Streamlit UI
st.set_page_config(page_title="ETL Planner AI", layout="wide")
st.title("💡 ETL Job Planner AI")

# Execution Mode Selection
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    Describe your ETL job — the source, the transformation, and where it should go.

    Say **"good to go"** or **"summarize it"** to generate a final JSON summary.
    """)
with col2:
    execution_mode = st.selectbox(
        "Execution Mode",
        ["local", "cloud"],
        help="Local: Run on your machine. Cloud: Provision infrastructure and run on cloud."
    )
    st.info(f"Mode: **{execution_mode.upper()}**")

file_ext = None
mode = st.radio("What would you like to upload?", ["Single file", "Folder"])
if mode == "Single file":
    uploaded_file = st.file_uploader("📁 Upload input file (optional)", type=["csv", "xml", "json", "parquet", "avro"])
    if uploaded_file:
        name = f"input_data_{generate_filename_timestamp()}_"
        file_ext = uploaded_file.name.split(".")[-1].lower()

        tmp_path = Path(
            tempfile.mkstemp(suffix="." + file_ext, prefix=name)[1]
        )

        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

elif mode == "Folder":
    uploaded_files = st.file_uploader("📁 Upload Directory", accept_multiple_files="directory")
    if uploaded_files:
        if any(is_delta(f.name) for f in uploaded_files):
            file_ext = "delta"
            name = f"input_data_{generate_filename_timestamp()}_"
            tmp_path = Path(tempfile.mkdtemp(prefix=name))

            for f in uploaded_files:
                file_name = os.path.join(tmp_path, *f.name.split(os.path.sep)[1:])
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, "wb") as file:
                    file.write(f.read())

if file_ext:
    st.session_state["uploaded_csv_path"] = str(tmp_path)
    st.session_state["input_data_type"] = file_ext


    result = sample_data_from_source(
        path=str(tmp_path),
        source_type=file_ext,
        stratify_by=None,
        sample_size=DEFAULT_SAMPLE_MAX_ROWS,   # "up to 1000"
        disable_cap_for_fraction=False,        # irrelevant now, but keep false
    )

    if result.get("error"):
        st.error(f"Sampling failed: {result['error']}")
        if result.get("trace"):
            with st.expander("Show traceback"):
                st.code(result["trace"])
    else:
        st.markdown("#### 📊 Sample of uploaded data:")

        # Planner-facing "working sample" (dict rows)
        full_rows = result["rows"] or []
        if full_rows:
            full_df = pd.DataFrame(full_rows)

            # UI preview (unchanged max)
            max_preview = int(os.getenv("STREAMLIT_PREVIEW_MAX_ROWS", "1000"))
            preview_df = full_df.head(max_preview)
            st.dataframe(preview_df, use_container_width=True)
            if len(full_df) > max_preview:
                st.caption(f"Showing first {max_preview:,} of {len(full_df):,} sampled rows.")

            # Keep your existing UI-friendly list-of-lists (if you still use it downstream)
            rows_as_list_of_lists = [list(row.values()) for row in full_rows]
            st.session_state["uploaded_csv_preview"] = rows_as_list_of_lists
            st.session_state["uploaded_csv_columns"] = result["schema"]
            # NEW: store a proper dict list for the planner (bigger context)
            st.session_state["working_sample"] = full_rows

            # NEW: build a tiny 5-row coder sample (dict rows)
            tiny_rows = full_rows[:TINY_SAMPLE_ROWS]
            st.session_state["tiny_sample_preview"] = tiny_rows

            # NEW: write tiny sample CSV on disk for executors
            sample_csv_path = Path(tempfile.mkstemp(suffix="_sample.csv", prefix="tiny_")[1])
            try:
                _write_rows_to_csv(tiny_rows, Path(sample_csv_path))
            except Exception:
                # fall back to the original upload if we can't write tiny file
                sample_csv_path = Path(st.session_state["uploaded_csv_path"])

            # NEW: keep both locations handy
            st.session_state["sample_data_location"] = str(sample_csv_path)           # tiny file (coder)
            st.session_state["full_data_location"] = st.session_state["uploaded_csv_path"]  # original (planner)
        else:
            st.error("Empty dataset provided. Please provide a new dataset.")


else:
    st.error("Please upload a file or folder to proceed.")

st.divider()

# Add the toggle for K8s deployment
deploy_on_k8s = st.toggle("🚀 Deploy on K8s", value=False)
st.session_state["deploy_on_k8s"] = deploy_on_k8s

# Initial State
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hi there! What kind of ETL job are you working on?")
    ]
    st.session_state.planner_definition = {}
    st.session_state.coder_definition = {}
    st.session_state.ready_to_summarize = False
    st.session_state.ready_to_code = False
    st.session_state.dataframe = None
    st.session_state.output_file_data = None
    st.session_state.infrastructure_request = {}
    st.session_state.infrastructure_provisioned = {}
    st.session_state.planner_graph_path = None

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

# Chat input
if prompt := st.chat_input("Describe your ETL job..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

        # Check if user wants ML training based on prompt
        enable_training = any(keyword in prompt.lower() for keyword in [
            "train", "training", "model", "ml", "machine learning",
            "predict", "classification", "regression", "neural network"
        ])

        # Show training status
        if enable_training:
            st.info("🧠 **ML Training Detected** - Training plan will be generated after ETL completion")

    input_state = {
        "messages": st.session_state.messages,
         # Planner gets more context
        "working_sample": st.session_state.get("working_sample", []),        # list[dict]

        # Coder gets tiny context
        "sample_data": st.session_state.get("tiny_sample_preview", []),      # list[dict]

        # Paths for executors
        "sample_data_location": st.session_state.get("sample_data_location"), # tiny CSV
        "full_data_location": st.session_state.get("full_data_location"),     # original upload

        # point to sample while execute_scope='sample'
        "data_source_location": st.session_state.get("sample_data_location"),

        # Hard-stop: never run coder on full from Streamlit
        "execute_scope": "sample",

        "planner_definition": st.session_state.planner_definition,
        "ready_to_summarize": st.session_state.ready_to_summarize,
        "ready_to_code": st.session_state.ready_to_code,
        "coder_definition": st.session_state.coder_definition,
        "data_source_location": st.session_state.get("sample_data_location"),
        "output_location": st.session_state.get("uploaded_csv_path", "").replace(".csv", "_output.csv") if st.session_state.get("uploaded_csv_path") else "",
        "infrastructure_request": st.session_state.infrastructure_request,
        "infrastructure_provisioned": st.session_state.infrastructure_provisioned,
        "execution_result": {},
        "input_data_type": st.session_state.get("input_data_type"),
        "output_file_data": {},
        "enable_training": enable_training,
        "deploy_on_k8s": st.session_state.get("deploy_on_k8s", False),
        "uploaded_csv_preview": st.session_state["uploaded_csv_preview"],
        "uploaded_csv_columns": st.session_state["uploaded_csv_columns"]
    }

    # Configure execution mode
    if execution_mode == "local":
        input_state = create_local_execution_state(input_state)
    else:
        input_state = create_cloud_execution_state(input_state)

        logger.info(f"Input State: {input_state}")

    with st.spinner("Thinking..."):
        try:
            # Check if we already have results for this input
            if (st.session_state.get("last_input") == prompt and
                st.session_state.get("execution_result", {}).get("status") == "success"):
                st.info("🔄 Using cached results from previous execution")
                final_state = st.session_state.get("last_final_state", input_state)
            else:
                # Run workflow with a single invocation
                final_state = graph.invoke(input_state, config={"configurable": {"thread_id": "etl-thread"}})
                st.success("✅ Workflow completed successfully!")

                # Cache the results
                st.session_state.last_input = prompt
                st.session_state.last_final_state = final_state
                st.session_state.execution_result = final_state.get("execution_result", {})
        except Exception as e:
            st.error(f"❌ Workflow failed: {e}")
            st.stop()

    # Show only the final results with deduplication
    if final_state.get("execution_result", {}).get("status") == "success":
        st.success("🎉 ETL Job Completed Successfully!")

        # Deduplicate messages by content
        seen_messages = set()
        unique_messages = []
        for msg in final_state["messages"]:
            if msg.content not in seen_messages:
                seen_messages.add(msg.content)
                unique_messages.append(msg)

        # Show only the most relevant unique messages
        relevant_messages = []
        for msg in unique_messages:
            if any(keyword in msg.content.lower() for keyword in ["json summary", "import pandas", "output saved", "model training agent", "task id:", "mlflow"]):
                relevant_messages.append(msg)

        # Show ETL results first, then training plan
        json_summary_shown = False
        code_shown = False
        result_shown = False
        training_shown = False

        # First pass: Show ETL results (JSON summary, code, results)
        for msg in relevant_messages:
            if "json summary" in msg.content.lower() and not json_summary_shown:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)
                json_summary_shown = True
            elif "import pandas" in msg.content.lower() and not code_shown:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)
                code_shown = True
            elif "output saved" in msg.content.lower() and not result_shown:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)
                result_shown = True

        # Second pass: Show training plan (if available)
        for msg in relevant_messages:
            # Look for actual training plan messages from the Model Training Agent
            if (("model training agent" in msg.content.lower() and "training plan created successfully" in msg.content.lower()) or
                ("task id:" in msg.content.lower() and "goal:" in msg.content.lower() and "strategy:" in msg.content.lower())) and not training_shown:
                with st.chat_message("assistant"):
                    st.markdown("**🧠 Training Plan Generated:**")
                    st.markdown(msg.content)
                training_shown = True

        # If training was requested but no training message shown, show it anyway
        if enable_training and not training_shown:
            # Show the training task details
            if final_state.get("training_task"):
                with st.chat_message("assistant"):
                    st.markdown("**🧠 Training Task Created:**")
                    st.json(final_state["training_task"])
    else:
        # Show the last few unique messages if not complete
        seen_messages = set()
        unique_messages = []
        for msg in final_state["messages"]:
            if msg.content not in seen_messages:
                seen_messages.add(msg.content)
                unique_messages.append(msg)

        last_messages = unique_messages[-3:]  # Show last 3 unique messages
        for msg in last_messages:
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Update session state with unique messages only
    seen_messages = set()
    unique_messages = []
    for msg in final_state["messages"]:
        if msg.content not in seen_messages:
            seen_messages.add(msg.content)
            unique_messages.append(msg)
    st.session_state.messages = unique_messages

    # --- Agentic Memory Diagnostics UI ---
    with st.expander("🧠 Agentic Memory Diagnostics (Traceability)"):
        st.markdown("Trace how historical execution contexts influenced this pipeline generation.")

        # Layer 1 & 4 (Domain & Time-Series Vector contexts)
        memory_hints = final_state.get("memory_hints", [])
        if memory_hints:
            st.write("##### 📡 Layer 1 & 4 Insights (Redis Domain / Milvus Time-Series)")
            for hint in memory_hints:
                st.info(hint)
        else:
             if final_state.get("memory_context_unavailable"):
                 st.error("Circuit Breaker Tripped - Distributed Memory offline.")
             else:
                 st.write("No distinct Domain/Time-Series hints triggered for this prompt.")

        # Layer 2 (Usage Rule formatting)
        session_logic = final_state.get("session_logic_signature")
        if session_logic:
            st.write("##### 👤 Layer 2 Usage Context (ChromaDB Persisted)")
            st.success(session_logic)

        # Layer 3 (Artifact Tracking)
        prior_artifact = final_state.get("prior_artifact_found")
        if prior_artifact:
            st.write("##### 💾 Layer 3 Artifact Redundancy (PostgreSQL)")
            st.warning("Historical Redundancy Check: Exact Matched Execution footprint found & optimized!")


    # Update state
    st.session_state.planner_definition = final_state["planner_definition"]
    st.session_state.ready_to_summarize = final_state["ready_to_summarize"]
    st.session_state.ready_to_code = final_state["ready_to_code"]
    st.session_state.coder_definition = final_state["coder_definition"]
    st.session_state.infrastructure_request = final_state["infrastructure_request"]
    st.session_state.infrastructure_provisioned = final_state["infrastructure_provisioned"]

    # Store file data in session state
    if final_state.get("output_file_data"):
        st.session_state.output_file_data = final_state["output_file_data"]

    if final_state.get("planner_graph_path"):
        st.session_state.planner_graph_path = final_state["planner_graph_path"]

#  Display planner graph if available
if st.session_state.get("planner_graph_path"):
    graph_path = st.session_state.planner_graph_path
    if os.path.exists(graph_path):
        st.divider()
        st.subheader("📊 Workflow Planner Graph")

        # Display the image
        st.image(graph_path, caption="ETL Workflow Diagram", use_container_width=True)

        # Optional: Add download button for the planner graph
        with open(graph_path, "rb") as img_file:
            st.download_button(
                label="⬇️ Download Planner Graph",
                data=img_file.read(),
                file_name=f"workflow_{os.path.basename(graph_path)}",
                mime="image/png"
            )
    else:
        st.warning(f"Planner graph file not found at: {graph_path}")

# Display download button if file data is available in session state
if st.session_state.get("output_file_data"):
    file_data = st.session_state.output_file_data

    # Decode base64 content for download
    try:
        csv_content = base64.b64decode(
            file_data["content"].split("base64,")[1]
        ).decode()

        # Add download button
        st.download_button(
            label=f"⬇️ Download {file_data['filename']}",
            data=csv_content,
            file_name=file_data["filename"],
            mime="text/csv",
            key=f"download_persistent"
        )

        # Show file info
        st.caption(f"File size: {file_data['size']} bytes")

    except Exception as e:
        logger.error(f"Failed to prepare download: {e}")
        st.error(f"Failed to prepare file download: {str(e)}")
