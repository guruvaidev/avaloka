# app/infra/python_app/main.py
from flask import Flask, request, jsonify
import pandas as pd
import io
import base64
import json
import traceback
import os
import tempfile
import shutil
from pathlib import Path
import sys
import numpy as np
import uuid
from datetime import datetime

import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)


@app.route('/groupby_sum', methods=['POST'])
def groupby_sum():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
            group_by_column = request.form.get('group_by_column')
            sum_column = request.form.get('sum_column')

            if not group_by_column or not sum_column:
                return jsonify({"error": "Missing group_by_column or sum_column in form data"}), 400

            if group_by_column not in df.columns or sum_column not in df.columns:
                return jsonify({"error": "One or both specified columns not found in CSV"}), 400

            result = df.groupby(group_by_column)[sum_column].sum().reset_index()
            return jsonify(result.to_dict(orient='records'))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/execute', methods=['POST'])
def execute():
    """
    Execute arbitrary Python code with provided input data.
    
    Expected JSON payload:
    {
        "code": "base64_encoded_python_code",
        "data": "base64_encoded_data",
        "data_format": "csv"  
    }
    """
    app.logger.info("Received request to /execute")
    temp_dir = None
    try:
        # Generate unique execution ID and timestamp for file naming
        execution_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp(prefix=f"execution_{execution_id}_")
        input_file_path = os.path.join(temp_dir, "input.csv")
        output_file_path = os.path.join(temp_dir, "output.csv")
        
        # Parse request JSON
        if not request.is_json:
            app.logger.error("Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.json

        # Required base64-encoded Python code
        code_b64 = data.get("code")
        if not code_b64:
            app.logger.error("Missing 'code' field in JSON payload")
            return jsonify({"error": "Missing 'code' field"}), 400

        try:
            code_str = base64.b64decode(code_b64).decode("utf-8")
            app.logger.info(f"Executing code:\n{code_str}")
        except Exception as e:
            app.logger.error(f"Invalid base64 for code: {e}")
            return jsonify({"error": f"Invalid base64 for code: {e}"}), 400

        # Expose file paths for user code
        os.environ["INPUT_FILE"] = input_file_path
        os.environ["OUTPUT_FILE"] = output_file_path

        # Prepare globals the user code can access
        execution_globals = {
            "pd": pd,
            "np": np,
            "os": os,
            "sys": sys,
            "Path": Path,
            "input_file": input_file_path,
            "output_file": output_file_path,
            "result": None,
        }

        # Optional CSV data
        csv_b64 = data.get("data")
        if csv_b64:
            try:
                csv_str = base64.b64decode(csv_b64).decode("utf-8")
                with open(input_file_path, "w") as f:
                    f.write(csv_str)
                execution_globals["input_df"] = pd.read_csv(io.StringIO(csv_str))
            except Exception as e:
                app.logger.error(f"Invalid CSV data: {e}")
                return jsonify({"error": f"Invalid CSV data: {e}"}), 400
        
        # Redirect stdout and stderr to capture print statements
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buffer, stderr_buffer
        
        # Execute the code
        try:
            app.logger.info("About to execute code string.")
            exec(code_str, execution_globals)
            app.logger.info("Code execution finished.")
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
        except Exception as e:
            error_trace = traceback.format_exc()
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue() + "\n" + error_trace
            app.logger.error(f"Code execution failed: {error_trace}")
            return jsonify({
                "error": f"Code execution failed: {str(e)}", 
                "traceback": error_trace,
                "stdout": stdout,
                "stderr": stderr
            }), 500
        finally:
            # Restore stdout and stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr

        
        # Check for output files
        result = execution_globals.get('result')
        output_content = None
        output_format = None
        
        # If the user created a CSV, drop timestamp column and return its content
        if os.path.exists(output_file_path):
            try:
                df_out = pd.read_csv(output_file_path)
                if "timestamp" in df_out.columns:
                    df_out = df_out.drop(columns=["timestamp"])
                    df_out.to_csv(output_file_path, index=False)

                encoded_output = base64.b64encode(df_out.to_csv(index=False).encode()).decode()
                output_data = f"data:text/csv;base64,{encoded_output}"
            except Exception as e:
                output_data = None
        else:
            output_data = None
        
        # Construct the response
        response = {
            "status": "success",
            "stdout": stdout,
            "stderr": stderr
        }
        
        # Include the result if available
        if result is not None:
            if isinstance(result, pd.DataFrame):
                response["result"] = result.to_dict(orient='records')
            elif hasattr(result, 'tolist'):  # For numpy arrays
                response["result"] = result.tolist()
            else:
                try:
                    response["result"] = result
                except:
                    response["result"] = str(result)
        
        if output_data:
            response["output"] = output_data
            # Add file download information
            response["output_file"] = {
                "filename": f"etl_results_{execution_id}_{timestamp}.csv",
                "content": output_data,
                "size": len(df_out.to_csv(index=False))
            }
            
        return jsonify(response)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}", "traceback": error_trace}), 500
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
