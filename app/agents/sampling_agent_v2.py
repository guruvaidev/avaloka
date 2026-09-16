import os
import json
import re
import ast
import traceback
from typing import Dict, Any, Optional, Union
from contextlib import asynccontextmanager

import pandas as pd
import pyarrow as pa

from mcp.client.sse import sse_client as MCPClientSSE
from mcp import ClientSession

# --- Schema Generation Logic (PyArrow, Spark-free) ---
# Derive the DDL from an Arrow schema instead of spinning up a local SparkSession
# (no JVM/JRE needed). PyArrow is Ray Data's own engine.

ARROW_TO_SQL_TYPE_MAP = {
    "int8": "SMALLINT", "int16": "SMALLINT", "int32": "INTEGER", "int64": "BIGINT",
    "uint8": "SMALLINT", "uint16": "INTEGER", "uint32": "BIGINT", "uint64": "BIGINT",
    "halffloat": "DECIMAL", "float": "DECIMAL", "double": "DECIMAL",
    "string": "TEXT", "large_string": "TEXT", "bool": "BOOLEAN",
    "binary": "BLOB", "large_binary": "BLOB",
}

def _arrow_type_to_sql(dtype: "pa.DataType") -> str:
    s = str(dtype).lower()
    if s.startswith("timestamp"):
        return "TIMESTAMP"
    if s.startswith("date"):
        return "DATE"
    if s.startswith("decimal"):
        return "DECIMAL"
    if s.startswith(("list", "large_list", "struct", "map")):
        return "TEXT"
    return ARROW_TO_SQL_TYPE_MAP.get(s, "TEXT")

def get_spark_schema_for_df(df: pd.DataFrame) -> "pa.Schema":
    """Generate an Arrow schema from a pandas DataFrame (name kept for callers)."""
    return pa.Table.from_pandas(df, preserve_index=False).schema

def generate_ddl_from_spark_schema(table_name: str, schema: "pa.Schema") -> str:
    """Convert an Arrow schema into a SQL DDL CREATE TABLE statement."""
    cols = [f"    {name} {_arrow_type_to_sql(schema.field(name).type)}" for name in schema.names]
    return f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);"

# --- Database-Specific Query Logic ---

def get_sample_query(base_query: str, sample_size: int, database_type: str) -> str:
    """Generate database-specific sampling query"""
    
    base_query = base_query.rstrip(';')
    
    if database_type in ['postgresql']:
        return f"""
        SELECT * FROM ({base_query}) AS base_query
        ORDER BY RANDOM()
        LIMIT {sample_size}
        """
    
    elif database_type in ['mysql', 'mariadb']:
        return f"""
        SELECT * FROM ({base_query}) AS base_query
        ORDER BY RAND()
        LIMIT {sample_size}
        """
    
    elif database_type == 'sqlite':
        return f"""
        SELECT * FROM ({base_query}) AS base_query
        ORDER BY RANDOM()
        LIMIT {sample_size}
        """
    
    elif database_type == 'mssql':
        return f"""
        SELECT TOP {sample_size} * FROM ({base_query}) AS base_query
        ORDER BY NEWID()
        """
    
    elif database_type == 'oracle':
        return f"""
        SELECT * FROM (
            SELECT * FROM ({base_query}) ORDER BY DBMS_RANDOM.VALUE
        ) WHERE ROWNUM <= {sample_size}
        """
    
    else:
        # Default to PostgreSQL-style
        return f"""
        SELECT * FROM ({base_query}) AS base_query
        ORDER BY RANDOM()
        LIMIT {sample_size}
        """

# --- MCP Connection Manager ---

class MCPConnectionManager:
    """Manages MCP client connections with multi-tenant awareness"""
    
    def __init__(self, mcp_url: str, api_key: str):
        self.mcp_url = mcp_url
        self.api_key = api_key
        self.client = None

    @asynccontextmanager
    async def get_session(self):
        """Context manager for an MCP session with proper cleanup"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with MCPClientSSE(url=self.mcp_url, headers=headers) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    yield session
        except Exception as e:
            print(f"MCP connection error: {e}")
            raise

# --- Core Sampling Logic ---

async def _sample_from_database_mcp(
    query: str,
    sample_size: int,
    mcp_config: Dict[str, str],
    database_type: str = "postgresql"
) -> Dict[str, Any]:
    """
    Connects to an MCP server to sample data from a customer's database.
    Supports multiple database types with appropriate sampling strategies.
    """
    try:
        connection_manager = MCPConnectionManager(
            mcp_url=mcp_config["mcp_url"],
            api_key=mcp_config["api_key"]
        )

        # Generate database-specific sampling query
        sample_query = get_sample_query(query, sample_size, database_type)
        
        print(f"--- SAMPLING-DEBUG: Executing {database_type} query: {sample_query[:100]}... ---")

        async with connection_manager.get_session() as session:
            tools_resp = await session.list_tools()
            tool_names = [t.name for t in tools_resp.tools]
            
            if "query" not in tool_names:
                return {"error": f"MCP server does not support the required 'query' tool. Available: {tool_names}"}

            result = await session.call_tool("query", {"sql": sample_query})

            print(f"--- SAMPLING-DEBUG: Raw MCP result type: {type(result)} ---")
            print(f"--- SAMPLING-DEBUG: Raw MCP result content length: {len(result.content) if result.content else 0} ---")
            
            if not result.content:
                return {"error": "Empty response from MCP server"}
            
            content_item = result.content[0]
            print(f"--- SAMPLING-DEBUG: Content item type: {type(content_item)} ---")
            
            payload = None
            
            # Enhanced parsing logic with better debugging
            try:
                if hasattr(content_item, 'text') and content_item.text:
                    raw_text = content_item.text
                    print(f"--- SAMPLING-DEBUG: Raw text length: {len(raw_text)} ---")
                    print(f"--- SAMPLING-DEBUG: Raw text preview: {raw_text[:200]}... ---")
                    
                    try:
                        payload = json.loads(raw_text)
                        print("--- SAMPLING-DEBUG: Successfully parsed as JSON ---")
                    except json.JSONDecodeError:
                        print("--- SAMPLING-DEBUG: JSON parse failed, trying literal_eval ---")
                        try:
                            clean_text = _clean_server_response_string(raw_text)
                            payload = ast.literal_eval(clean_text)
                            print("--- SAMPLING-DEBUG: Successfully parsed with literal_eval ---")
                        except Exception as e:
                            print(f"--- SAMPLING-DEBUG: literal_eval failed: {e} ---")
                            return {"error": f"Failed to parse MCP response: {e}"}
                
                elif hasattr(content_item, 'json') and callable(content_item.json):
                    try:
                        payload = content_item.json()
                        print("--- SAMPLING-DEBUG: Successfully got JSON from method ---")
                    except Exception as e:
                        print(f"--- SAMPLING-DEBUG: JSON method failed: {e} ---")
                
                elif isinstance(content_item, dict):
                    payload = content_item
                    print("--- SAMPLING-DEBUG: Content item is already a dict ---")
                    
            except Exception as e:
                print(f"--- SAMPLING-DEBUG: All parsing methods failed: {e} ---")
                return {"error": f"Failed to parse MCP server response: {e}"}

            # Handle double-encoded JSON strings
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                    print("--- SAMPLING-DEBUG: Successfully parsed nested JSON string ---")
                except json.JSONDecodeError:
                    return {"error": f"Failed to decode nested JSON payload: {payload[:200]}"}
            
            if not payload:
                return {"error": "No valid response payload from MCP server"}
            
            print(f"--- SAMPLING-DEBUG: Final payload keys: {payload.keys() if isinstance(payload, dict) else 'Not a dict'} ---")
            
            # Check for errors in the payload
            if isinstance(payload, dict) and "error" in payload:
                return {"error": f"MCP query error: {payload['error']}"}

            # Extract data from payload
            columns = payload.get("columns", []) if isinstance(payload, dict) else []
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}

            print(f"--- SAMPLING-DEBUG: Extracted - Columns: {len(columns)}, Rows: {len(rows)} ---")
            print(f"--- SAMPLING-DEBUG: Column names: {columns} ---")
            
            if not columns or not rows:
                print(f"--- SAMPLING-DEBUG: Empty results - Columns empty: {not columns}, Rows empty: {not rows} ---")
                print(f"--- SAMPLING-DEBUG: Full payload: {payload} ---")
                return {"error": "Query returned no data or an invalid format."}

            # Create DataFrame and generate schema
            try:
                sample_df = pd.DataFrame(rows, columns=columns)
                print(f"--- SAMPLING-DEBUG: Created DataFrame with shape: {sample_df.shape} ---")
                
                spark_schema = get_spark_schema_for_df(sample_df)
                schema_ddl = generate_ddl_from_spark_schema("sampled_data", spark_schema)
                
                row_dicts = sample_df.to_dict(orient="records")
                print(f"--- SAMPLING-DEBUG: Successfully created {len(row_dicts)} row dictionaries ---")

                return {
                    "schema": schema_ddl,
                    "rows": row_dicts,
                    "meta": {
                        "source_type": database_type,
                        "mcp_server": mcp_config["mcp_url"],
                        "customer_id": mcp_config.get("customer_id"),
                        "sampled_rows": len(row_dicts),
                        "columns": columns,
                        "execution_time_ms": meta.get("execution_time_ms"),
                        "database_type": database_type
                    }
                }
                
            except Exception as e:
                print(f"--- SAMPLING-DEBUG: DataFrame creation failed: {e} ---")
                return {"error": f"Failed to process query results: {e}"}

    except Exception as e:
        print(f"--- SAMPLING-DEBUG: Overall sampling failed: {e} ---")
        traceback.print_exc()
        
        def flatten_exceptions(ex):
            out = []
            if hasattr(ex, "exceptions") and ex.exceptions:
                for sub_ex in ex.exceptions:
                    out.extend(flatten_exceptions(sub_ex))
            else:
                out.append(f"{type(ex).__name__}: {ex}")
            return out
        
        error_message = " | ".join(flatten_exceptions(e))
        return {"error": f"MCP sampling failed: {error_message}", "trace": traceback.format_exc()}

def _clean_server_response_string(text: str) -> str:
    """Clean the raw string response from an older mcp-server to make it parsable"""
    text = re.sub(r"Decimal\('([^']*)'\)", r"\1", text)
    text = re.sub(r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)", r"'\1-\2-\3'", text)
    return text

# --- Main Entry Point ---

async def sample_data_from_source(
    source_type: str,
    path: Optional[str] = None,
    query: Optional[str] = None,
    sample_size: int = 10,
    mcp_config: Optional[Dict[str, str]] = None,
    database_type: str = "postgresql"
) -> Dict[str, Any]:
    """
    A hybrid sampling agent that routes requests to the appropriate data source handler.
    - Supports multiple database types via MCP connection
    - Supports CSV for local file sampling
    """
    if source_type in ["postgresql", "mysql", "mariadb", "sqlite", "mssql", "oracle"]:
        if not mcp_config:
            return {"error": "MCP configuration (mcp_config) is required for database sources."}
        if not query:
            return {"error": "A SQL query is required for database sources."}
        
        return await _sample_from_database_mcp(
            query=query,
            sample_size=sample_size,
            mcp_config=mcp_config,
            database_type=source_type
        )
    
    elif source_type == "csv":
        if not path or not os.path.exists(path):
            return {"error": f"CSV file not found at path: {path}"}
        try:
            df = pd.read_csv(path)
            if df.empty:
                return {"error": "CSV file is empty."}
            
            sample_n = min(sample_size, len(df))
            df_sampled = df.sample(n=sample_n, random_state=42)

            table_name = os.path.splitext(os.path.basename(path))[0]
            spark_schema = get_spark_schema_for_df(df)
            schema_ddl = generate_ddl_from_spark_schema(table_name, spark_schema)

            return {
                "schema": schema_ddl,
                "rows": df_sampled.to_dict('records'),
                "meta": {
                    "source_type": "csv",
                    "total_rows": len(df),
                    "sampled_rows": len(df_sampled),
                    "columns": list(df.columns)
                }
            }
        except Exception as e:
            return {"error": f"CSV sampling failed: {e}"}

    else:
        supported_types = ["postgresql", "mysql", "mariadb", "sqlite", "mssql", "oracle", "csv"]
        return {"error": f"Unsupported source type: '{source_type}'. Supported types are: {supported_types}"}

async def test_mcp_connection(mcp_url: str, api_key: str) -> Dict[str, Any]:
    """Tests the MCP connection and lists available tools"""
    try:
        connection_manager = MCPConnectionManager(mcp_url, api_key)
        
        async with connection_manager.get_session() as session:
            tools_resp = await session.list_tools()
            
            return {
                "status": "connected",
                "available_tools": [tool.name for tool in tools_resp.tools],
                "tool_count": len(tools_resp.tools)
            }
    
    except Exception as e:
        def flatten_exceptions(ex):
            out = []
            if hasattr(ex, "exceptions") and ex.exceptions:
                for sub_ex in ex.exceptions:
                    out.extend(flatten_exceptions(sub_ex))
            else:
                out.append(f"{type(ex).__name__}: {ex}")
            return out
        
        error_message = " | ".join(flatten_exceptions(e))
        return {"error": f"Connection test failed: {error_message}"}

# --- Database-Specific Utilities ---

def get_database_specific_queries() -> Dict[str, Dict[str, str]]:
    """Get database-specific test queries"""
    return {
        "postgresql": {
            "version": "SELECT version()",
            "current_db": "SELECT current_database()",
            "tables": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        },
        "mysql": {
            "version": "SELECT VERSION()",
            "current_db": "SELECT DATABASE()",
            "tables": "SHOW TABLES"
        },
        "mariadb": {
            "version": "SELECT VERSION()",
            "current_db": "SELECT DATABASE()",
            "tables": "SHOW TABLES"
        },
        "sqlite": {
            "version": "SELECT sqlite_version()",
            "current_db": "PRAGMA database_list",
            "tables": "SELECT name FROM sqlite_master WHERE type='table'"
        },
        "mssql": {
            "version": "SELECT @@VERSION",
            "current_db": "SELECT DB_NAME()",
            "tables": "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
        },
        "oracle": {
            "version": "SELECT * FROM v$version WHERE banner LIKE 'Oracle%'",
            "current_db": "SELECT SYS_CONTEXT('USERENV', 'DB_NAME') FROM dual",
            "tables": "SELECT table_name FROM user_tables"
        }
    }

async def test_database_specific_features(
    mcp_config: Dict[str, str], 
    database_type: str
) -> Dict[str, Any]:
    """Test database-specific features and queries"""
    try:
        queries = get_database_specific_queries().get(database_type, {})
        if not queries:
            return {"error": f"No test queries available for database type: {database_type}"}
        
        connection_manager = MCPConnectionManager(
            mcp_url=mcp_config["mcp_url"],
            api_key=mcp_config["api_key"]
        )
        
        results = {}
        async with connection_manager.get_session() as session:
            for test_name, query in queries.items():
                try:
                    result = await session.call_tool("query", {"sql": query})
                    
                    # Parse result
                    if result.content and hasattr(result.content[0], 'text'):
                        payload = json.loads(result.content[0].text)
                        results[test_name] = {
                            "success": True,
                            "data": payload.get("rows", [])[:3],  # First 3 rows
                            "columns": payload.get("columns", [])
                        }
                    else:
                        results[test_name] = {"success": False, "error": "No response"}
                        
                except Exception as e:
                    results[test_name] = {"success": False, "error": str(e)}
        
        return {
            "database_type": database_type,
            "test_results": results,
            "overall_status": "success" if any(r.get("success") for r in results.values()) else "error"
        }
        
    except Exception as e:
        return {"error": f"Database feature testing failed: {e}"}