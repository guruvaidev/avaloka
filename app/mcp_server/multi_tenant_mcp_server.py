"""
Enhanced Multi-Database MCP Server with Context Variables - FIXED AUTHENTICATION
Supports PostgreSQL, MySQL, SQLite, SQL Server, Oracle (via SQLAlchemy) and
MongoDB (via pymongo).

MongoDB notes
-------------
SQLAlchemy has no MongoDB dialect, so Mongo cannot go through create_engine().
Instead, every tool (query / list_tables / describe_table) dispatches to a
pymongo-backed implementation *before* any SQLAlchemy code runs, keeping the
existing SQL path byte-for-byte unchanged.

The `query` tool accepts, for a Mongo connection:
  * `SELECT * FROM <collection> [LIMIT n]`  (what server.py's
    db_tables_to_analysis sends -> translated to a find())
  * `SELECT a,b FROM <collection> [LIMIT n]` (projection)
  * `SHOW COLLECTIONS` / `SHOW TABLES`
  * a JSON spec: {"collection": "...", "filter": {...}, "projection": {...},
                  "sort": [["f",1]], "limit": n, "pipeline": [...]}
Only reads are allowed; aggregation `$out`/`$merge` stages are rejected.
"""

import os
import re
import json
import hashlib
import time
import uuid
import threading
import logging
from decimal import Decimal
from datetime import date, datetime
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager
from contextvars import ContextVar
from urllib.parse import quote_plus, urlparse

import sqlalchemy as sa
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, validator
import asyncio

# ---- Optional MongoDB driver (guarded so the module still imports without it) ----
try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    try:
        from bson import ObjectId
        from bson.decimal128 import Decimal128
    except Exception:  # pragma: no cover
        ObjectId = None
        Decimal128 = None
    _PYMONGO_AVAILABLE = True
except Exception:  # pragma: no cover
    MongoClient = None
    PyMongoError = Exception
    ObjectId = None
    Decimal128 = None
    _PYMONGO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000"))
MONGO_DESCRIBE_SAMPLE = int(os.getenv("MONGO_DESCRIBE_SAMPLE", "50"))

SUPPORTED_DB_TYPES = ['postgresql', 'mysql', 'sqlite', 'mssql', 'oracle', 'mariadb', 'mongodb']

# Context variable to store current customer ID
current_customer_id: ContextVar[Optional[str]] = ContextVar('current_customer_id', default=None)


# =====================================================================
# MongoDB helpers
# =====================================================================
def _require_pymongo():
    if not _PYMONGO_AVAILABLE:
        raise RuntimeError(
            "pymongo is not installed; MongoDB support is unavailable. "
            "Install it with: pip install pymongo dnspython"
        )


def _mongo_client_and_db(uri: str, query_timeout: int = 30):
    """Build a MongoClient and resolve the target database from the URI."""
    _require_pymongo()
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
        connectTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
    )
    parsed = urlparse(uri)
    db_name = (parsed.path or "").lstrip("/").split("?")[0] or None
    if not db_name:
        try:
            db = client.get_default_database()
            if db is not None and db.name:
                return client, db
        except Exception:
            pass
        raise ValueError(
            "MongoDB connection string must include a database name "
            "(e.g. mongodb://host:27017/<db>)."
        )
    return client, client[db_name]


def _bson_jsonable(v):
    """Recursively convert BSON/Mongo values to JSON-serializable python types."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if ObjectId is not None and isinstance(v, ObjectId):
        return str(v)
    if Decimal128 is not None and isinstance(v, Decimal128):
        try:
            return float(v.to_decimal())
        except Exception:
            return str(v)
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _bson_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_bson_jsonable(x) for x in v]
    return str(v)


def _flatten_mongo_docs(docs: List[dict]):
    """Turn a list of (possibly heterogeneous) documents into columns + rows.

    Columns are the ordered union of top-level keys across the sampled docs.
    Nested objects/arrays are JSON-stringified so each cell stays scalar, which
    is what the analysis CSV-materialization path downstream expects.
    """
    columns: List[str] = []
    seen = set()
    for d in docs:
        if isinstance(d, dict):
            for k in d.keys():
                sk = str(k)
                if sk not in seen:
                    seen.add(sk)
                    columns.append(sk)
    rows: List[List[Any]] = []
    for d in docs:
        row: List[Any] = []
        for c in columns:
            val = _bson_jsonable(d.get(c)) if isinstance(d, dict) else None
            if isinstance(val, (dict, list)):
                val = json.dumps(val, default=str)
            row.append(val)
        rows.append(row)
    return columns, rows


def _mongo_type(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "double"
    if isinstance(v, str):
        return "string"
    if ObjectId is not None and isinstance(v, ObjectId):
        return "objectId"
    if isinstance(v, (datetime, date)):
        return "date"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


_PSEUDO_SELECT_RE = re.compile(
    r'^\s*select\s+(?P<cols>.+?)\s+from\s+[`"\[]?(?P<coll>[A-Za-z_][\w.$\-]*)[`"\]]?'
    r'(?:\s+limit\s+(?P<limit>\d+))?\s*;?\s*$',
    re.IGNORECASE | re.DOTALL,
)
_SHOW_RE = re.compile(r'^\s*show\s+(?:collections|tables)\s*;?\s*$', re.IGNORECASE)


def _mongo_result(config, columns, rows) -> dict:
    return {
        "columns": columns,
        "rows": rows,
        "meta": {
            "customer_id": current_customer_id.get(),
            "database_type": "mongodb",
            "row_count": len(rows),
            "row_limit": config.max_rows,
        },
    }


def _mongo_execute_query(config, db, sql: str) -> dict:
    """Execute a read-only query against a Mongo database."""
    s = (sql or "").strip()
    max_rows = config.max_rows
    max_time_ms = max(int(config.query_timeout) * 1000, 1000)
    t0 = time.time()

    # ---- JSON spec form ----
    if s.startswith("{"):
        try:
            spec = json.loads(s)
        except Exception as e:
            return {"error": f"Invalid MongoDB query JSON: {e}"}
        coll = spec.get("collection") or spec.get("table")
        if not coll:
            return {"error": "MongoDB query JSON must include 'collection'."}

        pipeline = spec.get("pipeline")
        if pipeline is not None:
            if not isinstance(pipeline, list):
                return {"error": "'pipeline' must be a list of stages."}
            for stage in pipeline:
                if isinstance(stage, dict) and ({"$out", "$merge"} & set(stage.keys())):
                    return {"error": "Write stages ($out / $merge) are not allowed."}
            cursor = db[coll].aggregate(pipeline, maxTimeMS=max_time_ms)
            docs = list(cursor)[:max_rows]
        else:
            filt = spec.get("filter") or {}
            proj = spec.get("projection")
            sort = spec.get("sort")
            limit = min(int(spec.get("limit") or max_rows), max_rows)
            cursor = db[coll].find(filt, proj)
            if sort:
                cursor = cursor.sort(sort)
            cursor = cursor.limit(limit).max_time_ms(max_time_ms)
            docs = list(cursor)

        cols, rows = _flatten_mongo_docs(docs)
        result = _mongo_result(config, cols, rows)
        result["meta"]["execution_time_ms"] = int((time.time() - t0) * 1000)
        return result

    # ---- SHOW COLLECTIONS / SHOW TABLES ----
    if _SHOW_RE.match(s):
        names = db.list_collection_names()
        return _mongo_result(config, ["collection"], [[n] for n in names])

    # ---- pseudo-SQL SELECT ----
    m = _PSEUDO_SELECT_RE.match(s)
    if not m:
        return {
            "error": (
                "MongoDB backend accepts 'SELECT * FROM <collection> [LIMIT n]', "
                "'SELECT a,b FROM <collection> [LIMIT n]', 'SHOW COLLECTIONS', "
                "or a JSON query {collection, filter, projection, sort, limit, pipeline}."
            )
        }
    coll = m.group("coll")
    cols_raw = m.group("cols").strip()
    limit = int(m.group("limit")) if m.group("limit") else max_rows
    limit = min(limit, max_rows)

    projection = None
    if cols_raw != "*":
        fields = [c.strip().strip('`"[]') for c in cols_raw.split(",") if c.strip()]
        if fields:
            projection = {f: 1 for f in fields}

    cursor = db[coll].find({}, projection).limit(limit).max_time_ms(max_time_ms)
    docs = list(cursor)
    cols, rows = _flatten_mongo_docs(docs)
    result = _mongo_result(config, cols, rows)
    result["meta"]["execution_time_ms"] = int((time.time() - t0) * 1000)
    return result


def _mongo_list_tables(config, db) -> dict:
    tables = []
    for name in db.list_collection_names():
        try:
            doc = db[name].find_one()
            cols = list(doc.keys()) if isinstance(doc, dict) else []
            tables.append({
                "name": name,
                "type": "collection",
                "column_count": len(cols),
                "columns": [
                    {"name": str(k), "type": _mongo_type(doc.get(k))} for k in cols
                ],
            })
        except Exception as e:
            tables.append({
                "name": name,
                "type": "collection",
                "column_count": 0,
                "columns": [],
                "error": str(e),
            })
    return {
        "tables": tables,
        "meta": {
            "customer_id": current_customer_id.get(),
            "database_type": "mongodb",
            "table_count": len(tables),
        },
    }


def _mongo_describe_table(config, db, collection_name: str) -> dict:
    docs = list(db[collection_name].find().limit(MONGO_DESCRIBE_SAMPLE))
    keys: List[str] = []
    seen = set()
    types: Dict[str, str] = {}
    for d in docs:
        if isinstance(d, dict):
            for k, v in d.items():
                sk = str(k)
                if sk not in seen:
                    seen.add(sk)
                    keys.append(sk)
                if sk not in types and v is not None:
                    types[sk] = _mongo_type(v)

    try:
        idx_info = db[collection_name].index_information()
        indexes = [
            {
                "name": idx_name,
                "columns": [k for (k, _d) in idx.get("key", [])],
                "unique": bool(idx.get("unique", False)),
            }
            for idx_name, idx in idx_info.items()
        ]
    except Exception:
        indexes = []

    columns = [
        {
            "name": k,
            "type": types.get(k, "unknown"),
            "nullable": True,
            "default": None,
            "primary_key": (k == "_id"),
        }
        for k in keys
    ]

    return {
        "table_name": collection_name,
        "columns": columns,
        "primary_keys": ["_id"] if "_id" in seen else [],
        "foreign_keys": [],
        "indexes": indexes,
        "meta": {
            "customer_id": current_customer_id.get(),
            "database_type": "mongodb",
            "sampled_docs": len(docs),
        },
    }


# =====================================================================
# Database Configuration Models
# =====================================================================
class DatabaseConnection(BaseModel):
    connection_name: str
    database_type: str  # postgresql, mysql, sqlite, mssql, oracle, mariadb, mongodb
    host: Optional[str] = None
    port: Optional[int] = None
    database_name: str
    username: Optional[str] = None
    password: Optional[str] = None
    connection_params: Optional[Dict[str, Any]] = {}
    connection_string: Optional[str] = None  # allow a full URI (Mongo/Atlas srv)
    query_timeout: int = 30
    max_rows: int = 1000

    @validator('connection_name')
    def validate_connection_name(cls, v):
        if not v or len(v) < 3:
            raise ValueError('connection_name must be at least 3 characters')
        return v.replace(' ', '_').lower()

    @validator('database_type')
    def validate_database_type(cls, v):
        if v.lower() not in SUPPORTED_DB_TYPES:
            raise ValueError(f'database_type must be one of: {SUPPORTED_DB_TYPES}')
        return v.lower()


class DatabaseConfig(BaseModel):
    customer_id: str
    connection_string: str
    database_type: str
    max_rows: int = 1000
    query_timeout: int = 30
    created_at: str
    status: str = "active"
    api_key: str


class CustomerAPI(BaseModel):
    customer_id: str
    api_key: str
    permissions: List[str] = ["read"]
    created_at: str


class ConnectionResponse(BaseModel):
    customer_id: str
    api_key: str
    mcp_endpoint: str
    status: str
    message: str


class ToolCallParams(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolCallRequest(BaseModel):
    method: str = "call_tool"
    params: ToolCallParams


# =====================================================================
# Database Connection Manager
# =====================================================================
class DatabaseConnectionManager:
    """Manages database connections with connection pooling (SQL) and
    lazily-created pymongo databases (MongoDB)."""

    def __init__(self):
        self.engines: Dict[str, Engine] = {}
        self.connection_configs: Dict[str, DatabaseConfig] = {}
        # MongoDB
        self.mongo_clients: Dict[str, Any] = {}
        self.mongo_dbs: Dict[str, Any] = {}

    def build_connection_string(self, conn: DatabaseConnection) -> str:
        """Build connection string for different database types"""

        if conn.database_type == 'sqlite':
            return f"sqlite:///{conn.database_name}"

        elif conn.database_type == 'postgresql':
            password = quote_plus(conn.password) if conn.password else ""
            return f"postgresql://{conn.username}:{password}@{conn.host}:{conn.port}/{conn.database_name}"

        elif conn.database_type in ['mysql', 'mariadb']:
            password = quote_plus(conn.password) if conn.password else ""
            return f"mysql+pymysql://{conn.username}:{password}@{conn.host}:{conn.port}/{conn.database_name}"

        elif conn.database_type == 'mssql':
            password = quote_plus(conn.password) if conn.password else ""
            return f"mssql+pyodbc://{conn.username}:{password}@{conn.host}:{conn.port}/{conn.database_name}?driver=ODBC+Driver+17+for+SQL+Server"

        elif conn.database_type == 'oracle':
            password = quote_plus(conn.password) if conn.password else ""
            return f"oracle+cx_oracle://{conn.username}:{password}@{conn.host}:{conn.port}/{conn.database_name}"

        elif conn.database_type == 'mongodb':
            if conn.connection_string:
                return conn.connection_string
            user = quote_plus(conn.username) if conn.username else ""
            password = quote_plus(conn.password) if conn.password else ""
            auth = f"{user}:{password}@" if user else ""
            port = conn.port or 27017
            params = conn.connection_params or {}
            auth_source = params.get("authSource") or params.get("auth_source")
            q = f"?authSource={auth_source}" if auth_source else ""
            return f"mongodb://{auth}{conn.host}:{port}/{conn.database_name}{q}"

        else:
            raise ValueError(f"Unsupported database type: {conn.database_type}")

    def create_engine_for_connection(self, conn: DatabaseConnection) -> Engine:
        """Create SQLAlchemy engine with appropriate settings (SQL types only)."""
        if conn.database_type == 'mongodb':
            raise ValueError("MongoDB does not use a SQLAlchemy engine.")

        connection_string = self.build_connection_string(conn)
        engine_kwargs = {'pool_pre_ping': True, 'pool_recycle': 3600}

        if conn.database_type == 'sqlite':
            engine_kwargs['poolclass'] = sa.pool.StaticPool
            engine_kwargs['connect_args'] = {
                'timeout': conn.query_timeout,
                'check_same_thread': False
            }
        else:
            engine_kwargs['poolclass'] = QueuePool
            engine_kwargs['pool_size'] = 5
            engine_kwargs['max_overflow'] = 10
            engine_kwargs['connect_args'] = {
                'connect_timeout': conn.query_timeout,
                **conn.connection_params
            }

            if conn.database_type == 'postgresql':
                engine_kwargs['connect_args']['options'] = f'-c statement_timeout={conn.query_timeout * 1000}'
            elif conn.database_type in ['mysql', 'mariadb']:
                engine_kwargs['connect_args'].update({
                    'charset': 'utf8mb4',
                    'autocommit': True
                })

        return create_engine(connection_string, **engine_kwargs)

    def test_connection(self, conn: DatabaseConnection) -> Dict[str, Any]:
        """Test database connection and return metadata"""

        # ---- MongoDB ----
        if conn.database_type == 'mongodb':
            uri = self.build_connection_string(conn)
            client = None
            try:
                client, db = _mongo_client_and_db(uri, conn.query_timeout)
                client.admin.command("ping")
                version = ""
                try:
                    version = client.server_info().get("version", "")
                except Exception:
                    pass
                try:
                    table_count = len(db.list_collection_names())
                except Exception:
                    table_count = 0
                return {
                    "success": True,
                    "database": db.name,
                    "user": conn.username or "mongodb",
                    "version": version or "Unknown",
                    "table_count": table_count,
                    "database_type": "mongodb",
                }
            except Exception as e:
                return {"success": False, "error": str(e), "error_type": type(e).__name__}
            finally:
                if client is not None:
                    client.close()

        # ---- SQL ----
        try:
            engine = self.create_engine_for_connection(conn)

            with engine.connect() as connection:
                if conn.database_type == 'postgresql':
                    result = connection.execute(text("SELECT version(), current_database(), current_user"))
                    version, database, user = result.fetchone()
                elif conn.database_type in ['mysql', 'mariadb']:
                    result = connection.execute(text("SELECT VERSION(), DATABASE(), USER()"))
                    version, database, user = result.fetchone()
                elif conn.database_type == 'sqlite':
                    result = connection.execute(text("SELECT sqlite_version()"))
                    version = result.fetchone()[0]
                    database = conn.database_name
                    user = "sqlite"
                elif conn.database_type == 'mssql':
                    result = connection.execute(text("SELECT @@VERSION, DB_NAME(), SUSER_SNAME()"))
                    version, database, user = result.fetchone()
                elif conn.database_type == 'oracle':
                    result = connection.execute(text("SELECT * FROM v$version WHERE banner LIKE 'Oracle%'"))
                    version = result.fetchone()[0]
                    result = connection.execute(text("SELECT SYS_CONTEXT('USERENV', 'DB_NAME'), USER FROM dual"))
                    database, user = result.fetchone()

                inspector = inspect(engine)
                tables = inspector.get_table_names()

            return {
                "success": True,
                "database": database,
                "user": user,
                "version": version.split()[0] if version else "Unknown",
                "table_count": len(tables),
                "database_type": conn.database_type
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        finally:
            if 'engine' in locals():
                engine.dispose()

    def register_connection(self, conn: DatabaseConnection, api_key: str) -> str:
        """Register a new database connection"""

        test_result = self.test_connection(conn)
        if not test_result["success"]:
            raise ValueError(f"Connection test failed: {test_result['error']}")

        connection_string = self.build_connection_string(conn)

        config = DatabaseConfig(
            customer_id=conn.connection_name,
            connection_string=connection_string,
            database_type=conn.database_type,
            max_rows=conn.max_rows,
            query_timeout=conn.query_timeout,
            created_at=datetime.now().isoformat(),
            status="active",
            api_key=api_key
        )

        if conn.database_type == 'mongodb':
            # Mongo client is created lazily on first use (get_mongo_db).
            self.connection_configs[conn.connection_name] = config
        else:
            engine = self.create_engine_for_connection(conn)
            self.engines[conn.connection_name] = engine
            self.connection_configs[conn.connection_name] = config

        logger.info(f"Registered connection: {conn.connection_name} ({conn.database_type})")
        return conn.connection_name

    def get_engine(self, customer_id: str) -> Engine:
        """Get database engine for customer (SQL only)."""
        if customer_id not in self.engines:
            raise KeyError(f"No connection found for customer: {customer_id}")
        return self.engines[customer_id]

    def get_mongo_db(self, customer_id: str):
        """Get (lazily creating) the pymongo Database for a Mongo customer."""
        if customer_id in self.mongo_dbs:
            return self.mongo_dbs[customer_id]
        config = self.get_config(customer_id)
        if config.database_type != "mongodb":
            raise KeyError(f"Customer {customer_id} is not a MongoDB connection.")
        client, db = _mongo_client_and_db(config.connection_string, config.query_timeout)
        self.mongo_clients[customer_id] = client
        self.mongo_dbs[customer_id] = db
        return db

    def is_mongo(self, customer_id: str) -> bool:
        cfg = self.connection_configs.get(customer_id)
        return bool(cfg and cfg.database_type == "mongodb")

    def get_config(self, customer_id: str) -> DatabaseConfig:
        """Get database configuration for customer"""
        if customer_id not in self.connection_configs:
            raise KeyError(f"No configuration found for customer: {customer_id}")
        return self.connection_configs[customer_id]


# Global instances
db_manager = DatabaseConnectionManager()
CUSTOMER_API_KEYS: Dict[str, CustomerAPI] = {}

# Persist/reload the registry from the JSON file the onboarding service writes,
# so registered customers survive a restart.
CUSTOMERS_FILE = os.getenv("CUSTOMERS_FILE", "customers.json")
_customers_file_lock = threading.Lock()

# Security
security = HTTPBearer(auto_error=False)


def extract_customer_from_api_key(api_key: str) -> Optional[str]:
    """Extract customer_id from API key"""
    for customer_id, customer_api in CUSTOMER_API_KEYS.items():
        if customer_api.api_key == api_key:
            logger.info(f"Found customer {customer_id} for API key {api_key[:8]}...")
            return customer_id
    logger.warning(f"No customer found for API key {api_key[:8]}...")
    return None


def get_current_customer(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """Validate API key and return customer_id"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    api_key = credentials.credentials
    customer_id = extract_customer_from_api_key(api_key)

    if not customer_id:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return customer_id


def _jsonable(v):
    """Convert database types to JSON-serializable formats"""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, uuid.UUID):
        return str(v)
    return v


def _wrap_with_limit(sql: str, limit: int, database_type: str) -> str:
    """Add LIMIT clause with database-specific syntax"""
    s = sql.strip().rstrip(";")

    if database_type in ['postgresql', 'mysql', 'mariadb', 'sqlite']:
        return f"SELECT * FROM ({s}) AS _mcp_sub LIMIT {limit}"
    elif database_type == 'mssql':
        if 'SELECT' in s.upper():
            return s.replace('SELECT', f'SELECT TOP {limit}', 1)
        else:
            return f"SELECT TOP {limit} * FROM ({s}) AS _mcp_sub"
    elif database_type == 'oracle':
        return f"SELECT * FROM ({s}) WHERE ROWNUM <= {limit}"
    else:
        return f"SELECT * FROM ({s}) AS _mcp_sub LIMIT {limit}"


# Initialize FastMCP
mcp = FastMCP("avaloka-multi-database")


@mcp.tool()
def query(sql: str) -> dict:
    """Execute a read-only query against customer's database"""
    customer_id = current_customer_id.get()
    logger.info(f"MCP Tool: query called for customer: {customer_id}")

    if not customer_id:
        return {"error": "Customer ID not provided in context"}

    try:
        config = db_manager.get_config(customer_id)
    except KeyError as e:
        return {"error": str(e), "meta": {"customer_id": customer_id}}

    # ---- MongoDB dispatch (before the SQL read-only gate) ----
    if config.database_type == "mongodb":
        try:
            db = db_manager.get_mongo_db(customer_id)
            return _mongo_execute_query(config, db, sql)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Mongo query failed: {error_msg}")
            return {"error": error_msg, "meta": {"customer_id": customer_id}}

    # ---- SQL path (unchanged) ----
    first_word = sql.strip().split()[0].lower() if sql.strip() else ""
    if first_word not in {"select", "with", "show", "describe", "explain"}:
        return {"error": "Only read-only queries are allowed (SELECT, WITH, SHOW, DESCRIBE, EXPLAIN)"}

    try:
        engine = db_manager.get_engine(customer_id)

        logger.info(f"Using {config.database_type} database for customer {customer_id}")

        safe_sql = _wrap_with_limit(sql, config.max_rows, config.database_type)
        qhash = hashlib.sha256(f"{customer_id}:{sql}".encode()).hexdigest()[:12]
        t0 = time.time()

        with engine.connect() as conn:
            if config.database_type == 'postgresql':
                conn.execute(text(f"SET statement_timeout = '{config.query_timeout}s'"))
            elif config.database_type in ['mysql', 'mariadb']:
                conn.execute(text(f"SET SESSION max_execution_time = {config.query_timeout * 1000}"))

            result = conn.execute(text(safe_sql))
            rows = result.fetchall()
            cols = list(result.keys()) if result.keys() else []

        safe_rows = [[_jsonable(cell) for cell in row] for row in rows]
        execution_time_ms = int((time.time() - t0) * 1000)

        logger.info(f"Query executed successfully. Rows: {len(safe_rows)}, Columns: {cols}")

        return {
            "columns": cols,
            "rows": safe_rows,
            "meta": {
                "customer_id": customer_id,
                "database_type": config.database_type,
                "query_hash": qhash,
                "execution_time_ms": execution_time_ms,
                "row_count": len(safe_rows),
                "row_limit": config.max_rows
            }
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Query failed: {error_msg}")
        return {"error": error_msg, "meta": {"customer_id": customer_id}}


@mcp.tool()
def list_tables() -> dict:
    """List available tables/collections in customer's database"""
    customer_id = current_customer_id.get()
    logger.info(f"MCP Tool: list_tables called for customer: {customer_id}")

    if not customer_id:
        return {"error": "Customer ID not provided in context"}

    try:
        config = db_manager.get_config(customer_id)
    except KeyError as e:
        return {"error": str(e)}

    # ---- MongoDB ----
    if config.database_type == "mongodb":
        try:
            db = db_manager.get_mongo_db(customer_id)
            return _mongo_list_tables(config, db)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)}"}

    # ---- SQL path (unchanged) ----
    try:
        engine = db_manager.get_engine(customer_id)

        inspector = inspect(engine)
        tables = []

        table_names = inspector.get_table_names()
        for table_name in table_names:
            try:
                columns = inspector.get_columns(table_name)
                tables.append({
                    "name": table_name,
                    "type": "table",
                    "column_count": len(columns),
                    "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns]
                })
            except Exception as e:
                tables.append({
                    "name": table_name,
                    "type": "table",
                    "column_count": 0,
                    "columns": [],
                    "error": str(e)
                })

        try:
            view_names = inspector.get_view_names()
            for view_name in view_names:
                tables.append({
                    "name": view_name,
                    "type": "view",
                    "column_count": 0,
                    "columns": []
                })
        except Exception:
            pass

        return {
            "tables": tables,
            "meta": {
                "customer_id": customer_id,
                "database_type": config.database_type,
                "table_count": len(tables)
            }
        }

    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}


@mcp.tool()
def describe_table(table_name: str) -> dict:
    """Get detailed information about a specific table/collection"""
    customer_id = current_customer_id.get()
    logger.info(f"MCP Tool: describe_table called for customer: {customer_id}, table: {table_name}")

    if not customer_id:
        return {"error": "Customer ID not provided in context"}

    try:
        config = db_manager.get_config(customer_id)
    except KeyError as e:
        return {"error": str(e)}

    # ---- MongoDB ----
    if config.database_type == "mongodb":
        try:
            db = db_manager.get_mongo_db(customer_id)
            return _mongo_describe_table(config, db, table_name)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)}"}

    # ---- SQL path (unchanged) ----
    try:
        engine = db_manager.get_engine(customer_id)

        inspector = inspect(engine)

        columns = inspector.get_columns(table_name)

        try:
            pk_constraint = inspector.get_pk_constraint(table_name)
            primary_keys = pk_constraint.get('constrained_columns', [])
        except Exception:
            primary_keys = []

        try:
            foreign_keys = inspector.get_foreign_keys(table_name)
        except Exception:
            foreign_keys = []

        try:
            indexes = inspector.get_indexes(table_name)
        except Exception:
            indexes = []

        return {
            "table_name": table_name,
            "columns": [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": str(col["default"]) if col.get("default") is not None else None,
                    "primary_key": col["name"] in primary_keys
                }
                for col in columns
            ],
            "primary_keys": primary_keys,
            "foreign_keys": [
                {
                    "constrained_columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"]
                }
                for fk in foreign_keys
            ],
            "indexes": [
                {
                    "name": idx["name"],
                    "columns": idx["column_names"],
                    "unique": idx.get("unique", False)
                }
                for idx in indexes
            ],
            "meta": {
                "customer_id": customer_id,
                "database_type": config.database_type
            }
        }

    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}


# Create FastAPI app
app = FastAPI(
    title="Avaloka Multi-Database MCP Server",
    description="Multi-database MCP server with UI integration support",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the customer-onboarding routes onto this server so registration and
# /call_tool share one host. The standalone customer_dbs app on :8011 still works.
from app.mcp_server.customer_dbs import router as onboarding_router

app.include_router(onboarding_router)


def generate_api_key(customer_id: str) -> str:
    """Generate secure API key for customer"""
    timestamp = str(int(datetime.now().timestamp()))
    raw_key = f"{customer_id}:{timestamp}:{os.urandom(16).hex()}"
    return f"ak_{hashlib.sha256(raw_key.encode()).hexdigest()[:32]}"


def _register_in_memory(config: DatabaseConfig) -> None:
    """Add a customer's engine/mongo-db, config, and API key to the registry."""
    if config.database_type == "mongodb":
        # Mongo client/db is created lazily on first use (get_mongo_db).
        db_manager.connection_configs[config.customer_id] = config
    else:
        engine = create_engine(
            config.connection_string, pool_pre_ping=True, pool_recycle=3600
        )
        db_manager.engines[config.customer_id] = engine
        db_manager.connection_configs[config.customer_id] = config

    CUSTOMER_API_KEYS[config.customer_id] = CustomerAPI(
        customer_id=config.customer_id,
        api_key=config.api_key,
        created_at=config.created_at,
    )


def load_registered_customers() -> int:
    """Rehydrate db_manager + CUSTOMER_API_KEYS from CUSTOMERS_FILE on startup.

    Engines (and Mongo clients) are created lazily so an unreachable database
    does not block startup — the error surfaces on the actual tool call.
    """
    path = CUSTOMERS_FILE
    if not os.path.exists(path):
        logger.info(f"No customers file at '{path}'; starting with empty registry.")
        return 0
    try:
        with open(path, "r") as f:
            customers = json.load(f)
    except Exception as e:
        logger.error(f"Could not read customers file '{path}': {e}")
        return 0

    loaded = 0
    for customer_id, record in customers.items():
        if record.get("status") == "inactive":
            continue
        api_key = record.get("api_key")
        connection_string = record.get("connection_string")
        if not api_key or not connection_string:
            logger.warning(
                f"Skipping '{customer_id}': missing api_key or connection_string."
            )
            continue
        try:
            config = DatabaseConfig(
                customer_id=customer_id,
                connection_string=connection_string,
                database_type=record.get("database_type", ""),
                max_rows=record.get("max_rows", 1000),
                query_timeout=record.get("query_timeout", 30),
                created_at=record.get("created_at", datetime.now().isoformat()),
                status=record.get("status", "active"),
                api_key=api_key,
            )
            _register_in_memory(config)
            loaded += 1
        except Exception as e:
            logger.error(f"Failed to load customer '{customer_id}': {e}")

    logger.info(f"Loaded {loaded} customer(s) from '{path}' into MCP registry.")
    return loaded


def _persist_customer(config: DatabaseConfig) -> None:
    """Write-through a registration to CUSTOMERS_FILE so it survives a restart."""
    with _customers_file_lock:
        data: Dict[str, Any] = {}
        if os.path.exists(CUSTOMERS_FILE):
            try:
                with open(CUSTOMERS_FILE, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read '{CUSTOMERS_FILE}' before write: {e}")
                data = {}

        record = data.get(config.customer_id, {})
        record.update({
            "customer_id": config.customer_id,
            "connection_string": config.connection_string,
            "database_type": config.database_type,
            "api_key": config.api_key,
            "created_at": config.created_at,
            "status": config.status,
            "max_rows": config.max_rows,
            "query_timeout": config.query_timeout,
        })
        data[config.customer_id] = record

        try:
            with open(CUSTOMERS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist customer '{config.customer_id}': {e}")


@app.on_event("startup")
async def _startup_load_customers() -> None:
    load_registered_customers()


@app.post("/admin/customers")
async def create_customer_database(config: DatabaseConfig):
    """Create/update customer database configuration from onboarding service"""
    try:
        logger.info(f"Registering customer: {config.customer_id}")

        # Test the connection before registering so a bad DSN fails here, not on /call_tool.
        if config.database_type == "mongodb":
            client = None
            try:
                client, _db = _mongo_client_and_db(config.connection_string, config.query_timeout)
                client.admin.command("ping")
                logger.info(f"MongoDB connection test successful for {config.customer_id}")
            finally:
                if client is not None:
                    client.close()
        else:
            test_engine = create_engine(config.connection_string, pool_pre_ping=True)
            try:
                with test_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info(f"Database connection test successful for {config.customer_id}")
            finally:
                test_engine.dispose()

        _register_in_memory(config)
        _persist_customer(config)

        logger.info(f"Successfully registered customer {config.customer_id}")

        return {"customer_id": config.customer_id, "status": "registered"}

    except Exception as e:
        logger.error(f"Registration failed for {config.customer_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Database connection failed: {e}")


@app.post("/connect", response_model=ConnectionResponse)
async def connect_database(connection: DatabaseConnection):
    """Connect and register a new database connection"""
    try:
        api_key = generate_api_key(connection.connection_name)

        customer_id = db_manager.register_connection(connection, api_key)

        CUSTOMER_API_KEYS[customer_id] = CustomerAPI(
            customer_id=customer_id,
            api_key=api_key,
            permissions=["read"],
            created_at=datetime.now().isoformat()
        )

        mcp_endpoint = f"http://localhost:8080/sse"

        logger.info(f"Successfully connected database: {customer_id} ({connection.database_type})")

        return ConnectionResponse(
            customer_id=customer_id,
            api_key=api_key,
            mcp_endpoint=mcp_endpoint,
            status="active",
            message="Connection successful and registered."
        )

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/connections")
async def list_connections():
    """List all registered database connections"""
    connections = []

    for customer_id, config in db_manager.connection_configs.items():
        connections.append({
            "customer_id": customer_id,
            "database_type": config.database_type,
            "status": config.status,
            "created_at": config.created_at,
            "max_rows": config.max_rows,
            "has_api_key": customer_id in CUSTOMER_API_KEYS
        })

    return {
        "connections": connections,
        "total_count": len(connections)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "connections_registered": len(db_manager.connection_configs),
        "sql_engines": len(db_manager.engines),
        "mongo_connections": sum(
            1 for c in db_manager.connection_configs.values() if c.database_type == "mongodb"
        ),
        "api_keys_issued": len(CUSTOMER_API_KEYS),
        "pymongo_available": _PYMONGO_AVAILABLE,
        "supported_databases": SUPPORTED_DB_TYPES,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/call_tool")
async def call_tool(request: ToolCallRequest, customer_id: str = Depends(get_current_customer)):
    """Tool call endpoint with proper authentication"""

    # Set customer context for MCP tools
    token = current_customer_id.set(customer_id)

    try:
        tool_name = request.params.name
        arguments = request.params.arguments

        logger.info(f"Calling tool '{tool_name}' for customer '{customer_id}' with args: {arguments}")
        if tool_name == "query":
            if "sql" not in arguments:
                raise HTTPException(status_code=400, detail="Missing 'sql' argument for query tool")
            result = query(arguments["sql"])
        elif tool_name == "list_tables":
            result = list_tables()
        elif tool_name == "describe_table":
            if "table_name" not in arguments:
                raise HTTPException(status_code=400, detail="Missing 'table_name' argument for describe_table tool")
            result = describe_table(arguments["table_name"])
        else:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found. Available tools: query, list_tables, describe_table")

        logger.info(f"Tool '{tool_name}' completed successfully for customer '{customer_id}'")
        return result

    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}' for customer '{customer_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Reset the context
        current_customer_id.reset(token)


if __name__ == "__main__":
    import uvicorn

    print("Starting Avaloka Multi-Database MCP Server...")
    print("Supported databases: PostgreSQL, MySQL, MariaDB, SQLite, SQL Server, Oracle, MongoDB")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("MCP_SERVER_PORT", "8080")),
        log_level="info"
    )